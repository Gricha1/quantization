# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ['register', "get_adv_estimator_fn", "AdvantageEstimator"]

from collections import defaultdict
from enum import Enum
from typing import Optional

import numpy as np
import torch

import verl.utils.torch_functional as verl_F

ADV_ESTIMATOR_REGISTRY = {}

def register_adv_est(name_or_enum):
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """
    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}")
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn
    return decorator

def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]

class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    VTRACE = "vtrace"


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError

@register_adv_est(AdvantageEstimator.GAE) # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO) # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

@register_adv_est(AdvantageEstimator.GRPO_PASSK) # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config = None,
    **kwargs,
):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (dict) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}.")
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages

@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE) # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor,
                                                           epsilon: float = 1e-6, config=None, **kwargs):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores

@register_adv_est(AdvantageEstimator.RLOO) # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray,
                                   epsilon: float = 1e-6, config=None, **kwargs):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

@register_adv_est(AdvantageEstimator.OPO) # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6,
                                  config=None, **kwargs):
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.tensor(id2score[idx])
                len_tensor = torch.tensor(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS) # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config=None, **kwargs):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns

@register_adv_est(AdvantageEstimator.REMAX) # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor, config=None, **kwargs):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns

@register_adv_est(AdvantageEstimator.VTRACE) # or simply: @register_adv_est("vtrace")
def compute_vtrace_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    old_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    gamma: torch.Tensor,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
    dones: Optional[torch.Tensor] = None,
    recurrence: int = 0,
):
    """
    Compute V-trace advantage and returns for off-policy correction.
    
    Based on: "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures"
    https://arxiv.org/abs/1802.01561
    
    Implementation follows Sample Factory approach:
    - No clipping on delta_t, v_trace_t, or returns
    - Advantages computed inside loop: adv = ρ_t * (r_t + γ * v_{t+1} - V_t)
    - Advantage normalization: same as GAE/PPO — ``verl_F.masked_whiten(advantages, response_mask)``
    
    V-trace corrects value estimates when using off-policy data by applying truncated importance sampling.
    
    Mathematical formulation:
        ρ_t = min(ρ̄, π(a_t|s_t) / μ(a_t|s_t))
        c_t = min(c̄, π(a_t|s_t) / μ(a_t|s_t))
        δ_t = r_t + γV(s_{t+1}) - V(s_t)
        v_t = V(s_t) + ρ_t δ_t + γ c_t (v_{t+1} - V(s_{t+1}))
        adv_t = ρ_t * (r_t + γ * v_{t+1} - V_t)
    
    Done flags handling (implemented):
       - If 'dones' parameter is provided, uses: not_done_gamma = (1.0 - dones) * gamma
       - This means if done=True, gamma is multiplied by 0 (no future rewards)
       - Matches Sample Factory implementation
    
    Potential future improvements (matching Sample Factory more closely):
    
    4) Trajectory segmentation:
       - Sample Factory uses fixed-length segments (recurrence parameter)
       - Current VERL: processes full sequences
       - Proposal: Add optional 'recurrence' parameter to process sequences in segments
         This would require restructuring the loop to handle segments of fixed length
    
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length) - rewards at each timestep
        values: `(torch.Tensor)`
            shape: (bs, response_length) - value estimates V(s_t)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length) - mask for valid tokens
        old_log_probs: `(torch.Tensor)`
            shape: (bs, response_length) - log probs under current policy π
        rollout_log_probs: `(torch.Tensor)`
            shape: (bs, response_length) - log probs under behavior policy μ
        gamma: `(float or torch.Tensor)`
            discount factor (will be converted to tensor if float)
        rho_bar: `(float)`
            truncation threshold for importance weights (default: 1.0)
        c_bar: `(float)`
            truncation threshold for c weights (default: 1.0, should be <= rho_bar)
        dones: `(torch.Tensor, optional)`
            shape: (bs, response_length) - done flags indicating episode termination
            If provided, uses not_done_gamma = (1.0 - dones) * gamma to handle episode boundaries
            When done=True, future rewards are not discounted (gamma becomes 0)
        recurrence: `(int, optional)`
            V-trace segmentation length (Sample Factory-style recurrence). 0 disables segmentation and uses the full
            trajectory. When > 0, V-trace is computed within segments of length `recurrence` with bootstrap at segment
            boundaries so that corrections do not propagate across segments.
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length) - V-trace advantages
        returns: `(torch.Tensor)`
            shape: (bs, response_length) - V-trace returns
        vtrace_stats: `(dict)`
            Dictionary with V-trace statistics for logging:
            - "vtrace/c_t_product_mean": mean of cumulative c_t product
            - "vtrace/c_t_product_min": min of cumulative c_t product
            - "vtrace/c_t_product_max": max of cumulative c_t product
            - "vtrace/v_target_mean": mean of V-trace target values
            - "vtrace/v_target_min": min of V-trace target values
            - "vtrace/v_target_max": max of V-trace target values
            - "vtrace/v_target_std": std of V-trace target values
    """
    with torch.no_grad():
        # Ensure gamma is a tensor
        if isinstance(gamma, (int, float)):
            gamma = torch.tensor(gamma, device=values.device, dtype=values.dtype)
        
        # Compute importance ratios: π(a_t|s_t) / μ(a_t|s_t) = exp(log π - log μ)
        # Match Sample Factory (learner.py): clamp ratio only (no log clamp there)
        log_ratio = old_log_probs - rollout_log_probs
        importance_ratio = torch.exp(log_ratio)
        importance_ratio = torch.clamp(importance_ratio, min=0.05, max=20.0)

        # Truncated importance weights
        rho_t = torch.clamp(importance_ratio, max=rho_bar)
        c_t = torch.clamp(importance_ratio, max=c_bar)

        # Softer bounds: allow rho_t to be smaller (down to 0.01) to allow policy changes
        # Only cap maximum if rho_bar is too high
        if rho_bar > 2.0:
            rho_t = torch.clamp(rho_t, min=0.01, max=2.0)
        else:
            # Allow rho_t to be as small as needed (down to 0.01) for policy changes
            rho_t = torch.clamp(rho_t, min=0.01)
        
        # Initialize V-trace values and advantages
        gen_len = token_level_rewards.shape[-1]
        v_trace_values = torch.zeros_like(values)
        advantages = torch.zeros_like(values)
        
        # Optionally track statistics of cumulative c_t products for debugging / analysis.
        # We keep this inexpensive: compute per-sequence product over time, then log summary
        # stats (mean / min / max) once per call.
        cum_c_prod = None
        seg_c_prod_list = []  # list[Tensor(bs,)] cumulative c_t products per segment
        
        # Handle done flags (matching Sample Factory)
        # If dones is provided, use not_done_gamma = (1.0 - dones) * gamma
        # This ensures that when done=True, future rewards are not discounted (gamma becomes 0)
        if dones is not None:
            # Ensure dones has the same shape as values
            if dones.shape != values.shape:
                raise ValueError(f"dones shape {dones.shape} must match values shape {values.shape}")
            # Convert to float if needed
            if dones.dtype != torch.float32 and dones.dtype != torch.float64:
                dones = dones.float()
            not_done = 1.0 - dones
            # not_done_gamma will be computed per timestep in the loop
        else:
            not_done = None

        # Compute V-trace backwards (recursive form)
        # Original IMPALA V-trace (Espeholt et al., 2018), Eq. (9):
        #
        #   v_s = V(x_s) + Σ_{t=s}^{T-1} γ^{t-s} (∏_{i=s}^{t-1} c_i) ρ_t δ_t
        #
        # где:
        #   δ_t = r_t + γ V(x_{t+1}) - V(x_t)
        #
        # Эквивалентная рекурсивная форма:
        #
        #   v_t = V(x_t) + ρ_t δ_t + γ c_t (v_{t+1} - V(x_{t+1}))
        #
        # Мы реализуем именно эту рекурсивную форму, чтобы корректно
        # учитывать произведение c_t по времени (через γ c_t (v_{t+1} - V_{t+1})).
        # Маска применяется в конце, как и в GAE.
        # 
        # Following Sample Factory implementation:
        # - No clipping on delta_t or v_trace_t (removed for closer match to Sample Factory)
        # - Advantages computed inside loop: adv = ρ_t * (r_t + γ * v_{t+1} - V_t)
        # - Done flags handling: not_done_gamma = (1.0 - dones) * gamma
        def _process_range(t_start: int, t_end: int) -> None:
            """Compute V-trace within [t_start, t_end) (t_end exclusive).

            For recurrence-style segmentation, we bootstrap at the segment boundary by setting:
                v_{t_end} = V(x_{t_end})
            which makes (v_{t_end} - V(x_{t_end})) = 0 and prevents corrections from propagating across segments.
            """
            if t_end < gen_len:
                v_trace_values[:, t_end] = values[:, t_end]

            seg_cum_c_prod = None  # per-segment cumulative c_t product (per sequence in batch)
            for t in reversed(range(t_start, t_end)):
                # Get current values / rewards
                v_t = values[:, t]  # V(x_t)
                r_t = token_level_rewards[:, t]
                rho = rho_t[:, t]
                c = c_t[:, t]

                # Compute not_done_gamma if dones are provided (matching Sample Factory)
                if not_done is not None:
                    not_done_gamma = not_done[:, t] * gamma
                else:
                    not_done_gamma = gamma

                # For next timestep:
                #   - v_next: baseline V(x_{t+1})
                #   - v_trace_next: corrected value v_{t+1}
                if t < gen_len - 1:
                    v_next = values[:, t + 1]
                    v_trace_next = v_trace_values[:, t + 1]
                else:
                    v_next = torch.zeros_like(v_t)
                    v_trace_next = torch.zeros_like(v_t)

                # TD error: δ_t = r_t + γ V(x_{t+1}) - V(x_t)
                # Use not_done_gamma if dones are provided
                delta_t = r_t + not_done_gamma * v_next - v_t

                # V-trace update (recursive form):
                #   v_t = V(x_t) + ρ_t δ_t + γ c_t (v_{t+1} - V(x_{t+1}))
                # Use not_done_gamma if dones are provided (matching Sample Factory)
                # No clipping (matching Sample Factory implementation)
                v_trace_t = v_t + rho * delta_t + not_done_gamma * c * (v_trace_next - v_next)

                # Store V-trace value (mask will be applied later)
                v_trace_values[:, t] = v_trace_t

                # Compute advantage inside loop (matching Sample Factory):
                #   adv = ρ_t * (r_t + γ * v_{t+1} - V_t)
                # Use not_done_gamma if dones are provided
                # This uses the corrected v_{t+1} (v_trace_next) instead of V_{t+1}
                advantages[:, t] = rho * (r_t + not_done_gamma * v_trace_next - v_t)

                # Track cumulative product of c_t for analysis (forward direction).
                # Note: we compute it in the backward loop but conceptually this is:
                #   C_s = ∏_{i=s}^{T-1} c_i
                nonlocal cum_c_prod
                if cum_c_prod is None:
                    cum_c_prod = c.clone()
                else:
                    cum_c_prod = c * cum_c_prod

                # Track per-segment cumulative product as well
                if seg_cum_c_prod is None:
                    seg_cum_c_prod = c.clone()
                else:
                    seg_cum_c_prod = c * seg_cum_c_prod

            # Save per-segment product (one value per sequence in batch)
            if seg_cum_c_prod is not None:
                seg_c_prod_list.append(seg_cum_c_prod)

        # Process either full trajectory or segmented ranges (recurrence)
        rec = int(recurrence) if recurrence is not None else 0
        if rec <= 0 or rec >= gen_len:
            _process_range(0, gen_len)
        else:
            for seg_end in range(gen_len, 0, -rec):
                seg_start = max(0, seg_end - rec)
                _process_range(seg_start, seg_end)

        # Returns are V-trace values
        # Apply mask to returns (similar to GAE - mask is applied after computation)
        returns = v_trace_values * response_mask + values * (1 - response_mask)
        
        # No clipping on returns (matching Sample Factory implementation)

        # Compute statistics for logging (c_t product and V-trace values)
        vtrace_stats = {}
        
        # Statistics for cumulative c_t product
        if cum_c_prod is not None:
            # Mask out invalid tokens when computing stats
            c_prod_mask = response_mask[:, 0] if response_mask.ndim == 2 else response_mask
            # Use absolute to avoid sign issues in case of numerical noise
            c_prod_mean = verl_F.masked_mean(cum_c_prod.abs(), c_prod_mask)
            c_prod_min = (cum_c_prod.abs() * c_prod_mask).min()
            c_prod_max = (cum_c_prod.abs() * c_prod_mask).max()
            vtrace_stats["vtrace/c_t_product_mean"] = float(c_prod_mean)
            vtrace_stats["vtrace/c_t_product_min"] = float(c_prod_min)
            vtrace_stats["vtrace/c_t_product_max"] = float(c_prod_max)
            # Also keep print for backward compatibility
            print(
                "[VTRACE] cumulative c_t product stats "
                f"(mean={float(c_prod_mean):.4e}, "
                f"min={float(c_prod_min):.4e}, "
                f"max={float(c_prod_max):.4e})"
            )

        # Statistics for per-segment cumulative c_t products (only meaningful when recurrence > 0).
        # We log aggregated stats across segments to avoid spamming Comet with per-segment time series.
        if len(seg_c_prod_list) > 0:
            # Stack: (n_segments, bs)
            seg_c_prod = torch.stack(seg_c_prod_list, dim=0).abs()

            # Batch mask: keep sequences that have any valid token in the response (simple proxy)
            batch_mask = response_mask[:, 0] if response_mask.ndim == 2 else response_mask
            batch_mask = batch_mask.to(dtype=seg_c_prod.dtype)

            # Mean per segment over batch (masked), then aggregate across segments
            seg_mean_over_batch = (seg_c_prod * batch_mask.unsqueeze(0)).sum(dim=1) / (batch_mask.sum() + 1e-8)
            vtrace_stats["vtrace/seg_c_t_product_mean_mean"] = float(seg_mean_over_batch.mean())
            vtrace_stats["vtrace/seg_c_t_product_mean_min"] = float(seg_mean_over_batch.min())
            vtrace_stats["vtrace/seg_c_t_product_mean_max"] = float(seg_mean_over_batch.max())
        
        # Statistics for V-trace target values (before normalization)
        v_trace_masked = v_trace_values * response_mask
        vtrace_stats["vtrace/v_target_mean"] = float(verl_F.masked_mean(v_trace_masked, response_mask))
        vtrace_stats["vtrace/v_target_min"] = float((v_trace_masked * response_mask).min())
        vtrace_stats["vtrace/v_target_max"] = float((v_trace_masked * response_mask).max())
        # Compute std using masked_var: std = sqrt(var)
        # Clamp var to non-negative to avoid NaN from sqrt
        v_trace_var = verl_F.masked_var(v_trace_masked, response_mask)
        v_trace_var = torch.clamp(v_trace_var, min=0.0)
        vtrace_stats["vtrace/v_target_std"] = float(torch.sqrt(v_trace_var))
        
        # Advantages are already computed inside the loop (matching Sample Factory)
        # Apply mask to advantages (zero out invalid tokens)
        advantages = advantages * response_mask
        
        # Check for NaN or Inf before normalization
        if torch.any(torch.isnan(advantages)) or torch.any(torch.isinf(advantages)):
            # Replace NaN/Inf with zeros
            advantages = torch.where(
                torch.isnan(advantages) | torch.isinf(advantages),
                torch.zeros_like(advantages),
                advantages
            )
        
        # Normalize advantages like GAE/PPO (compute_gae_advantage_return)
        if response_mask.any():
            advantages = verl_F.masked_whiten(advantages, response_mask)
        else:
            advantages = torch.zeros_like(advantages)
    
    return advantages, returns, vtrace_stats


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
    rollout_log_probs=None,
    imp_ratio_cap=-1,
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    
    if imp_ratio_cap > 0:
        # Apply importance sampling ratio clipping
        imp_ratio = torch.exp(old_log_prob - rollout_log_probs)
        imp_ratio = torch.clamp(imp_ratio, max=imp_ratio_cap)
        pg_losses = pg_losses * imp_ratio
    
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(vpreds: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, cliprange_value: float, loss_agg_mode: str = "token-mean"):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac

def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:

    forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty)
    backward_score = 0.5 * (logprob - ref_logprob).square()

    return backward_score - backward_score.detach() + forward_score.detach()

def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data
