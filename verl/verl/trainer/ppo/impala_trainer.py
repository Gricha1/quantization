# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
IMPALA-style PPO trainer with async rollout actors and learner.
Implements actor-learner architecture where rollouts are collected in parallel
with model updates, using V-trace for off-policy correction.
"""
import logging
import queue
import threading
from collections import defaultdict
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl.protocol import DataProto, pad_dataproto_to_divisor
from verl.single_controller.base import Worker
from verl.single_controller.ray.base import RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward
from verl.trainer.ppo.rollout_actor import RolloutActorManager
from verl.trainer.ppo.rollout_buffer import RolloutBuffer
from verl.trainer.ppo.weight_sync import WeightSyncManager
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager, find_latest_ckpt_path
from verl.utils.debug.performance import _timer
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import ValidationGenerationsLogger

logger = logging.getLogger(__file__)


class ImpalaPPOTrainer:
    """
    IMPALA-style PPO trainer with async rollout collection.
    
    This trainer implements an actor-learner architecture where:
    - Multiple async actors continuously collect rollouts in the background
    - A learner process consumes rollouts from a buffer and updates the model
    - Weights are periodically synchronized from learner to actors
    - V-trace is used for off-policy correction
    """
    
    def __init__(
        self,
        config: DictConfig,
        tokenizer,
        actor_rollout_wg: RayWorkerGroup,
        critic_wg: Optional[RayWorkerGroup] = None,
        ref_policy_wg: Optional[RayWorkerGroup] = None,
        rm_wg: Optional[RayWorkerGroup] = None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        resource_pool_manager=None,
        role_worker_mapping=None,
        ray_worker_group_cls=None,
    ):
        """
        Initialize IMPALA-style PPO trainer.
        
        Args:
            config: Training configuration
            tokenizer: Tokenizer for text processing
            actor_rollout_wg: Worker group for actor/rollout (used by async actors)
            critic_wg: Optional worker group for critic model
            ref_policy_wg: Optional worker group for reference policy
            rm_wg: Optional worker group for reward model
            reward_fn: Reward function
            val_reward_fn: Validation reward function
            train_dataset: Training dataset
            val_dataset: Validation dataset
            collate_fn: Collate function for dataloader
            train_sampler: Optional sampler for training
            device_name: Device name (cuda/cpu)
        """
        self.config = config
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg  # Learner's worker group
        self.critic_wg = critic_wg
        self.ref_policy_wg = ref_policy_wg
        self.rm_wg = rm_wg
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.device_name = device_name
        self.resource_pool_manager = resource_pool_manager
        self.role_worker_mapping = role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        
        # IMPALA-specific settings
        self.n_actors = config.trainer.get("n_async_actors", 4)
        self.rollout_buffer_size = config.trainer.get("rollout_buffer_size", 1000)
        self.weight_sync_freq = config.trainer.get("weight_sync_freq", 1)
        self.min_buffer_size = config.trainer.get("min_buffer_size", 10)  # Min rollouts before starting updates
        
        # Use V-trace for off-policy correction
        if config.algorithm.adv_estimator != AdvantageEstimator.VTRACE:
            logger.warning(f"IMPALA trainer recommends V-trace, but adv_estimator is {config.algorithm.adv_estimator}")
        
        # Create rollout buffer
        self.rollout_buffer = RolloutBuffer(max_size=self.rollout_buffer_size)
        
        # Create rollout actor manager (will create Ray remote actors)
        self.rollout_actor_manager = RolloutActorManager(
            n_actors=self.n_actors,
            config=config,
            resource_pool_manager=resource_pool_manager,
            role_worker_mapping=role_worker_mapping,
            ray_worker_group_cls=ray_worker_group_cls,
        )
        
        # Create dataloader
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        
        # Training state
        self.global_steps = 0
        self.total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs if self.train_dataloader else 0
        
        # Validation logger
        self.validation_generations_logger = ValidationGenerationsLogger()
        
        # IMPALA-specific statistics for logging
        self.weight_sync_count = 0
        self.total_weight_sync_time = 0.0
        self.total_rollout_wait_time = 0.0
        self.rollout_wait_count = 0
        self.empty_buffer_count = 0
        self.total_pending_rollouts = 0
        self.pending_rollout_samples = 0

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """Create dataloaders for training and validation."""
        from torchdata.stateful_dataloader import StatefulDataLoader
        
        if train_dataset is not None:
            self.train_dataloader = StatefulDataLoader(
                train_dataset,
                batch_size=self.config.data.train_batch_size,
                collate_fn=collate_fn,
                sampler=train_sampler,
            )
        else:
            self.train_dataloader = None
        
        if val_dataset is not None:
            self.val_dataloader = StatefulDataLoader(
                val_dataset,
                batch_size=self.config.data.val_batch_size,
                collate_fn=collate_fn,
            )
        else:
            self.val_dataloader = None

    def fit(self):
        """
        Main training loop for IMPALA-style PPO.
        
        This loop:
        1. Starts async rollout actors
        2. Feeds prompts to actors via queue
        3. Consumes rollouts from buffer
        4. Updates model using V-trace
        5. Synchronizes weights to actors periodically
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking
        
        tracking_logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        
        self.global_steps = 0
        
        # Load checkpoint if exists
        self._load_checkpoint()
        
        # Initial validation
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            # Allow empty metrics if validation dataset is empty or validation was skipped
            if val_metrics:
                from pprint import pprint
                pprint(f"Initial validation metrics: {val_metrics}")
                tracking_logger.log(data=val_metrics, step=self.global_steps)
            else:
                logging.warning("Validation returned empty metrics. Skipping validation logging.")
            if self.config.trainer.get("val_only", False):
                return
        
        # Start async rollout actors (Ray remote actors)
        logging.info(f"Creating {self.n_actors} Ray remote rollout actors")
        self.rollout_actor_manager.create_actors()
        
        # Progress bar
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="IMPALA Training")
        
        # Pending rollout futures (parallel generation)
        pending_rollout_futures = []
        
        try:
            # Main training loop
            for epoch in range(self.config.trainer.total_epochs):
                if self.train_dataloader is None:
                    break
                
                for batch_dict in self.train_dataloader:
                    metrics = {}
                    timing_raw = {}
                    
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    
                    # Prepare prompts for async actors
                    gen_batch = self._prepare_gen_batch(batch)
                    
                    # Split batch across actors and send to parallel rollout generation
                    # For simplicity, we'll send the same batch to all actors
                    # In a more sophisticated implementation, we'd split the batch
                    gen_batches = [gen_batch] * self.n_actors
                    rollout_futures = self.rollout_actor_manager.generate_rollouts_parallel(gen_batches)
                    pending_rollout_futures.extend(rollout_futures)
                    
                    # Wait for at least one rollout to complete
                    if len(pending_rollout_futures) < self.min_buffer_size:
                        import time
                        wait_start = time.time()
                        time.sleep(0.1)
                        self.total_rollout_wait_time += time.time() - wait_start
                        self.rollout_wait_count += 1
                        self.empty_buffer_count += 1
                        continue
                    
                    # Track pending rollouts for statistics
                    self.total_pending_rollouts += len(pending_rollout_futures)
                    self.pending_rollout_samples += 1
                    
                    # Get completed rollouts (non-blocking)
                    import time
                    wait_start = time.time()
                    ready_futures, pending_rollout_futures = ray.wait(
                        pending_rollout_futures,
                        num_returns=min(len(pending_rollout_futures), self.config.data.train_batch_size),
                        timeout=0.1,
                    )
                    self.total_rollout_wait_time += time.time() - wait_start
                    self.rollout_wait_count += 1
                    
                    if len(ready_futures) == 0:
                        continue
                    
                    # Get rollout results
                    rollout_results = ray.get(ready_futures)
                    rollout_results = [r for r in rollout_results if r is not None]  # Filter None results
                    
                    if len(rollout_results) == 0:
                        continue
                    
                    # Combine rollouts into batch
                    combined_batch = self._combine_rollouts(rollout_results)
                    
                    is_last_step = self.global_steps >= self.total_training_steps
                    
                    with _timer("step", timing_raw):
                        # Compute rewards
                        with _timer("reward", timing_raw):
                            reward_tensor, reward_extra_infos_dict = compute_reward(
                                combined_batch, self.reward_fn
                            )
                            combined_batch.batch["token_level_scores"] = reward_tensor
                        
                        # Compute old log probs (current/learner policy)
                        with _timer("old_log_prob", timing_raw):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(combined_batch)
                            combined_batch = combined_batch.union(old_log_prob)
                        
                        # Compute reference log probs if needed
                        if self.ref_policy_wg is not None:
                            with _timer("ref", timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(combined_batch)
                                combined_batch = combined_batch.union(ref_log_prob)
                        
                        # Compute values
                        if self.critic_wg is not None:
                            with _timer("values", timing_raw):
                                values = self.critic_wg.compute_values(combined_batch)
                                combined_batch = combined_batch.union(values)
                        
                        # Compute advantages using V-trace
                        with _timer("adv", timing_raw):
                            from verl.trainer.ppo.ray_trainer import compute_advantage
                            
                            combined_batch.batch["token_level_rewards"] = combined_batch.batch["token_level_scores"]
                            
                            # Ensure we have rollout_log_probs (from rollout policy)
                            if "rollout_log_probs" not in combined_batch.batch:
                                # Use old_log_probs as rollout_log_probs if not available
                                combined_batch.batch["rollout_log_probs"] = combined_batch.batch["old_log_probs"]
                            
                            # Compute importance weights for V-trace logging (before compute_advantage)
                            if self.config.algorithm.adv_estimator == AdvantageEstimator.VTRACE:
                                if "old_log_probs" in combined_batch.batch and "rollout_log_probs" in combined_batch.batch:
                                    log_ratio = combined_batch.batch["old_log_probs"] - combined_batch.batch["rollout_log_probs"]
                                    log_ratio = torch.clamp(log_ratio, min=-10.0, max=10.0)
                                    importance_ratio = torch.exp(log_ratio)
                                    
                                    # Get V-trace parameters (can be in algorithm.vtrace.rho_bar or algorithm.vtrace_rho_bar)
                                    vtrace_config = self.config.algorithm.get("vtrace", {})
                                    rho_bar = vtrace_config.get("rho_bar", self.config.algorithm.get("vtrace_rho_bar", 1.0))
                                    c_bar = vtrace_config.get("c_bar", self.config.algorithm.get("vtrace_c_bar", 1.0))
                                    
                                    # Truncated importance weights
                                    rho_t = torch.clamp(importance_ratio, max=rho_bar)
                                    c_t = torch.clamp(importance_ratio, max=c_bar)
                                    
                                    combined_batch.batch["importance_weights"] = rho_t
                                    combined_batch.batch["vtrace_c_weights"] = c_t
                                    combined_batch.batch["importance_ratio"] = importance_ratio
                                    combined_batch.batch["_vtrace_rho_bar"] = rho_bar  # Store for later use
                                    combined_batch.batch["_vtrace_c_bar"] = c_bar  # Store for later use
                            
                            combined_batch = compute_advantage(
                                combined_batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                                config=self.config.algorithm,
                            )
                        
                        # Update critic
                        if self.critic_wg is not None:
                            with _timer("update_critic", timing_raw):
                                critic_output = self.critic_wg.update_critic(combined_batch)
                                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                                metrics.update(critic_output_metrics)
                        
                        # Update actor (learner)
                        if self.config.trainer.critic_warmup <= self.global_steps:
                            with _timer("update_actor", timing_raw):
                                combined_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                actor_output = self.actor_rollout_wg.update_actor(combined_batch)
                                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                                metrics.update(actor_output_metrics)
                        
                        # Synchronize weights to actors periodically
                        if self.global_steps % self.weight_sync_freq == 0:
                            import time
                            sync_start = time.time()
                            with _timer("weight_sync", timing_raw):
                                try:
                                    learner_state_dict = self.actor_rollout_wg.get_state_dict()
                                    self.rollout_actor_manager.sync_weights_to_actors(learner_state_dict)
                                    sync_time = time.time() - sync_start
                                    self.weight_sync_count += 1
                                    self.total_weight_sync_time += sync_time
                                except Exception as e:
                                    logging.warning(f"Could not sync weights: {e}")
                        
                        # Validation
                        if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (
                            is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                        ):
                            with _timer("testing", timing_raw):
                                val_metrics = self._validate()
                                if is_last_step:
                                    last_val_metrics = val_metrics
                                metrics.update(val_metrics)
                        
                        # Save checkpoint
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                        ):
                            with _timer("save_checkpoint", timing_raw):
                                self._save_checkpoint()
                    
                    # Training metrics
                    metrics.update({
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    })
                    
                    # Collect metrics
                    metrics.update(compute_data_metrics(batch=combined_batch, use_critic=self.critic_wg is not None))
                    metrics.update(compute_timing_metrics(batch=combined_batch, timing_raw=timing_raw))
                    
                    # Throughput metrics
                    n_gpus = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
                    metrics.update(compute_throughout_metrics(
                        batch=combined_batch, timing_raw=timing_raw, n_gpus=n_gpus
                    ))
                    
                    # Buffer stats
                    buffer_stats = self.rollout_buffer.get_stats()
                    metrics.update({
                        "impala/buffer_size": buffer_stats["current_size"],
                        "impala/buffer_utilization": buffer_stats["utilization"],
                        "impala/total_rollouts_added": buffer_stats["total_added"],
                        "impala/total_rollouts_sampled": buffer_stats["total_sampled"],
                    })
                    
                    # Actor stats
                    actor_stats = self.rollout_actor_manager.get_stats()
                    metrics.update({
                        "impala/n_actors": actor_stats["n_actors"],
                        "impala/total_actor_rollouts": actor_stats["total_rollouts"],
                    })
                    
                    # Pending rollouts statistics
                    avg_pending_rollouts = (
                        self.total_pending_rollouts / self.pending_rollout_samples 
                        if self.pending_rollout_samples > 0 else 0
                    )
                    metrics.update({
                        "impala/avg_pending_rollouts": avg_pending_rollouts,
                        "impala/current_pending_rollouts": len(pending_rollout_futures),
                    })
                    
                    # Weight synchronization statistics
                    avg_weight_sync_time = (
                        self.total_weight_sync_time / self.weight_sync_count 
                        if self.weight_sync_count > 0 else 0
                    )
                    metrics.update({
                        "impala/weight_sync_count": self.weight_sync_count,
                        "impala/avg_weight_sync_time": avg_weight_sync_time,
                        "impala/weight_sync_freq": self.weight_sync_freq,
                    })
                    
                    # Rollout wait statistics
                    avg_rollout_wait_time = (
                        self.total_rollout_wait_time / self.rollout_wait_count 
                        if self.rollout_wait_count > 0 else 0
                    )
                    metrics.update({
                        "impala/avg_rollout_wait_time": avg_rollout_wait_time,
                        "impala/rollout_wait_count": self.rollout_wait_count,
                        "impala/empty_buffer_count": self.empty_buffer_count,
                        "impala/min_buffer_size": self.min_buffer_size,
                    })
                    
                    # V-trace statistics (if available in batch)
                    if "importance_weights" in combined_batch.batch:
                        importance_weights = combined_batch.batch["importance_weights"]
                        importance_ratio = combined_batch.batch.get("importance_ratio", importance_weights)
                        c_weights = combined_batch.batch.get("vtrace_c_weights", importance_weights)
                        
                        # Apply response mask if available
                        response_mask = combined_batch.batch.get("response_mask", None)
                        if response_mask is not None:
                            importance_weights_masked = importance_weights * response_mask
                            importance_ratio_masked = importance_ratio * response_mask
                            c_weights_masked = c_weights * response_mask
                        else:
                            importance_weights_masked = importance_weights
                            importance_ratio_masked = importance_ratio
                            c_weights_masked = c_weights
                        
                        metrics.update({
                            "impala/vtrace/mean_rho_weight": importance_weights_masked.mean().item(),
                            "impala/vtrace/max_rho_weight": importance_weights_masked.max().item(),
                            "impala/vtrace/min_rho_weight": importance_weights_masked.min().item(),
                            "impala/vtrace/std_rho_weight": importance_weights_masked.std().item(),
                            "impala/vtrace/mean_c_weight": c_weights_masked.mean().item(),
                            "impala/vtrace/mean_importance_ratio": importance_ratio_masked.mean().item(),
                            "impala/vtrace/max_importance_ratio": importance_ratio_masked.max().item(),
                        })
                        
                        # Count clipped importance weights
                        # Get from batch if stored, otherwise from config
                        rho_bar = combined_batch.batch.get("_vtrace_rho_bar", None)
                        c_bar = combined_batch.batch.get("_vtrace_c_bar", None)
                        if rho_bar is None:
                            vtrace_config = self.config.algorithm.get("vtrace", {})
                            rho_bar = vtrace_config.get("rho_bar", self.config.algorithm.get("vtrace_rho_bar", 1.0))
                        if c_bar is None:
                            vtrace_config = self.config.algorithm.get("vtrace", {})
                            c_bar = vtrace_config.get("c_bar", self.config.algorithm.get("vtrace_c_bar", 1.0))
                        if isinstance(rho_bar, torch.Tensor):
                            rho_bar = rho_bar.item() if rho_bar.numel() == 1 else rho_bar[0].item()
                        if isinstance(c_bar, torch.Tensor):
                            c_bar = c_bar.item() if c_bar.numel() == 1 else c_bar[0].item()
                        rho_clipped = (importance_ratio > rho_bar).float()
                        c_clipped = (importance_ratio > c_bar).float()
                        if response_mask is not None:
                            rho_clipped = (rho_clipped * response_mask).sum() / response_mask.sum()
                            c_clipped = (c_clipped * response_mask).sum() / response_mask.sum()
                        else:
                            rho_clipped = rho_clipped.mean()
                            c_clipped = c_clipped.mean()
                        
                        metrics.update({
                            "impala/vtrace/rho_clipped_ratio": rho_clipped.item(),
                            "impala/vtrace/c_clipped_ratio": c_clipped.item(),
                        })
                    
                    # Parallelism efficiency metrics
                    if len(pending_rollout_futures) > 0:
                        parallelism_ratio = len(ready_futures) / len(pending_rollout_futures) if len(pending_rollout_futures) > 0 else 0
                        metrics.update({
                            "impala/parallelism_ratio": parallelism_ratio,
                            "impala/ready_rollouts": len(ready_futures),
                        })
                    
                    # Log metrics
                    tracking_logger.log(data=metrics, step=self.global_steps)
                    
                    progress_bar.update(1)
                    self.global_steps += 1
                    
                    if is_last_step:
                        from pprint import pprint
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        progress_bar.close()
                        break
        
        finally:
            # Stop async actors
            logging.info("Stopping async rollout actors")
            self.rollout_actor_manager.stop_all()
            # Finish logging (Tracking.__del__ will also call finish, but explicit is better)
            if hasattr(tracking_logger, 'logger') and "comet_ml" in tracking_logger.logger:
                tracking_logger.logger["comet_ml"].finish()

    def _prepare_gen_batch(self, batch: DataProto) -> DataProto:
        """Prepare batch for generation by removing training-specific keys."""
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        return gen_batch

    def _combine_rollouts(self, rollout_samples: list) -> DataProto:
        """Combine multiple rollout samples into a single batch."""
        if len(rollout_samples) == 0:
            raise ValueError("Cannot combine empty rollout samples")
        
        if len(rollout_samples) == 1:
            return rollout_samples[0]["data"]
        
        # Combine all rollouts
        combined = rollout_samples[0]["data"]
        for sample in rollout_samples[1:]:
            combined = combined.union(sample["data"])

        return combined

    def _validate(self):
        """Perform validation."""
        from collections import defaultdict
        import numpy as np
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
        
        if self.val_dataloader is None or self.val_reward_fn is None:
            return {}
        
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        
        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_ground_truths = []
        sample_correct = []
        
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            
            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True
            )
            
            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch.get("reward_model", {}).get("style") == "model":
                return {}
            
            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            
            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, self.actor_rollout_wg.world_size
            )
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            
            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            
            test_batch = test_batch.union(test_output_gen_batch)
            
            # Extract ground truth from test_batch if available
            ground_truths = []
            for i in range(len(output_texts)):
                gt = None
                # Try to get from non_tensor_batch
                if "ground_truth" in test_batch.non_tensor_batch:
                    gt_list = test_batch.non_tensor_batch["ground_truth"]
                    if isinstance(gt_list, (list, tuple)) and i < len(gt_list):
                        gt = str(gt_list[i])
                # Try to get from reward_model info
                if gt is None and "reward_model" in test_batch.non_tensor_batch:
                    rm_info = test_batch.non_tensor_batch["reward_model"]
                    if isinstance(rm_info, (list, tuple)) and i < len(rm_info):
                        rm_item = rm_info[i]
                        if isinstance(rm_item, dict):
                            if "answer" in rm_item:
                                gt = str(rm_item["answer"])
                            elif "solution" in rm_item:
                                gt = str(rm_item["solution"])
                            elif "target" in rm_item:
                                gt = str(rm_item["target"])
                if gt is None:
                    gt = "N/A"
                ground_truths.append(gt)
            
            # Extend ground_truths for repeated samples if needed
            if self.config.actor_rollout_ref.rollout.val_kwargs.n > 1:
                ground_truths = [gt for gt in ground_truths for _ in range(self.config.actor_rollout_ref.rollout.val_kwargs.n)]
            
            sample_ground_truths.extend(ground_truths)
            
            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            
            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    # Try to extract ground truth from reward_extra_info if not found earlier
                    if key in ["answer", "solution", "target", "ground_truth"]:
                        # Update ground truths if they were "N/A"
                        start_idx = len(sample_ground_truths) - len(output_texts)
                        for idx, gt in enumerate(lst):
                            if start_idx + idx < len(sample_ground_truths) and sample_ground_truths[start_idx + idx] == "N/A":
                                sample_ground_truths[start_idx + idx] = str(gt)
            
            # Compute correctness for this batch
            for i, (output, gt) in enumerate(zip(output_texts, ground_truths)):
                # Simple string matching for correctness (can be improved)
                # For GSM8K, we typically extract the final number from the answer
                is_correct = self._check_answer_correctness(output, gt)
                sample_correct.append(is_correct)
            
            data_source_lst.append(
                test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            )
        
        # Log validation generations (ground_truths and correct are already collected)
        self._maybe_log_val_generations(
            inputs=sample_inputs, 
            outputs=sample_outputs, 
            scores=sample_scores,
            ground_truths=sample_ground_truths if sample_ground_truths else None,
            correct=sample_correct if sample_correct else None
        )
        
        # Process validation metrics
        from verl.trainer.ppo.metric_utils import process_validation_metrics
        
        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
        
        data_sources = np.concatenate(data_source_lst, axis=0) if data_source_lst else np.array([])
        
        # Process validation metrics (same as RayPPOTrainer)
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def _maybe_log_val_generations(self, inputs, outputs, scores, ground_truths=None, correct=None):
        """Log validation generations to logger."""
        from verl.utils.tracking import ValidationGenerationsLogger
        
        # Create tuples of (input, output, score, ground_truth, correct) to match expected format
        if ground_truths is None:
            ground_truths = ["N/A"] * len(inputs)
        if correct is None:
            correct = [False] * len(inputs)
        
        samples = list(zip(inputs, outputs, scores, ground_truths, correct))
        
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _check_answer_correctness(self, output: str, ground_truth: str) -> bool:
        """Check if the output answer matches the ground truth.
        
        This is a simple string matching method. For GSM8K, we extract
        the final number from the answer and compare with ground truth.
        """
        if ground_truth == "N/A":
            return False
        
        # Extract final answer from output (GSM8K format: #### number)
        import re
        solution = re.search(r"####\s*([\-0-9\.\,]+)", output)
        if solution:
            extracted_answer = solution.group(1).replace(",", "").replace("$", "").strip()
            gt_clean = str(ground_truth).replace(",", "").replace("$", "").strip()
            return extracted_answer == gt_clean
        
        # Fallback: simple string matching
        output_clean = output.lower().strip()
        gt_clean = str(ground_truth).lower().strip()
        return gt_clean in output_clean or output_clean in gt_clean
    
    def _save_checkpoint(self):
        """Save checkpoint."""
        # TODO: Implement checkpoint saving
        pass

    def _load_checkpoint(self):
        """Load checkpoint."""
        # TODO: Implement checkpoint loading
        pass
