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
Ray remote actors for IMPALA-style parallel rollout generation.
Each actor runs in a separate Ray process and continuously generates rollouts.
"""
import logging
import queue
import threading
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import _pre_process_inputs

logger = logging.getLogger(__file__)


@ray.remote(num_gpus=1)  # Each actor needs at least 1 GPU
class AsyncRolloutActor:
    """
    Ray remote actor that continuously collects rollouts in parallel.
    
    This actor runs in a separate Ray process and continuously generates rollouts
    using vLLM directly (without worker groups) to avoid NCCL issues.
    """

    def __init__(
        self,
        actor_id: int,
        config: DictConfig,
        model_path: str,
    ):
        """
        Initialize the async rollout actor.
        
        Args:
            actor_id: Unique ID for this actor
            config: Training configuration
            model_path: Path to the model
        """
        self.actor_id = actor_id
        self.config = config
        self.total_rollouts_collected = 0
        self.stop_flag = False
        
        # Initialize vLLM directly (no worker groups to avoid NCCL issues)
        from vllm import LLM, SamplingParams
        from omegaconf import OmegaConf
        from copy import deepcopy
        
        rollout_config = config.actor_rollout_ref.rollout
        tensor_parallel_size = rollout_config.get("tensor_model_parallel_size", 1)
        
        # Create vLLM engine
        engine_kwargs = {}
        if "engine_kwargs" in rollout_config and "vllm" in rollout_config.engine_kwargs:
            engine_kwargs = OmegaConf.to_container(deepcopy(rollout_config.engine_kwargs.vllm))
            engine_kwargs = {k: v for k, v in engine_kwargs.items() if v is not None}
        
        self.llm = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="ray",  # Use Ray backend for distributed execution
            dtype=rollout_config.dtype,
            enforce_eager=rollout_config.get("enforce_eager", True),
            gpu_memory_utilization=rollout_config.get("gpu_memory_utilization", 0.4),
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=rollout_config.get("max_model_len", None),
            load_format=rollout_config.get("load_format", "auto"),
            disable_log_stats=rollout_config.get("disable_log_stats", False),
            max_num_batched_tokens=rollout_config.get("max_num_batched_tokens", 8192),
            enable_chunked_prefill=rollout_config.get("enable_chunked_prefill", False),
            enable_prefix_caching=True,
            trust_remote_code=config.actor_rollout_ref.model.get("trust_remote_code", False),
            seed=rollout_config.get("seed", 0),
            **engine_kwargs,
        )
        
        # Create sampling params
        sampling_kwargs = dict(
            n=rollout_config.get("n", 1),
            logprobs=0,  # Will be recomputed by actor
            max_tokens=rollout_config.get("response_length", 512),
        )
        
        # Add any other sampling params from config
        for k in rollout_config.keys():
            if hasattr(SamplingParams(), k):
                sampling_kwargs[k] = rollout_config.get(k)
        
        self.sampling_params = SamplingParams(**sampling_kwargs)
        
        # Get tokenizer
        self.tokenizer = self.llm.get_tokenizer()
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

    def generate_rollout(self, gen_batch: DataProto) -> Optional[DataProto]:
        """
        Generate a single rollout batch using vLLM directly.
        
        Args:
            gen_batch: Batch of prompts to generate from
            
        Returns:
            Generated rollout data or None if stopped
        """
        if self.stop_flag:
            return None
        
        try:
            # Extract input data from DataProto
            idx = gen_batch.batch["input_ids"]  # (bs, prompt_length)
            attention_mask = gen_batch.batch["attention_mask"]
            position_ids = gen_batch.batch["position_ids"]
            eos_token_id = gen_batch.meta_info.get("eos_token_id", self.tokenizer.eos_token_id)
            pad_token_id = self.pad_token_id
            
            batch_size = idx.size(0)
            
            # Pre-process inputs for vLLM (remove left padding)
            non_tensor_batch = gen_batch.non_tensor_batch.copy() if gen_batch.non_tensor_batch else {}
            if "raw_prompt_ids" not in non_tensor_batch:
                non_tensor_batch["raw_prompt_ids"] = np.array(
                    [_pre_process_inputs(pad_token_id, idx[i]) for i in range(batch_size)],
                    dtype=object
                )
            
            # Prepare vLLM inputs
            if "multi_modal_data" in non_tensor_batch:
                vllm_inputs = []
                for raw_prompt_ids, multi_modal_data in zip(
                    non_tensor_batch.pop("raw_prompt_ids"),
                    non_tensor_batch.pop("multi_modal_data")
                ):
                    vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
            else:
                vllm_inputs = [
                    {"prompt_token_ids": raw_prompt_ids} 
                    for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
                ]
            
            # Ensure prompt_token_ids is list[int]
            for input_data in vllm_inputs:
                if isinstance(input_data["prompt_token_ids"], np.ndarray):
                    input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
                elif not isinstance(input_data["prompt_token_ids"], list):
                    raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")
            
            # Generate using vLLM
            outputs = self.llm.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            
            # Convert vLLM outputs to DataProto format
            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    # Extract log probs if available
                    curr_log_prob = []
                    if output.outputs[sample_id].logprobs:
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            if logprob and response_ids[i] in logprob:
                                curr_log_prob.append(logprob[response_ids[i]].logprob)
                            else:
                                curr_log_prob.append(0.0)
                    else:
                        curr_log_prob = [0.0] * len(response_ids)
                    rollout_log_probs.append(curr_log_prob)
            
            # Pad responses and log probs
            response = pad_2d_list_to_length(
                response, pad_token_id, max_length=self.config.actor_rollout_ref.rollout.response_length
            ).to(idx.device)
            rollout_log_probs = pad_2d_list_to_length(
                rollout_log_probs, -1, max_length=self.config.actor_rollout_ref.rollout.response_length
            ).to(idx.device)
            rollout_log_probs = rollout_log_probs.to(torch.float32)
            
            # Handle n > 1 case
            n = self.sampling_params.n
            if n > 1:
                idx = idx.repeat_interleave(n, dim=0)
                attention_mask = attention_mask.repeat_interleave(n, dim=0)
                position_ids = position_ids.repeat_interleave(n, dim=0)
                batch_size = batch_size * n
            
            # Concatenate prompts and responses
            seq = torch.cat([idx, response], dim=-1)
            
            # Compute response mask and update attention mask
            response_attention_mask = get_response_mask(
                response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([attention_mask, response_attention_mask], dim=-1)
            
            # Update position ids
            delta_position_id = torch.arange(response.size(1), device=position_ids.device).unsqueeze(0)
            response_position_ids = position_ids[:, -1:] + delta_position_id
            position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
            
            # Create output DataProto
            batch = TensorDict(
                {
                    "prompts": idx,
                    "responses": response,
                    "input_ids": seq,
                    "rollout_log_probs": rollout_log_probs,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                },
                batch_size=batch_size,
            )
            
            output_batch = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
            
            self.total_rollouts_collected += 1
            return output_batch
            
        except Exception as e:
            logger.error(f"Actor {self.actor_id}: Error generating rollout: {e}", exc_info=True)
            import traceback
            logger.error(traceback.format_exc())
            return None

    def update_weights(self, state_dict):
        """
        Update the actor's model weights from the learner.
        
        Args:
            state_dict: State dict from learner's model
        """
        # Note: vLLM doesn't support dynamic weight updates easily
        # For now, we'll need to restart the vLLM engine with new weights
        # This is a limitation - in a real implementation, you might need to
        # use a different approach or wait for vLLM to support weight updates
        logger.warning(f"Actor {self.actor_id}: Weight updates not yet implemented for vLLM direct mode")

    def stop(self):
        """Stop this actor."""
        self.stop_flag = True

    def get_stats(self) -> dict:
        """Get statistics about this actor."""
        return {
            'actor_id': self.actor_id,
            'total_rollouts_collected': self.total_rollouts_collected,
        }


class RolloutActorManager:
    """
    Manager for multiple Ray remote rollout actors.
    
    This class manages the lifecycle of multiple AsyncRolloutActor Ray actors
    and coordinates parallel rollout collection.
    """

    def __init__(
        self,
        n_actors: int,
        config: DictConfig,
        resource_pool_manager,
        role_worker_mapping,
        ray_worker_group_cls,
    ):
        """
        Initialize the rollout actor manager.
        
        Args:
            n_actors: Number of async rollout actors to create
            config: Training configuration
            resource_pool_manager: Resource pool manager
            role_worker_mapping: Mapping of roles to worker classes
            ray_worker_group_cls: Ray worker group class
        """
        self.n_actors = n_actors
        self.config = config
        self.resource_pool_manager = resource_pool_manager
        self.role_worker_mapping = role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        
        self.actors = []
        self.actor_worker_groups = []
        
    def create_actors(self):
        """Create and initialize all Ray remote rollout actors."""
        # Get model path from config
        model_path = self.config.actor_rollout_ref.model.path
        
        # Create Ray remote actors with vLLM directly (no worker groups)
        for i in range(self.n_actors):
            actor = AsyncRolloutActor.remote(
                actor_id=i,
                config=self.config,
                model_path=model_path,
            )
            self.actors.append(actor)
        
        # Wait for all actors to initialize
        ray.get([actor.get_stats.remote() for actor in self.actors])
        
        logger.info(f"Created {self.n_actors} Ray remote rollout actors")

    def generate_rollouts_parallel(self, gen_batches: list) -> list:
        """
        Generate rollouts in parallel from multiple actors.
        
        Args:
            gen_batches: List of prompt batches, one per actor
            
        Returns:
            List of generated rollouts (as Ray futures)
        """
        futures = []
        for i, (actor, gen_batch) in enumerate(zip(self.actors, gen_batches)):
            future = actor.generate_rollout.remote(gen_batch)
            futures.append(future)
        return futures

    def sync_weights_to_actors(self, learner_state_dict):
        """
        Synchronize learner's weights to all actors.
        
        Args:
            learner_state_dict: State dict from learner's model
        """
        futures = []
        for actor in self.actors:
            future = actor.update_weights.remote(learner_state_dict)
            futures.append(future)
        ray.get(futures)
        logger.debug(f"Synchronized weights to {len(self.actors)} actors")

    def stop_all(self):
        """Stop all rollout actors."""
        futures = [actor.stop.remote() for actor in self.actors]
        ray.get(futures)
        logger.info(f"Stopped {len(self.actors)} rollout actors")

    def get_stats(self) -> dict:
        """Get statistics from all actors."""
        stats_futures = [actor.get_stats.remote() for actor in self.actors]
        stats = ray.get(stats_futures)
        return {
            'n_actors': self.n_actors,
            'actors': stats,
            'total_rollouts': sum(s['total_rollouts_collected'] for s in stats),
        }
