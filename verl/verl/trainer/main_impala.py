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
Main entry point for IMPALA-style PPO training with async rollout actors.
"""

import hydra
import ray

from verl.trainer.ppo.impala_trainer import ImpalaPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_impala(config)


# Define a function to run the IMPALA-style PPO training process
def run_impala(config) -> None:
    # Reduce urllib3 and requests logging to speed up training
    import logging
    import os
    
    urllib3_logger = logging.getLogger("urllib3")
    urllib3_logger.setLevel(logging.WARNING)
    
    requests_logger = logging.getLogger("requests")
    requests_logger.setLevel(logging.WARNING)
    
    # Reduce Comet ML logging if environment variable is set
    comet_log_level = os.environ.get("COMET_LOGGING_CONSOLE", "INFO")
    if comet_log_level == "WARNING":
        comet_ml_logger = logging.getLogger("comet_ml")
        comet_ml_logger.setLevel(logging.WARNING)
    
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        ray_init_kwargs = {
            "runtime_env": {"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"}},
            "num_cpus": config.ray_init.num_cpus,
        }
        
        # Automatically determine num_gpus from trainer.n_gpus_per_node
        if hasattr(config.trainer, 'n_gpus_per_node') and config.trainer.n_gpus_per_node is not None:
            total_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
            ray_init_kwargs["num_gpus"] = total_gpus
            print(f"[INFO] Ray init: automatically set num_gpus={total_gpus} from trainer.n_gpus_per_node={config.trainer.n_gpus_per_node} * nnodes={config.trainer.nnodes}")
        
        ray.init(**ray_init_kwargs)

    # Create a remote instance of the TaskRunner class
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # Print the initial configuration
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        from verl.utils.fs import copy_to_local
        
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # Instantiate the tokenizer and processor
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Version validation for vllm
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # Define worker classes based on the actor strategy
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # Map roles to their corresponding remote worker classes
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Define the resource pool specification
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # Add reward model if enabled
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Add reference policy worker if needed
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Load the reward manager
        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Create worker groups (similar to RayPPOTrainer.init_workers)
        from verl.single_controller.ray import RayClassWithInitArgs, create_colocated_worker_cls

        # Create resource pools first
        resource_pool_manager.create_resource_pool()
        
        # Initialize resource_pool_to_cls dictionary
        resource_pool_to_cls = {pool: {} for pool in resource_pool_manager.resource_pool_dict.values()}
        
        # Create actor and rollout
        resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=role_worker_mapping[Role.ActorRollout],
            config=config.actor_rollout_ref,
            role="actor_rollout",
        )
        resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        
        # Create critic
        if Role.Critic in role_worker_mapping:
            resource_pool = resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=role_worker_mapping[Role.Critic],
                config=config.critic,
            )
            resource_pool_to_cls[resource_pool]["critic"] = critic_cls
        
        # Create reference policy if needed
        if Role.RefPolicy in role_worker_mapping:
            resource_pool = resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                cls=role_worker_mapping[Role.RefPolicy],
                config=config.actor_rollout_ref,
                role="ref",
            )
            resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls
        
        # Create reward model if needed
        if Role.RewardModel in role_worker_mapping:
            resource_pool = resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=role_worker_mapping[Role.RewardModel],
                config=config.reward_model,
            )
            resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # Create worker groups
        all_wg = {}
        for resource_pool, class_dict in resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        # Initialize worker groups
        actor_rollout_wg = all_wg["actor_rollout"]
        actor_rollout_wg.init_model()

        critic_wg = all_wg.get("critic", None)
        if critic_wg is not None:
            critic_wg.init_model()

        ref_policy_wg = all_wg.get("ref", None)
        if ref_policy_wg is not None:
            ref_policy_wg.init_model()

        rm_wg = all_wg.get("rm", None)
        if rm_wg is not None:
            rm_wg.init_model()

        # Initialize the IMPALA trainer
        trainer = ImpalaPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            actor_rollout_wg=actor_rollout_wg,
            critic_wg=critic_wg,
            ref_policy_wg=ref_policy_wg,
            rm_wg=rm_wg,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            resource_pool_manager=resource_pool_manager,
            role_worker_mapping=role_worker_mapping,
            ray_worker_group_cls=ray_worker_group_cls,
        )

        # Start the training process
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset."""
    from torch.utils.data import Dataset
    from verl.utils.dataset.rl_dataset import RLHFDataset

    # Check if a custom dataset class is specified
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset."""
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
