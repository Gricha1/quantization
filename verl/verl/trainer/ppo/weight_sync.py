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
Weight synchronization manager for IMPALA-style training.
Synchronizes weights between learner and async rollout actors.
"""
import logging
from typing import List, Optional

import ray

from verl.single_controller.ray.base import RayWorkerGroup

logger = logging.getLogger(__file__)


class WeightSyncManager:
    """
    Manages synchronization of model weights between learner and async actors.
    
    In IMPALA-style training, the learner updates the model weights, and these
    weights need to be periodically synchronized to the rollout actors to ensure
    they are using relatively up-to-date policies.
    """

    def __init__(
        self,
        learner_worker_group: RayWorkerGroup,
        actor_worker_groups: List[RayWorkerGroup],
        sync_freq: int = 1,
    ):
        """
        Initialize the weight sync manager.
        
        Args:
            learner_worker_group: Worker group containing the learner model
            actor_worker_groups: List of worker groups for async rollout actors
            sync_freq: Frequency of weight synchronization (every N updates)
        """
        self.learner_wg = learner_worker_group
        self.actor_wgs = actor_worker_groups
        self.sync_freq = sync_freq
        self.last_sync_step = -1

    def should_sync(self, step: int) -> bool:
        """Check if weights should be synchronized at this step."""
        return step % self.sync_freq == 0 and step != self.last_sync_step

    def sync_to_actors(self, step: int):
        """
        Synchronize weights from learner to all actor worker groups.
        
        Args:
            step: Current training step
        """
        if not self.should_sync(step):
            return
        
        try:
            # Use checkpoint save/load mechanism for weight synchronization
            # Save learner weights to temporary path
            import tempfile
            import os
            temp_dir = tempfile.mkdtemp()
            learner_checkpoint_path = os.path.join(temp_dir, "learner_weights")
            
            # Save learner checkpoint
            self.learner_wg.save_checkpoint(
                local_path=learner_checkpoint_path,
                remote_path=None,
                global_step=step,
                max_ckpt_to_keep=1,
            )
            
            # Load checkpoint to all actor worker groups
            for actor_wg in self.actor_wgs:
                actor_wg.load_checkpoint(
                    local_path=learner_checkpoint_path,
                    del_local_after_load=True,
                )
            
            # Cleanup
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
            
            self.last_sync_step = step
            logger.info(f"Weights synchronized to {len(self.actor_wgs)} actor groups at step {step}")
            
        except Exception as e:
            logger.error(f"Error synchronizing weights at step {step}: {e}", exc_info=True)
            # Don't raise - allow training to continue even if sync fails
            pass
