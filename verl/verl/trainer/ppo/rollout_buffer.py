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
Rollout buffer for storing rollouts from async actors in IMPALA-style training.
"""
import threading
from collections import deque
from typing import Optional

import numpy as np

from verl.protocol import DataProto


class RolloutBuffer:
    """
    Thread-safe buffer for storing rollouts from multiple async actors.
    
    This buffer implements a FIFO queue with a maximum size to prevent
    unbounded memory growth. Rollouts are stored with metadata including
    the actor ID and step number for tracking.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize the rollout buffer.
        
        Args:
            max_size: Maximum number of rollouts to store. When full, oldest rollouts are removed.
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.total_added = 0
        self.total_sampled = 0

    def add(self, rollout_data: DataProto, actor_id: Optional[int] = None, step: Optional[int] = None):
        """
        Add a rollout to the buffer.
        
        Args:
            rollout_data: DataProto containing the rollout data
            actor_id: Optional ID of the actor that generated this rollout
            step: Optional training step when this rollout was generated
        """
        with self.lock:
            entry = {
                'data': rollout_data,
                'actor_id': actor_id,
                'step': step,
                'buffer_index': self.total_added,
            }
            self.buffer.append(entry)
            self.total_added += 1

    def sample(self, batch_size: int) -> list:
        """
        Sample a batch of rollouts from the buffer.
        
        Args:
            batch_size: Number of rollouts to sample
            
        Returns:
            List of rollout entries (dicts with 'data', 'actor_id', 'step', 'buffer_index')
        """
        with self.lock:
            if len(self.buffer) == 0:
                return []
            
            # Sample randomly from available rollouts
            n_samples = min(batch_size, len(self.buffer))
            indices = np.random.choice(len(self.buffer), size=n_samples, replace=False)
            samples = [self.buffer[i] for i in indices]
            self.total_sampled += n_samples
            return samples

    def get_all(self) -> list:
        """
        Get all rollouts from the buffer (clears the buffer).
        
        Returns:
            List of all rollout entries
        """
        with self.lock:
            samples = list(self.buffer)
            self.buffer.clear()
            return samples

    def size(self) -> int:
        """Get current number of rollouts in buffer."""
        with self.lock:
            return len(self.buffer)

    def clear(self):
        """Clear all rollouts from the buffer."""
        with self.lock:
            self.buffer.clear()

    def get_stats(self) -> dict:
        """Get statistics about the buffer."""
        with self.lock:
            return {
                'current_size': len(self.buffer),
                'max_size': self.max_size,
                'total_added': self.total_added,
                'total_sampled': self.total_sampled,
                'utilization': len(self.buffer) / self.max_size if self.max_size > 0 else 0.0,
            }
