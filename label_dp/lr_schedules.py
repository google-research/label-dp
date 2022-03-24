# Copyright 2022 Google LLC.
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

"""Learning rate schedules.

A thin wrapper over optax schedulers with a unifying constructing interface.
"""

from typing import Callable
import optax


Schedule = Callable[[int], float]


def constant(base_lr, num_train_steps) -> Schedule:
  del num_train_steps
  return lambda _: base_lr


def piecewise_constant(base_lr, num_train_steps, *,
                       rampup_thresh=0.15,
                       stages=((0.3, 0.1), (0.6, 0.1), (0.9, 0.1))) -> Schedule:
  """Piecewise constant learning rate with optional linear rampup.

  Args:
    base_lr: base learning rate.
    num_train_steps: total number of training steps.
    rampup_thresh: if not None, can specify a linear rampup.
    stages: a sequence of (step_ratio, scaling_factor). The step_ratio times
      the num_train_steps is the decaying boundary. The scaling factor for all
      the stages whose decaying boundary is less than the current step is
      multiplied to the base learning rate.

  Returns:
    A learning rate schedule.
  """
  lr_fn = optax.piecewise_constant_schedule(
      init_value=base_lr,
      boundaries_and_scales={int(r*num_train_steps): s for r, s in stages})
  if rampup_thresh is not None and rampup_thresh > 0:
    rampup_steps = int(rampup_thresh * num_train_steps)
    rampup_fn = optax.linear_schedule(
        init_value=0, end_value=base_lr, transition_steps=rampup_steps)
    lr_fn = optax.join_schedules(
        schedules=[rampup_fn, lr_fn], boundaries=[rampup_steps])
  return lr_fn
