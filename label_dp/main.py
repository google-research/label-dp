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

"""Sets up environment and calls training."""

import copy
import datetime
import logging as native_logging
import os
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
from clu import platform

import jax
import ml_collections
import tensorflow as tf

from label_dp import profiles
from label_dp import train


FLAGS = flags.FLAGS


_BASE_WORKDIR = flags.DEFINE_string(
    'base_workdir', '',
    'Base directory for logs, checkpoints, and other outputs. '
    'When a profile key is given, logs will go into subfolders '
    'specified by the key.')
_PROFILE_KEY = flags.DEFINE_string('profile_key', None,
                                   'Key to a pre-defined profile.')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if jax.process_count() > 1:
    raise NotImplementedError()

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  configs = profiles.Registry.get_profile(_PROFILE_KEY.value)
  configs = ml_collections.ConfigDict(copy.deepcopy(configs))
  workdir = os.path.join(_BASE_WORKDIR.value, _PROFILE_KEY.value)

  # logging
  logdir = os.path.join(workdir, 'logs')
  tf.io.gfile.makedirs(logdir)
  log_file = os.path.join(
      logdir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.txt')
  log_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
  formatter = native_logging.Formatter(log_format)
  file_stream = tf.io.gfile.GFile(log_file, 'w')
  handler = native_logging.StreamHandler(file_stream)
  handler.setLevel(native_logging.INFO)
  handler.setFormatter(formatter)
  logging.get_absl_logger().addHandler(handler)

  if jax.process_index() == 0:
    work_unit = platform.work_unit()
    work_unit.create_artifact(
        artifact_type=platform.ArtifactType.DIRECTORY,
        artifact=workdir, description='Working directory')
    work_unit.create_artifact(
        artifact_type=platform.ArtifactType.FILE,
        artifact=log_file, description='Log file')

  train.multi_stage_train(configs.train, workdir)
  logging.flush()


if __name__ == '__main__':
  app.run(main)
