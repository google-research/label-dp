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

"""CIFAR-10 experiments."""


def register_cifar10_experiment(registry):
  """Register experiment configs."""
  batch_size = 512
  learning_rate = 0.4

  num_epochs = 200
  cutout = 8
  lr_fn = 'piecewise_constant'
  optimizer = 'sgd'
  for epsilon, hparam in [
      (1, {'mixup': (16, 8), 'data_split': 80, 'temperature': 0.6}),
      (2, {'mixup': (8, 8), 'data_split': 70, 'temperature': 0.5}),
      (4, {'mixup': (8, 4), 'data_split': 60, 'temperature': 0.5}),
      (8, {'mixup': (4, 4), 'data_split': 60, 'temperature': 0.5}),
  ]:
    for rep in range(5):
      mixup1, mixup2 = hparam['mixup']
      first_stage_data_ratio = hparam['data_split']
      temperature = hparam['temperature']
      key = f'cifar10/e{epsilon}/lp-2st/run{rep}'
      meta = {'target': 'main', 'platform': 'v100'}
      train_configs = {
          'run_seed': 1234 + rep,
          'batch_size': batch_size,
          'half_precision': False,
          'l2_regu': 1e-4,
          'num_epochs': num_epochs,
          'eval_splits': ['test'],
          'reuse_last_stage_data': True,
          'mask_last_stage_label_by_prior': True,
          'data': {'name': 'cifar10', 'kwargs': {'random_cutout': cutout}},
          'base_lr': learning_rate,
          'lr_fn': {'name': lr_fn, 'kwargs': {}},
          'optimizer': get_optimizer(optimizer),
          'model': {'arch': 'CifarResNet18V2', 'kwargs': {}},
          'stage_specs': [
              dict(type='rr', seed=2019, eps=epsilon,
                   data_split=first_stage_data_ratio/100, mixup=mixup1),
              dict(type='rr-with-prior', seed=2020, eps=epsilon,
                   data_split=1 - first_stage_data_ratio/100,
                   temperature=temperature, mixup=mixup2)
          ]
      }

      spec = {'key': key, 'meta': meta, 'train': train_configs}
      registry.register(spec)


def get_optimizer(name):
  cfg = {'name': name}
  if name == 'sgd':
    cfg['kwargs'] = {'momentum': 0.9, 'nesterov': True}
  else:
    cfg['kwargs'] = {}
  return cfg
