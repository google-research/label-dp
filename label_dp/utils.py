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

"""Training related utilities."""

import functools
from typing import Optional

from absl import logging

from flax import jax_utils
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm

from label_dp import datasets
from label_dp import lr_schedules


def derive_subset_dataset(dataset, n_stages=2, stage_splits=None, seed=None):
  """Specifies data splits for each stage of training.

  Args:
    dataset: The original dataset to derive from.
    n_stages: number of stages.
    stage_splits: Must be a list of numbers summing to 1, indicating
        the ratio of data for each stage.
    seed: if stage_split is not None, this should not be None, and should
        be the random seed for generating the splits.

  Returns:
    A LabelRemappedTrainDataset.
  """
  n_tr = dataset.get_num_examples('train')
  assert len(stage_splits) == n_stages
  assert seed is not None
  assert np.isclose(sum(stage_splits), 1)
  cum_splits = np.cumsum(stage_splits)
  cum_split_counts = [0] + [int(n_tr * x) for x in cum_splits]
  cum_split_counts[-1] = n_tr
  rng = np.random.RandomState(seed=seed)
  perm = rng.permutation(n_tr)
  stage_subsets = [perm[cum_split_counts[i]:cum_split_counts[i+1]]
                   for i in range(n_stages)]

  return [datasets.LabelRemappedTrainDataset(dataset, subset_index)
          for subset_index in stage_subsets]


def cross_entropy_loss(logits, onehot_labels):
  log_softmax_logits = jax.nn.log_softmax(logits)
  batch_size = onehot_labels.shape[0]
  return -jnp.sum(onehot_labels * log_softmax_logits) / batch_size


def classification_metrics(logits, onehot_labels):
  loss = cross_entropy_loss(logits, onehot_labels)
  acc = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(onehot_labels, -1))
  metrics = {'loss': loss, 'accuracy': acc}
  metrics = jax.lax.pmean(metrics, 'batch')
  return metrics


def l2_regularizer(coefficient, params):
  if coefficient <= 0:
    return 0
  params = jax.tree_leaves(params)
  weight_l2 = sum([jnp.sum(x ** 2) for x in params])
  weight_penalty = coefficient * 0.5 * weight_l2
  return weight_penalty


def block_until_computation_finish():
  """Wait until computations are done."""
  logging.flush()
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()


def get_local_batch_size(batch_size: int) -> int:
  """Gets local batch size to a host."""
  # For example, if we have 2 hosts, each with 8 devices, batch_size=2048, then
  # - batch_size == 2048
  # - local_batch_size == 2048 / 2 == 1024
  # - jax.device_count() == 2*8 == 16
  # The dataset object will be sharded at host level, so each host will see
  # a different subset of the data.
  if batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  local_batch_size = batch_size // jax.process_count()
  logging.info(
      'JAX process_index=%d, process_count=%d, device_count=%d, '
      'local_device_count=%d', jax.process_index(), jax.process_count(),
      jax.device_count(), jax.local_device_count())
  return local_batch_size


def get_dtype(half_precision: bool, platform: Optional[str] = None):
  """Gets concrete data type according to precision and platform.

  Args:
    half_precision: whether to use half precision (float16).
    platform: 'tpu' or 'gpu'.

  Returns:
    A data type to use according to the specification.
  """
  if platform is None:
    platform = jax.local_devices()[0].platform

  if half_precision:
    if platform == 'tpu':
      return jnp.bfloat16
    else:
      return jnp.float16
  else:
    return jnp.float32


def initialize_model(rng, input_shape, model):
  """Initializes parameters and states for a model."""
  input_shape = (1, *input_shape)  # add a dummy batch dimension
  @jax.jit
  def init(*args):
    return model.init(*args)
  variables = init({'params': rng}, jnp.ones(input_shape, model.dtype))
  model_states, params = variables.pop('params')
  return params, model_states


def build_optimizer(name, learning_rate, kwargs):
  """Builds an optimizer."""
  ctor = getattr(optax, name)
  return ctor(learning_rate=learning_rate, **kwargs)


def build_lr_fn(name, base_lr, num_train_steps, kwargs):
  """Builds learning rate scheduler."""
  return getattr(lr_schedules, name)(base_lr, num_train_steps, **kwargs)


def train_step(apply_fn, state, batch, l2_regu,
               f_metrics=classification_metrics):
  """Performs a single training step."""
  def loss_fn(params):
    variables = {'params': params, **state.model_states}
    logits, new_model_states = apply_fn(
        variables, batch['image'], train=True, mutable=['batch_stats'])
    loss = cross_entropy_loss(logits, batch['label'])
    loss = loss + l2_regularizer(l2_regu, variables['params'])
    return loss, (new_model_states, logits)

  state, aux, metrics = optimizer_step(loss_fn, state)
  new_model_states, logits = aux[1]
  new_state = state.replace(model_states=new_model_states)
  metrics.update(f_metrics(logits, batch['label']))

  return new_state, metrics


def optimizer_step(loss_fn, state):
  """Applies one optimizer step."""
  dynamic_scale = state.dynamic_scale

  if dynamic_scale:
    grad_fn = dynamic_scale.value_and_grad(
        loss_fn, has_aux=True, axis_name='batch')
    dynamic_scale, is_fin, aux, grad = grad_fn(state.params)
    # dynamic loss takes care of averaging gradients across replicas
  else:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grad = grad_fn(state.params)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grad = jax.lax.pmean(grad, axis_name='batch')

  metrics = {}
  new_model_states = aux[1][0]
  new_state = state.apply_gradients(grads=grad, model_states=new_model_states)
  if dynamic_scale:
    # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
    # params should be restored (= skip this step).
    new_state = new_state.replace(
        opt_state=jax.tree_multimap(
            functools.partial(jnp.where, is_fin),
            new_state.opt_state,
            state.opt_state),
        params=jax.tree_multimap(
            functools.partial(jnp.where, is_fin),
            new_state.params,
            state.params))
    metrics['scale'] = dynamic_scale.scale

  return new_state, aux, metrics


def eval_step(apply_fn, state, batch, f_metrics=classification_metrics):
  """Performs a single evaluation step."""
  variables = {'params': state.params, **state.model_states}
  logits = apply_fn(variables, batch['image'], train=False, mutable=False)
  return f_metrics(logits, batch['label'])


def iterate_data(dataset, split_name, batch_size, augmentation=False,
                 shuffle=False, desc='', mixup_sampler=None, **kwargs):
  """Iterates over data."""
  iterator = dataset.iterate(split_name, batch_size, shuffle=shuffle,
                             augmentation=augmentation, **kwargs)
  if mixup_sampler is not None:
    def apply_mixup(batch):
      lm = mixup_sampler()
      batch['label'] = lm*batch['label'] + (1-lm)*np.flipud(batch['label'])
      batch['image'] = lm*batch['image'] + (1-lm)*np.flipud(batch['image'])
      return batch
    iterator = map(apply_mixup, iterator)

  iterator = map(common_utils.shard, iterator)
  iterator = jax_utils.prefetch_to_device(iterator, 2)
  iterator = tqdm.tqdm(iterator, desc=desc, disable=None,
                       total=dataset.get_num_examples(split_name) // batch_size)
  return iterator


def metrics_to_numpy(metrics):
  # We select the first element of x in order to get a single copy of a
  # device-replicated metric.
  metrics = jax.tree_map(lambda x: x[0], metrics)
  metrics_np = jax.device_get(metrics)
  return metrics_np


def sync_batch_stats(state):
  """Sync the batch statistics across replicas."""
  if 'batch_stats' not in state.model_states:
    return state

  # An axis_name is passed to pmap which can then be used by pmean.
  # In this case each device has its own version of the batch statistics and
  # we average them.
  avg = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')

  new_model_states = state.model_states.copy({
      'batch_stats': avg(state.model_states['batch_stats'])})
  return state.replace(model_states=new_model_states)
