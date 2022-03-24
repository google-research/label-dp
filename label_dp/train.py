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

"""Training script."""

import functools
import pathlib
from typing import Any

from absl import logging

from clu import metric_writers

from flax import jax_utils
from flax.optim import dynamic_scale as dynamic_scale_lib
from flax.training import common_utils
from flax.training import train_state
import jax
import ml_collections
import numpy as np
import tensorflow as tf

from label_dp import datasets
from label_dp import models
from label_dp import utils


class TrainState(train_state.TrainState):
  epoch: int
  model_states: Any
  dynamic_scale: dynamic_scale_lib.DynamicScale


def create_train_state(rng, input_shape, half_precision, model, optimizer_cfgs):
  """Creates initial training state."""
  dynamic_scale = None
  platform = jax.local_devices()[0].platform
  if half_precision and platform == 'gpu':
    dynamic_scale = dynamic_scale_lib.DynamicScale()
  else:
    dynamic_scale = None

  params, model_states = utils.initialize_model(
      rng, input_shape, model)
  tx = utils.build_optimizer(**optimizer_cfgs)
  state = TrainState.create(
      apply_fn=model.apply, params=params, tx=tx, model_states=model_states,
      dynamic_scale=dynamic_scale, epoch=0)
  return state


def build_model(model_configs, num_classes, dtype):
  kwargs = dict(model_configs.kwargs, num_classes=num_classes, dtype=dtype)
  return getattr(models, model_configs.arch)(**kwargs)


def report_metrics(i_stage, report_name, epoch, metrics, writer):
  metrics = common_utils.stack_forest(metrics)
  summary = jax.tree_map(lambda x: float(x.mean()), metrics)
  summary = {f'stage{i_stage}/{report_name}/{k}': v for k, v in summary.items()}
  writer.write_scalars(epoch, summary)


def multi_stage_train(configs: ml_collections.ConfigDict, workdir: str):
  """Multi-stage training."""
  orig_dataset = datasets.TFDSNumpyDataset(
      name=configs.data.name, **configs.data.kwargs)
  stage_datasets = utils.derive_subset_dataset(
      orig_dataset, n_stages=len(configs.stage_specs),
      stage_splits=[spec['data_split'] for spec in configs.stage_specs],
      seed=sum([spec['seed'] for spec in configs.stage_specs]))

  n_tr_total = orig_dataset.get_num_examples('train')
  input_dtype = utils.get_dtype(configs.half_precision)
  model = build_model(configs.model, orig_dataset.num_classes, input_dtype)
  last_stage_state = None
  for i_stage in range(len(stage_datasets)):
    if configs.stage_specs[i_stage]['type'] == 'rr':
      k_for_prior = compute_randomized_labels(
          stage_datasets[i_stage], configs.batch_size,
          configs.stage_specs[i_stage], n_tr_total)
    elif configs.stage_specs[i_stage]['type'] == 'rr-with-prior':
      k_for_prior = compute_randomized_labels_with_priors(
          stage_datasets[i_stage], model, last_stage_state, configs.batch_size,
          configs.stage_specs[i_stage], n_tr_total)
    else:
      raise KeyError('Unknown label randomization type')

    if i_stage != 0 and configs.reuse_last_stage_data:
      logging.info('Reusing randomized data from stage %d in stage %d',
                   i_stage - 1, i_stage)
      if configs.mask_last_stage_label_by_prior:
        logging.info('Filtering out egs from stage %d using learned prior.',
                     i_stage - 1)
        # Note we are passing in the dataset from the last stage and the k
        # from this stage
        filter_stage_data_by_prior(
            stage_datasets[i_stage-1], model, last_stage_state, k_for_prior,
            n_tr_total, configs.batch_size)
      merge_stage_data(stage_datasets[i_stage-1], stage_datasets[i_stage])

    logging.info('#' * 60)
    logging.info('# Stage %d training', i_stage)
    logging.info('#' * 60)
    last_stage_state = single_stage_train(
        configs, workdir, i_stage, stage_datasets, model, last_stage_state)


def single_stage_train(
    configs: ml_collections.ConfigDict, workdir: str, i_stage: int,
    stage_datasets, model, last_stage_state=None):
  """Training for a single stage."""
  workdir = pathlib.Path(workdir) / f'stage{i_stage}'
  tf.io.gfile.makedirs(str(workdir))

  writer = metric_writers.create_default_writer(
      str(workdir / 'tensorboard'), just_logging=jax.process_index() > 0)

  rng = jax.random.PRNGKey(configs.run_seed + i_stage)
  np_rng = np.random.RandomState(seed=configs.run_seed + i_stage + 123)
  mixup_alpha = configs.stage_specs[i_stage]['mixup']
  mixup_sampler = functools.partial(
      np_rng.beta, mixup_alpha, mixup_alpha) if mixup_alpha > 0 else None
  local_batch_size = utils.get_local_batch_size(configs.batch_size)
  dataset = stage_datasets[i_stage]

  n_train = dataset.get_num_examples('train')
  n_train_steps = int(configs.num_epochs * n_train / local_batch_size)
  lr_fn = utils.build_lr_fn(
      configs.lr_fn.name, configs.base_lr, n_train_steps, configs.lr_fn.kwargs)
  optimizer_cfgs = dict(configs.optimizer, learning_rate=lr_fn)
  state = create_train_state(
      rng, dataset.get_input_shape('image'), configs.half_precision, model,
      optimizer_cfgs)
  if last_stage_state is not None:
    state = state.replace(
        params=last_stage_state.params,
        model_states=last_stage_state.model_states)

  state = jax_utils.replicate(state)

  # pmap the train and eval functions
  p_train_step = jax.pmap(
      functools.partial(utils.train_step, model.apply, l2_regu=configs.l2_regu),
      axis_name='batch')
  p_eval_step = jax.pmap(functools.partial(utils.eval_step, model.apply),
                         axis_name='batch')

  def run_eval(epoch=0, split_name='test'):
    eval_metrics = []
    for batch in utils.iterate_data(dataset, split_name, local_batch_size,
                                    desc=f'E{epoch:03d} eval-{split_name}'):
      metrics = p_eval_step(state, batch)
      eval_metrics.append(utils.metrics_to_numpy(metrics))

    report_metrics(i_stage, f'eval-{split_name}', epoch, eval_metrics, writer)

  start_epoch = jax_utils.unreplicate(state.epoch)
  logging.info('Start training from epoch %d...', start_epoch)

  with metric_writers.ensure_flushes(writer):
    while True:
      epoch = int(jax_utils.unreplicate(state.epoch))
      if epoch >= configs.num_epochs:
        break

      train_metrics = []
      for batch in utils.iterate_data(dataset, 'train', local_batch_size,
                                      augmentation=True, shuffle=True,
                                      desc=f'E{epoch+1:03d} train',
                                      mixup_sampler=mixup_sampler):
        state, metrics = p_train_step(state, batch)
        train_metrics.append(utils.metrics_to_numpy(metrics))

      state = state.replace(epoch=state.epoch + 1)
      epoch += 1
      report_metrics(i_stage, 'train', epoch, train_metrics, writer)

      # sync batch statistics across replicas
      state = utils.sync_batch_stats(state)

      for split in configs.eval_splits:
        run_eval(epoch, split)

  utils.block_until_computation_finish()
  return jax_utils.unreplicate(state)


def compute_randomized_labels(dataset, batch_size, spec, n_tr_total):
  """Computes randomized labels."""
  assert isinstance(dataset, datasets.LabelRemappedTrainDataset)
  n_classes = dataset.num_classes
  orig_labels = np.zeros(n_tr_total, dtype=np.int64)

  for batch in dataset.iterate('train', batch_size):
    orig_labels[batch['index']] = batch['orig_label']

  # assign new labels
  rng = np.random.RandomState(seed=spec['seed'])
  dataset.label_mapping = np.zeros((n_tr_total, n_classes), dtype=np.float32)
  dataset.subset_mask = np.ones(len(dataset.subset_index), dtype=np.bool)
  for idx in dataset.subset_index:
    if spec['type'] == 'rr':
      label = orig_labels[idx]
      rate = 1 / (np.exp(spec['eps']) + n_classes - 1)
      prob = np.zeros(n_classes) + rate
      prob[label] = 1 - rate * (n_classes - 1)
      new_label = rng.choice(n_classes, 1, p=prob)
      dataset.label_mapping[idx][new_label] = 1
    else:
      raise KeyError(f'Unknown type: {spec["type"]}')

  if spec['type'] == 'rr':
    return n_classes


def compute_randomized_labels_with_priors(
    dataset, model, last_stage_state, batch_size, spec, n_tr_total):
  """Computes randomized labels based on lp."""
  assert isinstance(dataset, datasets.LabelRemappedTrainDataset)
  assert spec['type'] == 'rr-with-prior'
  ds_weight = spec.get('domain_specific_prior_weight', 0.0)
  assert ds_weight >= 0.0 and ds_weight <= 1.0

  n_classes = dataset.num_classes
  dataset.label_mapping = np.zeros((n_tr_total, n_classes), dtype=np.float32)
  dataset.subset_mask = np.ones(len(dataset.subset_index), dtype=np.bool)
  rng = np.random.RandomState(seed=spec['seed'])
  logging.info('RRWithPrior labeling (T=%f, ds_weight=%f)',
               spec['temperature'], ds_weight)

  model_vars = {'params': last_stage_state.params,
                **last_stage_state.model_states}
  j_pred = jax.jit(functools.partial(model.apply, train=False))

  soft_k = 0.0
  for batch in dataset.iterate('train', batch_size):
    if np.isclose(ds_weight, 1.0):
      p_last_model = 0.0
    else:
      logits = j_pred(model_vars, batch['image'])
      logits /= spec['temperature']
      p_last_model = jax.device_get(jax.nn.softmax(logits))

    if np.isclose(ds_weight, 0.0):
      p_domain = 0.0
    else:
      p_domain = batch['prior']
      p_domain = np.power(p_domain, 1/spec['temperature'])
      p_domain = p_domain / np.sum(p_domain, axis=1, keepdims=True)

    probs = ds_weight*p_domain + (1-ds_weight)*p_last_model
    orig_labels = batch['orig_label']

    for i, prob in enumerate(probs):
      k, new_label = rr_with_prior(prob, spec['eps'], orig_labels[i], rng)
      soft_k += k
      dataset.label_mapping[batch['index'][i]][new_label] = 1

  soft_k /= len(dataset.subset_index)
  logging.info('effective soft_k ~= %.2f (averaged on %d examples).',
               soft_k, len(dataset.subset_index))
  return soft_k


def rr_with_prior(prior, eps, y, rng):
  """Randomized response with prior.

  Args:
    prior: A K-length array where the k-th entry is the probability that the
      true label is k.
    eps: the epsilon value for which the randomized response is epsilon-DP.
    y: an integer indicating the true label.
    rng: a numpy random number generator for sampling.

  Returns:
    k, y_rr: k is the value used in rr-top-k; y_rr is the randomized label.
  """
  idx_sort = np.flipud(np.argsort(prior))
  prior_sorted = prior[idx_sort]
  tmp = np.exp(-eps)
  wks = [np.sum(prior_sorted[:(k+1)]) / (1 + (k-1)*tmp)
         for k in range(len(prior))]
  optim_k = np.argmax(wks) + 1

  adjusted_prior = np.zeros_like(prior) + tmp / (1 + (optim_k-1)*tmp)
  adjusted_prior[y] = 1 / (1 + (optim_k-1)*tmp)
  adjusted_prior[idx_sort[optim_k:]] = 0
  adjusted_prior /= np.sum(adjusted_prior)  # renorm in case y not in topk
  rr_label = rng.choice(len(prior), 1, p=adjusted_prior)
  return optim_k, rr_label


def filter_stage_data_by_prior(last_dataset, model, last_stage_state,
                               k_for_prior, n_tr_total, batch_size):
  """Filtering out the egs from last stage according to prior."""
  assert k_for_prior is not None
  k_for_prior = int(k_for_prior)

  model_vars = {'params': last_stage_state.params,
                **last_stage_state.model_states}
  j_pred = jax.jit(functools.partial(model.apply, train=False))
  global_mask = np.ones(n_tr_total, dtype=np.bool)

  last_dataset.subset_mask[:] = True  # enable all examples
  for batch in last_dataset.iterate('train', batch_size):
    logits = j_pred(model_vars, batch['image'])
    _, topk_idx = jax.device_get(jax.lax.top_k(logits, k=k_for_prior))

    for j in range(batch['image'].shape[0]):
      if np.isclose(np.sum(batch['label'][j, topk_idx[j]]), 0):
        # the randomized label from the last stage is not in the topk prior
        global_mask[batch['index'][j]] = False

  n_filtered = 0
  for i, idx in enumerate(last_dataset.subset_index):
    if not global_mask[idx]:
      last_dataset.subset_mask[i] = False
      n_filtered += 1

  logging.info('%d egs removed due to randomized labels not in top %d prior',
               n_filtered, k_for_prior)


def merge_stage_data(dset_last_stage, dset):
  """Merges the (randomized) data from the last stage to reused in this stage."""
  assert isinstance(dset_last_stage, datasets.LabelRemappedTrainDataset)
  assert isinstance(dset, datasets.LabelRemappedTrainDataset)
  assert dset.label_mapping is not None
  assert dset_last_stage.label_mapping is not None
  # pylint: disable=g-explicit-length-test
  assert len(
      np.intersect1d(
          dset.subset_index, dset_last_stage.subset_index,
          assume_unique=True)) == 0

  dset.label_mapping[dset_last_stage.subset_index,
                     ...] = dset_last_stage.label_mapping[
                         dset_last_stage.subset_index, ...]
  dset.subset_index = np.concatenate(
      [dset_last_stage.subset_index, dset.subset_index], axis=0)
  dset.subset_mask = np.concatenate(
      [dset_last_stage.subset_mask, dset.subset_mask], axis=0)
