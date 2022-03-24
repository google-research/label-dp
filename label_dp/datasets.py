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

"""Dataset."""

import numpy as np
import tensorflow_datasets as tfds


def batch_random_crop(batch_image_np, pad=4):
  """Randomly cropping images for data augmentation."""
  n, h, w, c = batch_image_np.shape
  # pad
  padded_image = np.zeros((n, h+2*pad, w+2*pad, c),
                          dtype=batch_image_np.dtype)
  padded_image[:, pad:-pad, pad:-pad, :] = batch_image_np
  # crop
  idxs = np.random.randint(2*pad, size=(n, 2))
  cropped_image = np.array([
      padded_image[i, y:y+h, x:x+w, :]
      for i, (y, x) in enumerate(idxs)])
  return cropped_image


def batch_random_fliplr(batch_image_np):
  """Randomly do left-right flip on images."""
  n = batch_image_np.shape[0]
  coins = np.random.choice([-1, 1], size=n)
  flipped_image = np.array([
      batch_image_np[i, :, ::coins[i], :]
      for i in range(n)])
  return flipped_image


def batch_random_cutout(batch_image_np, size=8):
  """Random cutout.

  Note we are using the same cutout region for all the images in the batch.

  Args:
    batch_image_np: images.
    size: cutout size.

  Returns:
    Images with random cutout.
  """
  _, h, w, _ = batch_image_np.shape
  h0 = np.random.randint(0, h-size)
  w0 = np.random.randint(0, w-size)
  batch_image_np[:, h0:h0+size, w0:w0+size, :] = 0
  return batch_image_np


class TFDSNumpyDataset:
  """Images dataset loaded into memory as numpy array.

  The full data array in numpy format can be easily accessed. Suitable for
  smaller scale image datasets like MNIST (and variants), CIFAR-10 / CIFAR-100,
  SVHN, etc.
  """

  def __init__(self, name, random_crop=True, random_fliplr=True,
               random_cutout=0):
    """Constructs a dataset from tfds.

    Args:
      name: name of the dataset.
      random_crop: whether to perform random crop in data augmentation.
      random_fliplr: whether to perform random left-right flip in data aug.
      random_cutout: if non-zero, denote the cutout length.
    """
    self.name = name

    self._random_crop = random_crop
    self._random_fliplr = random_fliplr
    self._random_cutout = random_cutout
    self.ds, self.info = tfds.load(name, batch_size=-1,
                                   as_dataset_kwargs={'shuffle_files': False},
                                   with_info=True)
    self.ds_np = tfds.as_numpy(self.ds)

    self._add_index_feature()

  def _add_index_feature(self):
    """Adds 'index' feature if not present."""
    for split in self.ds_np:
      if 'index' in self.ds_np[split]:
        continue
      n_sample = len(self.ds_np[split]['label'])
      index = np.arange(n_sample)
      if 'id' in self.ds_np[split]:
        # remove the 'id' feature, b/c jax cannot handle string type
        self.ds_np[split].pop('id')
      self.ds_np[split]['index'] = index

  @property
  def num_classes(self):
    return self.info.features['label'].num_classes

  @property
  def use_onehot_label(self):
    return False

  @property
  def data_scale(self):
    return 255.0

  def get_num_examples(self, split_name):
    return self.ds_np[split_name]['image'].shape[0]

  def get_input_shape(self, input_name):
    if input_name == 'image':
      return self.ds_np['train']['image'].shape[1:]
    raise KeyError(f'getting input shape for {input_name}')

  def normalize_images(self, batch_image_np):
    images = batch_image_np.astype(np.float32) / self.data_scale
    return images

  def iterate(self, split_name, batch_size, shuffle=False, augmentation=False,
              subset_index=None):
    """Iterates over the dataset."""
    n_sample = self.get_num_examples(split_name)
    # make a shallow copy
    dset = dict(self.ds_np[split_name])

    if subset_index is not None:
      n_sample = len(subset_index)
      for key in dset:
        dset[key] = dset[key][subset_index]

    if shuffle:
      rp = np.random.permutation(n_sample)
      for key in dset:
        dset[key] = dset[key][rp]

    for i in range(0, n_sample, batch_size):
      batch = {key: val[i:i+batch_size]
               for key, val in dset.items()}
      batch['image'] = self.normalize_images(batch['image'])
      if augmentation:
        if self._random_crop:
          batch['image'] = batch_random_crop(batch['image'])
        if self._random_fliplr:
          batch['image'] = batch_random_fliplr(batch['image'])
        if self._random_cutout > 0:
          batch['image'] = batch_random_cutout(
              batch['image'], self._random_cutout)

      yield batch


class LabelRemappedTrainDataset:
  """A derived dataset where the labels are remapped according to the index."""

  def __init__(self, dataset, subset_index):
    self.dataset = dataset
    self.subset_index = subset_index
    # if not None, could be a (n, k) array that defines the k-dimensional
    # one-hot vector label for each vector. Note n here is the total number
    # of examples in the original dataset as the index in the original data
    # is used to address this array. However, this array does not need to have
    # meaningful values outside of the examples specified by subset_index.
    self.label_mapping = None
    # subset mask can be used to further filter out some
    # of the examples in the training set
    self.subset_mask = np.ones(len(subset_index), dtype=np.bool)

  @property
  def num_classes(self):
    return self.dataset.num_classes

  @property
  def use_onehot_label(self):
    return True

  def get_num_examples(self, split_name):
    if split_name == 'train':
      return len(self.subset_index[self.subset_mask])
    else:
      return self.dataset.get_num_examples(split_name)

  def get_input_shape(self, input_name):
    return self.dataset.get_input_shape(input_name)

  def iterate(self, split_name, batch_size, shuffle=False, augmentation=False,
              subset_index=None):
    """Iterate over the dataset."""
    assert subset_index is None, ('LabelRemappedTrainDataset does not support '
                                  'further subset indexing.')

    if split_name == 'train':
      for batch in self.dataset.iterate(
          'train', batch_size, shuffle=shuffle, augmentation=augmentation,
          subset_index=self.subset_index[self.subset_mask]):
        yield self.remap_batch_label(batch)
    else:
      for batch in self.dataset.iterate(
          split_name, batch_size, shuffle=shuffle, augmentation=augmentation):
        batch['orig_label'] = batch['label']
        batch['label'] = self._make_onehot(batch['label'])
        yield batch

  def remap_batch_label(self, batch):
    batch['orig_label'] = batch['label']
    if self.label_mapping is not None:
      batch['label'] = self.label_mapping[batch['index'], :]
    else:
      batch['label'] = self._make_onehot(batch['label'])
    return batch

  def _make_onehot(self, labels):
    return np.eye(self.num_classes, dtype=np.float32)[labels, :]
