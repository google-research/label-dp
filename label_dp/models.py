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

"""ResNet."""


import functools
from typing import Any, Callable, Optional, Sequence, Tuple

from flax import linen as nn
import jax.numpy as jnp

ModuleDef = Any


################################################################################
# ResNet V2
################################################################################
class BasicBlockV2(nn.Module):
  """Basic Block for a ResNet V2."""

  channels: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    preact = self.act(self.norm()(x))
    y = self.conv(self.channels, (3, 3), self.strides)(preact)
    y = self.act(self.norm()(y))
    y = self.conv(self.channels, (3, 3))(y)

    if y.shape != x.shape:
      shortcut = self.conv(self.channels, (1, 1), self.strides)(preact)
    else:
      shortcut = x
    return shortcut + y


class BottleneckBlockV2(nn.Module):
  """Bottleneck Block for a ResNet V2."""

  channels: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    preact = self.act(self.norm()(x))
    y = self.conv(self.channels, (1, 1))(preact)
    y = self.act(self.norm()(y))
    y = self.conv(self.channels, (3, 3), self.strides)(y)
    y = self.act(self.norm()(y))
    y = self.conv(self.channels * 4, (1, 1))(y)

    if y.shape != x.shape:
      shortcut = self.conv(self.channels * 4, (1, 1), self.strides)(preact)
    else:
      shortcut = x

    return shortcut + y


class ResNetV2(nn.Module):
  """ResNet v2.

      K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual
      networks. In ECCV, pages 630–645, 2016.
  """

  stage_sizes: Sequence[int]
  block_class: ModuleDef
  num_classes: Optional[int] = None
  base_channels: int = 64
  act: Callable = nn.relu
  dtype: Any = jnp.float32
  small_image: bool = False
  # if not None, batch statistics are sync-ed across replica according to
  # this axis_name used in pmap
  bn_cross_replica_axis_name: Optional[str] = None

  @nn.compact
  def __call__(self, x, train: bool = True):
    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)
    norm = functools.partial(
        nn.BatchNorm, use_running_average=not train,
        momentum=0.9, epsilon=1e-5, dtype=self.dtype,
        axis_name=self.bn_cross_replica_axis_name)

    if self.small_image:  # suitable for Cifar
      x = conv(self.base_channels, (3, 3), padding='SAME')(x)
    else:
      x = conv(self.base_channels, (7, 7), (2, 2), padding=[(3, 3), (3, 3)])(x)
      x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')

    for i, n_blocks in enumerate(self.stage_sizes):
      for j in range(n_blocks):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_class(self.base_channels * 2 ** i, strides=strides,
                             conv=conv, norm=norm, act=self.act)(x)

    x = self.act(norm(name='bn_final')(x))
    x = jnp.mean(x, axis=(1, 2))
    if self.num_classes is not None:
      x = nn.Dense(self.num_classes, dtype=self.dtype, name='classifier')(x)
    return x


CifarResNet18V2 = functools.partial(
    ResNetV2, stage_sizes=[2, 2, 2, 2], block_class=BasicBlockV2,
    small_image=True)

CifarResNet50V2 = functools.partial(
    ResNetV2, stage_sizes=[3, 4, 6, 3], block_class=BottleneckBlockV2,
    small_image=True)

ResNet50V2 = functools.partial(
    ResNetV2, stage_sizes=[3, 4, 6, 3], block_class=BottleneckBlockV2)


################################################################################
# ResNet V1
################################################################################
class BasicBlockV1(nn.Module):
  """Basic block for a ResNet V1."""

  channels: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x,):
    residual = x
    y = self.conv(self.channels, (3, 3), self.strides)(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.channels, (3, 3))(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape:
      residual = self.conv(self.channels, (1, 1),
                           self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class BottleneckBlockV1(nn.Module):
  """Bottleneck block for ResNet V1."""

  channels: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    residual = x
    y = self.conv(self.channels, (1, 1))(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.channels, (3, 3), self.strides)(y)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.channels * 4, (1, 1))(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape:
      residual = self.conv(self.channels * 4, (1, 1),
                           self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class ResNetV1(nn.Module):
  """ResNetV1.

      K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image
      recognition. In CVPR, pages 770–778, 2016.
  """

  stage_sizes: Sequence[int]
  block_class: ModuleDef
  num_classes: Optional[int] = None
  base_channels: int = 64
  act: Callable = nn.relu
  dtype: Any = jnp.float32
  small_image: bool = False
  # if not None, batch statistics are sync-ed across replica according to
  # this axis_name used in pmap
  bn_cross_replica_axis_name: Optional[str] = None

  @nn.compact
  def __call__(self, x, train: bool = True):
    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)
    norm = functools.partial(
        nn.BatchNorm, use_running_average=not train, momentum=0.9,
        epsilon=1e-5, dtype=self.dtype,
        axis_name=self.bn_cross_replica_axis_name)

    if self.small_image:  # suitable for Cifar
      x = conv(self.base_channels, (3, 3), padding='SAME', name='conv_init')(x)
      x = norm(name='bn_init')(x)
      x = self.act(x)
    else:
      x = conv(self.base_channels, (7, 7), (2, 2), padding=[(3, 3), (3, 3)],
               name='conv_init')(x)
      x = norm(name='bn_init')(x)
      x = nn.relu(x)
      x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')

    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_class(self.base_channels * 2 ** i, strides=strides,
                             conv=conv, norm=norm, act=self.act)(x)

    x = jnp.mean(x, axis=(1, 2))
    if self.num_classes is not None:
      x = nn.Dense(self.num_classes, dtype=self.dtype, name='classifier')(x)
    return x


CifarResNet18V1 = functools.partial(
    ResNetV1, stage_sizes=[2, 2, 2, 2], block_class=BasicBlockV1,
    small_image=True)

CifarResNet50V1 = functools.partial(
    ResNetV1, stage_sizes=[3, 4, 6, 3], block_class=BottleneckBlockV1,
    small_image=True)

ResNet50V1 = functools.partial(
    ResNetV1, stage_sizes=[3, 4, 6, 3], block_class=BottleneckBlockV1)
