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

"""Profile registry."""
import re
import sys


class Registry(object):
  """Registry holding all model specs in the model zoo."""
  registry = None

  @classmethod
  def build_registry(cls):
    """Builds registry, called upon first zoo query."""
    if cls.registry is not None:
      return

    cls.registry = dict()

    mod_profiles = sys.modules[sys.modules[__name__].__package__]
    profile_mods = [getattr(mod_profiles, x) for x in dir(mod_profiles)
                    if re.match(r'^p[0-9]+_', x)]
    profile_mods = [x for x in profile_mods if isinstance(x, type(sys))]
    profile_mods.sort(key=str)
    for p_mod in profile_mods:
      register_funcs = [getattr(p_mod, x) for x in dir(p_mod)
                        if x.startswith('register_')]
      register_funcs = filter(callable, register_funcs)

      for func in register_funcs:
        func(cls)

  @classmethod
  def register(cls, profile):
    key = profile['key']
    if key in cls.registry:  # pylint: disable=unsupported-membership-test
      raise KeyError('duplicated profile key: {}'.format(key))
    cls.registry[key] = profile  # pylint: disable=unsupported-assignment-operation

  @classmethod
  def list_profiles(cls, regex):
    cls.build_registry()
    profiles = [cls.registry[key] for key in cls.registry.keys()
                if re.search(regex, key)]
    return profiles

  @classmethod
  def print_profiles(cls, regex):
    profiles = cls.list_profiles(regex)
    print('{} profiles found ====== with regex: {}'.format(
        len(profiles), regex))
    for i, profile in enumerate(profiles):
      print('  {:>3d}) {}'.format(i, profile['key']))

  @classmethod
  def get_profile(cls, key):
    cls.build_registry()
    assert key is not None
    return cls.registry[key]
