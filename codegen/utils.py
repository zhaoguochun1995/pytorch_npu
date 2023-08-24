# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from collections import defaultdict
from typing import Tuple, List, Iterable, Iterator, Callable, Sequence, TypeVar, Optional, Dict
from enum import Enum
import contextlib
import textwrap
import yaml
import os

# Safely load fast C Yaml loader/dumper if they are available
try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader  # type: ignore[misc]

try:
    from yaml import CSafeDumper as Dumper
except ImportError:
    from yaml import SafeDumper as Dumper  # type: ignore[misc]

from codegen.model import NativeFunction, FunctionSchema
from codegen.api import cpp

YamlDumper = Dumper

GLOBAL_STRUCTURED_OP_INFO_CACHE = defaultdict(str)

CUSTOM_YAML_NAME = "npu_native_functions_by_codegen.yaml"
FIELDS_TO_REMOVE = ["wrap_impl", "impl_name", "impl_ns"]

# A custom loader for YAML that errors on duplicate keys.
# This doesn't happen by default: see https://github.com/yaml/pyyaml/issues/165
class YamlLoader(Loader):
    def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
        mapping = []
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)  # type: ignore[no-untyped-call]
            assert key not in mapping, f"Found a duplicate key in the yaml. key={key}, line={node.start_mark.line}"
            mapping.append(key)
        mapping = super().construct_mapping(node, deep=deep)  # type: ignore[no-untyped-call]
        return mapping

# Many of these functions share logic for defining both the definition
# and declaration (for example, the function signature is the same), so
# we organize them into one function that takes a Target to say which
# code we want.
#
# This is an OPEN enum (we may add more cases to it in the future), so be sure
# to explicitly specify with Union[Literal[Target.XXX]] what targets are valid
# for your use.
Target = Enum('Target', (
    # top level namespace (not including at)
    'DEFINITION',
    'DECLARATION',
    # TORCH_LIBRARY(...) { ... }
    'REGISTRATION',
    # namespace { ... }
    'ANONYMOUS_DEFINITION',
    'ANONYMOUS_DEFINITION_UNSUPPORT',
    # namespace cpu { ... }
    'NAMESPACED_DEFINITION',
    'NAMESPACED_DECLARATION',
))

# Matches "foo" in "foo, bar" but not "foobar". Used to search for the
# occurrence of a parameter in the derivative formula
IDENT_REGEX = r'(^|\W){}($|\W)'

# TODO: Use a real parser here; this will get bamboozled
def split_name_params(schema: str) -> Tuple[str, List[str]]:
    m = re.match(r'(\w+)(\.\w+)?\((.*)\)', schema)
    if m is None:
        raise RuntimeError(f'Unsupported function schema: {schema}')
    name, _, params = m.groups()
    return name, params.split(', ')

T = TypeVar('T')
S = TypeVar('S')

# These two functions purposely return generators in analogy to map()
# so that you don't mix up when you need to list() them

# Map over function that may return None; omit Nones from output sequence
def map_maybe(func: Callable[[T], Optional[S]], xs: Iterable[T]) -> Iterator[S]:
    for x in xs:
        r = func(x)
        if r is not None:
            yield r

# Map over function that returns sequences and cat them all together
def concat_map(func: Callable[[T], Sequence[S]], xs: Iterable[T]) -> Iterator[S]:
    for x in xs:
        for r in func(x):
            yield r

# Conveniently add error context to exceptions raised.  Lets us
# easily say that an error occurred while processing a specific
# context.
@contextlib.contextmanager
def context(msg_fn: Callable[[], str]) -> Iterator[None]:
    try:
        yield
    except Exception as e:
        # TODO: this does the wrong thing with KeyError
        msg = msg_fn()
        msg = textwrap.indent(msg, '  ')
        msg = f'{e.args[0]}\n{msg}' if e.args else msg
        e.args = (msg,) + e.args[1:]
        raise


def parse_npu_yaml(custom_path: str, use_line_loader=True) -> List:
    if not os.path.exists(custom_path):
        return {}

    with open(custom_path, 'r') as yaml_file:
        source_es = yaml.safe_load(yaml_file)

    return source_es


def merge_yaml(base_data, additional_data):
    """Merge two YAML data structures. If there's a conflict, the base data will take precedence."""
    map_dict = {"official": "supported"}
    key_map = lambda x: map_dict.get(x, x)
    if isinstance(base_data, dict):
        for key, value in additional_data.items():
            if key_map(key) not in base_data:
                base_data[key_map(key)] = value
            else:
                base_data[key_map(key)] = merge_yaml(base_data[key_map(key)], value)
    elif isinstance(base_data, list):
        for item in additional_data:
            if item not in base_data:
                base_data.append(item)
    return base_data


def merge_custom_yaml(pta_path, op_plugin_path):
    not_codegen_ops = ["argmin", "argmax", "nan_to_num", "nan_to_num_",
                       "nan_to_num.out", "_embedding_bag_dense_backward"]
    pta_es = parse_npu_yaml(pta_path)
    op_es = parse_npu_yaml(op_plugin_path)

    op_es["official"] = [op for op in op_es["official"]
                         if op["func"].split("(")[0] not in not_codegen_ops]

    merged_yaml = merge_yaml(pta_es, op_es)
    merged_yaml_path = gen_custom_yaml_path(pta_path)
    with open(merged_yaml_path, 'w') as outfile:
        yaml.dump(merged_yaml, outfile, default_flow_style=False, width=float("inf"))
    return merged_yaml


def filed_tag(custom_es):
    for e in custom_es:
        if not isinstance(e, Dict):
            continue
        for field in FIELDS_TO_REMOVE:
            e.pop(field, None)
    return custom_es


def parse_opplugin_yaml(custom_path: str) -> None:
    source_es = parse_npu_yaml(custom_path)

    custom = source_es.pop('custom', [])
    if custom is None:
        custom = []  # Allow an empty list of supported ops
    official = source_es.pop('official', [])
    if official is None:
        official = []  # Allow an empty list of supported ops

    support_ops = custom + official

    symint = source_es.pop("symint", [])
    if symint is None:
        symint = []
    symint = [op['func'] if isinstance(op, Dict) else op for op in symint]
    symint_set = set([str(FunctionSchema.parse(op).name) for op in symint])

    global GLOBAL_STRUCTURED_OP_INFO_CACHE
    for x in support_ops:
        funcs = x.get("func", None)
        assert isinstance(funcs, str), f'not a str : {funcs}'
        func = FunctionSchema.parse(funcs)
        wrap_name = cpp.name(func)
        op_key = str(func.name)
        if op_key in symint_set:
            wrap_name += "_symint"
        cur_wrap_name = GLOBAL_STRUCTURED_OP_INFO_CACHE.get(op_key, "")
        if cur_wrap_name and cur_wrap_name != wrap_name:
            print(f"Find different wrap_name for {cur_wrap_name} and {wrap_name} between pta and opplugin, ",
                  f"with {wrap_name} being used as the actual wrap_name")
        GLOBAL_STRUCTURED_OP_INFO_CACHE[op_key] = wrap_name


def enable_opplugin() -> bool:
    # enable op_plugin, if path of third_party/op-plugin is valid.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    op_plugin_path = os.path.join(base_dir, '../third_party/op-plugin/op_plugin')
    return os.path.exists(op_plugin_path)


def is_op_valid(op_key: str) -> bool:
    return True if op_key in GLOBAL_STRUCTURED_OP_INFO_CACHE else False


def get_opplugin_wrap_name(func) -> str:
    op_key = str(func.func.name) if type(func) is NativeFunction else func
    return GLOBAL_STRUCTURED_OP_INFO_CACHE.get(op_key, "")


def gen_custom_yaml_path(original_path, codegen_yaml_filename=CUSTOM_YAML_NAME):
    new_path = os.path.join(os.path.dirname(original_path), codegen_yaml_filename)
    return new_path
