# Generates VariableType.h/cpp
#
#If you need any guidance, please refer to the comments in derivatives.yaml in PyTorch.
#

import re
import os
from typing import List

from torchgen.api.autograd import NativeFunctionWithDifferentiabilityInfo
from torchgen.api import cpp
from torchgen.code_template import CodeTemplate
from torchgen.context import native_function_manager
from torchgen.gen import FileManager

from codegen.torch_autograd.gen_inplace_or_view_type import (
    use_derived, METHOD_DEFINITION, gen_formals,
)
from codegen.torch_autograd.gen_trace_type import type_wrapper_name
from codegen.torch_autograd.gen_variable_type import (
    emit_body,
    gen_variable_type_func,
    gen_wrapper_registration
)
from codegen.utils import parse_derivatives_yaml

# NPU methods require special processing. 
NPU_AUTOGRAD_FUNCTION = parse_derivatives_yaml(os.path.join(os.path.dirname(__file__), 'derivatives.yaml'))

try_jit_decomposition_pattern = (r'if \(\(.*?\)\) \{.*?static c10::OperatorName full_name\("aten::.*?", .*?\);\n.*?'
                                 r'return impl::run_jit_decomposition_with_args_for_jvp<at::Tensor>'
                                 r'\(".*?", \*opt_op, ks, .*?\);\n\s*\} '
                                 r'else \{\n\s*(.*?)\n\s*\}')
use_count_pattern = (r'if \(\S+\.has_storage\(\) && !at::impl::dispatch_mode_enabled\(\) && '
                     r'!at::impl::tensor_has_dispatch\(\S+\)\) {\s+TORCH_INTERNAL_ASSERT\('
                     r'\S+\.storage\(\)\.use_count\(\) == 1, "function: \S+"\);\s+}')

METHOD_HEADER_DEFINITION = CodeTemplate("""\
${return_type} ${type_wrapper_name}(${formals});
""")


def gen_variable_type(
    out: str,
    fns_with_diff_infos: List[NativeFunctionWithDifferentiabilityInfo],
    template_path: str,
) -> None:

    """VariableType.cpp body

    This is the at::Type subclass for differentiable tensors. The
    implementation of each function dispatches to the base tensor type to
    compute the output. The grad_fn is attached to differentiable functions.
    
    For npu, we keep original process for torch method. 
    """
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)

    # NOTE: see Note [Sharded File] at the top of the VariableType.cpp
    # template regarding sharding of the generated files.
    fm.write_sharded(
        'VariableType.cpp',
        [fn for fn in fns_with_diff_infos if use_derived(fn)],
        key_fn=lambda fn: cpp.name(fn.func.func),
        base_env={
            'generated_comment':
            "@" f'generated from {template_path}/VariableType.cpp',
        },
        env_callable=gen_variable_type_func,
        num_shards=1,
        sharded_keys={'type_derived_method_definitions', 'wrapper_registrations'}
    )


def gen_npu_variable_type(
    out: str,
    fns_with_diff_infos: List[NativeFunctionWithDifferentiabilityInfo],
    template_path: str,
) -> None:
    
    """Generate VariableTypeNPU.cpp body
    
    Generate variable type definition for npu method here.
    """
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    
    npu_method_definitions: List[str] = []
    wrapper_registrations: List[str] = []
    for fn in fns_with_diff_infos:
        if use_derived(fn):
            f = fn.func
            with native_function_manager(f):
                formals = gen_formals(f)
                key = "Default"
                type_definition = METHOD_DEFINITION.substitute(
                    return_type=cpp.returns_type(f.func.returns).cpp_type(),
                    type_wrapper_name=type_wrapper_name(f),
                    type_definition_body=emit_body(fn, key),
                    formals=formals,
                )

                type_definition = type_definition.replace('at::redispatch', 'op_plugin')
                type_definition = type_definition.replace('ks & c10::after_autograd_keyset, ', '')
                type_definition = re.sub(try_jit_decomposition_pattern, r"\1", type_definition, flags=re.DOTALL)
                type_definition = re.sub(use_count_pattern, "", type_definition, flags=re.DOTALL)
                wrapper_registrations.append(gen_wrapper_registration(f, key))
                npu_method_definitions.append(type_definition)
    
    fm.write_with_template('VariableTypeNPU.cpp', 'VariableTypeNPU.cpp', lambda: {
        'npu_method_definitions': npu_method_definitions,
        'wrapper_registrations': wrapper_registrations
    })
                

def gen_variable_type_head(
    out: str,
    fns_with_diff_infos: List[NativeFunctionWithDifferentiabilityInfo],
    template_path: str,
) -> None:
    
    """Generate VariableType.h body
    """
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    fm.write_with_template('VariableType.h', 'VariableType.h', lambda: {
        'generated_comment': "@" f'generated from {template_path}/VariableType.h',
        'npu_variable_type': gen_variable_type_header(fns_with_diff_infos)
    })


def gen_variable_type_header(
    fns_with_diff_infos: List[NativeFunctionWithDifferentiabilityInfo]
) -> List[str]:
    variable_type_header: List[str] = []
    for fn in fns_with_diff_infos:
        f = fn.func
        with native_function_manager(f):
            formals = gen_formals(f)
            wrapper_name = type_wrapper_name(f)
            name = str(fn.func.func.name)
            if name in NPU_AUTOGRAD_FUNCTION:
                type_header_definition = METHOD_HEADER_DEFINITION.substitute(
                        return_type=cpp.returns_type(f.func.returns).cpp_type(),
                        type_wrapper_name=wrapper_name,
                        formals=formals,
                    )
                variable_type_header.append(type_header_definition)
    return variable_type_header
