# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os

from .builder import SYCLOpBuilder


class RaggedUtilsBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_RAGGED_OPS"
    NAME = "ragged_ops"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def load(self):

        class RaggedOPS():

            def allocate_fast_host_buffer(self, device_mirror):
                return device_mirror

            def allocate_view_on(self, tensor, buffer, offset):
                return buffer[offset:offset + tensor.numel()].view(tensor.shape)

            def allocate_view_like(self, shape, strides, dummy_tensor, buffer, offset):
                pass

        return RaggedOPS()

    def absolute_name(self):
        return f'deepspeed.inference.v2.{self.NAME}'

    def filter_ccs(self, ccs):
        ccs_retained = []
        ccs_pruned = []
        for cc in ccs:
            if int(cc[0]) >= 6:
                ccs_retained.append(cc)
            else:
                ccs_pruned.append(cc)
        if len(ccs_pruned) > 0:
            self.warning(f"Filtered compute capabilities {ccs_pruned}")
        return ccs_retained

    def get_prefix(self):
        ds_path = self.deepspeed_src_path("deepspeed")
        return "deepspeed" if os.path.isdir(ds_path) else ".."

    def sources(self):
        sources = [
            "inference/v2/ragged/csrc/fast_host_buffer.cu",
            "inference/v2/ragged/csrc/ragged_ops.cpp",
        ]

        prefix = self.get_prefix()
        sources = [os.path.join(prefix, src) for src in sources]
        return sources

    def extra_ldflags(self):
        return []

    def include_paths(self):
        include_dirs = ['inference/v2/ragged/includes', 'inference/v2/kernels/includes']
        prefix = self.get_prefix()
        includes = [os.path.join(prefix, include_dir) for include_dir in include_dirs]

        return includes
