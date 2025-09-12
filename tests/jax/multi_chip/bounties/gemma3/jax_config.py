# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax

NUM_VIRTUAL_DEVICES = 4
# jax.config.update("jax_num_cpu_devices", NUM_VIRTUAL_DEVICES)
devices = jax.devices("tpu")[:NUM_VIRTUAL_DEVICES]
axis_name = "X"
num_devices = len(devices)
device_mesh = jax.make_mesh((num_devices,), (axis_name), devices=devices)
