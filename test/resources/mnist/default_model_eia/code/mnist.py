# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import
import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def predict_fn(input_data, model):
    logger.info('Performing EIA inference with Torch JIT context with input of size {}'.format(input_data.shape))
    # With EI, client instance should be CPU for cost-efficiency.
    # Sub-graphs with unsupported arguments run locally. Server runs with CUDA
    device = torch.device('cpu')
    model = model.to(device)
    input_data = input_data.to(device)
    with torch.no_grad():
        # Set the target device to the accelerator ordinal
        with torch.jit.optimized_execution(True, {'target_device': 'eia:0'}):
            return model(input_data)
