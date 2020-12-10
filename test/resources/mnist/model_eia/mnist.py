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

VERSIONS_USE_NEW_API = ["1.5.1"]

def predict_fn(input_data, model):
    logger.info('Performing EIA inference with Torch JIT context with input of size {}'.format(input_data.shape))
    # With EI, client instance should be CPU for cost-efficiency. Subgraphs with unsupported arguments run locally. Server runs with CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_data = input_data.to(device)
    model = model.eval() 
    with torch.no_grad():
        print("current torch version is: ", torch.__version__)
        if torch.__version__ in VERSIONS_USE_NEW_API:
            import torcheia
            # we need to set the profiling executor for EIA
            torch._C._jit_set_profiling_executor(False)
            # Here want to use the first attached accelerator, so we specify ordinal 0.
            model = torcheia.jit.attach_eia(model, 0)
            with torch.jit.optimized_execution(True):
                return model.forward(input_data)
        else:
            # Set the target device to the accelerator ordinal
            with torch.jit.optimized_execution(True, {'target_device': 'eia:0'}):
                return model(input_data)


def model_fn(model_dir):
    logger.info('model_fn: Loading model with TorchScript from {}'.format(model_dir))
    # Scripted model is serialized with torch.jit.save().
    # No need to instantiate model definition then load state_dict
    model = torch.jit.load('model.pth')
    return model


def save_model(model, model_dir):
    logger.info("Saving the model to {}.".format(model_dir))
    path = os.path.join(model_dir, 'model.pth')
    torch.jit.save(model, path)
