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

from sagemaker_inference.default_handler_service import DefaultHandlerService
from sagemaker_pytorch_serving_container.transformer import PyTorchTransformer

import os
import sys

PYTHON_PATH_ENV = "PYTHONPATH"
ENABLE_MULTI_MODEL = os.getenv("SAGEMAKER_MULTI_MODEL", "false") == "true"


class HandlerService(DefaultHandlerService):
    """
    Handler service that is executed by the model server.

    Determines specific default inference handlers to use based on the type pytorch model being used.

    This class extends ``DefaultHandlerService``, which define the following:
        - The ``handle`` method is invoked for all incoming inference requests to the model server.
        - The ``initialize`` method is invoked at model server start up.
    """

    def __init__(self):
        self._initialized = False

        transformer = PyTorchTransformer()
        super(HandlerService, self).__init__(transformer=transformer)

    def initialize(self, context):
        # Adding the 'code' directory path to sys.path to allow importing user modules when multi-model mode is enabled.
        if (not self._initialized) and ENABLE_MULTI_MODEL:
            code_dir = os.path.join(context.system_properties.get("model_dir"), 'code')
            sys.path.append(code_dir)
            self._initialized = True

        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # add model_dir/code to python path
        code_dir_path = "{}:".format(model_dir + "/code")
        if PYTHON_PATH_ENV in os.environ:
            os.environ[PYTHON_PATH_ENV] = code_dir_path + os.environ[PYTHON_PATH_ENV]
        else:
            os.environ[PYTHON_PATH_ENV] = code_dir_path

        self._service.validate_and_initialize(model_dir=model_dir, context=context)
