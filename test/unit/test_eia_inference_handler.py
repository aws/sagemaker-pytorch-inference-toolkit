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

import unittest.mock as mock
import pytest
import torch
import torch.nn as nn

from sagemaker_pytorch_serving_container import default_inference_handler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DummyModel(nn.Module):

    def __init__(self, ):
        super(DummyModel, self).__init__()

    def forward(self, x):
        pass

    def __call__(self, tensor):
        return 3 * tensor


@pytest.fixture(scope='session', name='tensor')
def fixture_tensor():
    tensor = torch.rand(5, 10, 7, 9)
    return tensor.to(device)


@pytest.fixture()
def eia_inference_handler():
    return default_inference_handler.DefaultPytorchInferenceHandler()


def test_eia_default_model_fn(eia_inference_handler):
    with mock.patch("sagemaker_pytorch_serving_container.default_inference_handler.os") as mock_os:
        mock_os.getenv.return_value = "true"
        mock_os.path.join.return_value = "model_dir"
        mock_os.path.exists.return_value = True
        with mock.patch("torch.jit.load") as mock_torch:
            mock_torch.return_value = DummyModel()
            model = eia_inference_handler.default_model_fn("model_dir")
    assert model is not None


def test_eia_default_model_fn_error(eia_inference_handler):
    with mock.patch("sagemaker_pytorch_serving_container.default_inference_handler.os") as mock_os:
        mock_os.getenv.return_value = "true"
        mock_os.path.join.return_value = "model_dir"
        mock_os.path.exists.return_value = False
        with pytest.raises(FileNotFoundError):
            eia_inference_handler.default_model_fn("model_dir")


def test_eia_default_predict_fn(eia_inference_handler, tensor):
    model = DummyModel()
    with mock.patch("sagemaker_pytorch_serving_container.default_inference_handler.os") as mock_os:
        mock_os.getenv.return_value = "true"
        with mock.patch("torch.jit.optimized_execution") as mock_torch:
            mock_torch.__enter__.return_value = "dummy"
            eia_inference_handler.default_predict_fn(tensor, model)
        mock_torch.assert_called_once()
