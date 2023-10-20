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

import numpy as np
import pytest
import sagemaker
from sagemaker.pytorch import PyTorchModel

from integration import mnist_cpu_tar, mnist_gpu_tar, mnist_cpu_script, mnist_gpu_script
from integration.sagemaker.timeout import timeout_and_delete_endpoint


@pytest.mark.cpu_test
def test_mnist_cpu(sagemaker_session, image_uri, instance_type):
    instance_type = instance_type or 'ml.c4.xlarge'
    _test_mnist_distributed(sagemaker_session, image_uri, instance_type, mnist_cpu_tar, mnist_cpu_script)


@pytest.mark.gpu_test
def test_mnist_gpu(sagemaker_session, image_uri, instance_type):
    instance_type = instance_type or 'ml.p2.xlarge'
    _test_mnist_distributed(sagemaker_session, image_uri, instance_type, mnist_gpu_tar, mnist_gpu_script)


def _test_mnist_distributed(sagemaker_session, image_uri, instance_type, model_tar, mnist_script,
                            env_vars=None):
    endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-pytorch-serving")

    model_data = sagemaker_session.upload_data(
        path=model_tar,
        key_prefix="sagemaker-pytorch-serving/models",
    )

    if 'gpu' in image_uri:
        env_vars = {
            'NCCL_SHM_DISABLE': '1'
        }

    pytorch = PyTorchModel(model_data=model_data, role='SageMakerRole', entry_point=mnist_script,
                           image_uri=image_uri, sagemaker_session=sagemaker_session, env=env_vars)
    with timeout_and_delete_endpoint(endpoint_name, sagemaker_session, minutes=30):
        predictor = pytorch.deploy(initial_instance_count=1, instance_type=instance_type,
                                   endpoint_name=endpoint_name)

        batch_size = 100
        data = np.random.rand(batch_size, 1, 28, 28).astype(np.float32)
        output = predictor.predict(data)

        assert output.shape == (batch_size, 10)
