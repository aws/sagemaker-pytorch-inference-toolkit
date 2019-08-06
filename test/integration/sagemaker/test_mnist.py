# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from sagemaker.pytorch import PyTorchModel

from test.integration import (DEFAULT_TIMEOUT, mnist_script, model_cpu_dir, model_gpu_dir)
from test.integration.sagemaker.timeout import timeout, timeout_and_delete_endpoint


@pytest.mark.skip_gpu
@pytest.mark.skip_py2
def test_mnist_distributed_cpu(sagemaker_session, ecr_image, instance_type, dist_cpu_backend, py_version):
    instance_type = instance_type or 'ml.c4.xlarge'
    _test_mnist_distributed(sagemaker_session, ecr_image, instance_type, dist_cpu_backend)


@pytest.mark.skip_cpu
@pytest.mark.skip_py2
def test_mnist_distributed_gpu(sagemaker_session, ecr_image, instance_type, dist_gpu_backend, py_version):
    instance_type = instance_type or 'ml.p2.xlarge'
    _test_mnist_distributed(sagemaker_session, ecr_image, instance_type, dist_gpu_backend)


def _test_mnist_distributed(sagemaker_session, ecr_image, instance_type):
    use_gpu = instance_type.startswith('ml.p') or instance_type.startswith('ml.g')
    model_dir = model_gpu_dir if use_gpu else model_cpu_dir

    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorchModel('file://{}'.format(model_dir),
                               'SageMakerRole',
                               mnist_script,
                               ecr_image,
                               sagemaker_session)

    with timeout_and_delete_endpoint(estimator=pytorch, minutes=30):
        predictor = pytorch.deploy(initial_instance_count=1, instance_type=instance_type)

        batch_size = 100
        data = np.random.rand(batch_size, 1, 28, 28).astype(np.float32)
        output = predictor.predict(data)

        assert output.shape == (batch_size, 10)
