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

import json
import numpy as np
import pytest
import requests
import sagemaker
from sagemaker.predictor import RealTimePredictor
from sagemaker.pytorch import PyTorchModel, PyTorchPredictor

from integration import (
    default_model_script,
    default_model_tar,
    default_traced_resnet_script,
    default_model_traced_resnet18_tar,
    default_model_eia_script,
    default_model_eia_tar,
)
from integration.sagemaker.timeout import timeout_and_delete_endpoint


@pytest.mark.cpu_test
def test_default_inference_cpu(sagemaker_session, image_uri, instance_type):
    instance_type = instance_type or "ml.c4.xlarge"
    # Scripted model is serialized with torch.jit.save().
    # Default inference test doesn't need to instantiate model definition
    _test_default_inference(
        sagemaker_session, image_uri, instance_type, default_model_tar, default_model_script
    )


@pytest.mark.gpu_test
def test_default_inference_gpu(sagemaker_session, image_uri, instance_type):
    instance_type = instance_type or "ml.p2.xlarge"
    # Scripted model is serialized with torch.jit.save().
    # Default inference test doesn't need to instantiate model definition
    _test_default_inference(
        sagemaker_session, image_uri, instance_type, default_model_tar, default_model_script
    )


@pytest.mark.skip(
    reason="Latest EIA version - 1.5.1 uses mms. Enable when EIA images use torchserve"
)
@pytest.mark.eia_test
def test_default_inference_eia(sagemaker_session, image_uri, instance_type, accelerator_type):
    instance_type = instance_type or "ml.c4.xlarge"
    # Scripted model is serialized with torch.jit.save().
    # Default inference test doesn't need to instantiate model definition
    _test_default_inference(
        sagemaker_session,
        image_uri,
        instance_type,
        default_model_eia_tar,
        default_model_eia_script,
        accelerator_type=accelerator_type,
    )


@pytest.mark.gpu_test
def test_default_inference_any_model_name_gpu(sagemaker_session, image_uri, instance_type):
    instance_type = instance_type or "ml.p2.xlarge"
    # Scripted model is serialized with torch.jit.save().
    # Default inference test doesn't need to instantiate model definition
    _test_default_inference(
        sagemaker_session,
        image_uri,
        instance_type,
        default_model_traced_resnet18_tar,
        default_traced_resnet_script,
    )


def _test_default_inference(
    sagemaker_session, image_uri, instance_type, model_tar, mnist_script, accelerator_type=None
):
    endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-pytorch-serving")

    model_data = sagemaker_session.upload_data(
        path=model_tar,
        key_prefix="sagemaker-pytorch-serving/models",
    )

    pytorch = PyTorchModel(
        model_data=model_data,
        role="SageMakerRole",
        predictor_cls=RealTimePredictor if not accelerator_type else PyTorchPredictor,
        entry_point=mnist_script,
        image=image_uri,
        sagemaker_session=sagemaker_session,
    )
    with timeout_and_delete_endpoint(endpoint_name, sagemaker_session, minutes=30):
        predictor = pytorch.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            accelerator_type=accelerator_type,
            endpoint_name=endpoint_name,
        )

        if accelerator_type:
            batch_size = 100
            data = np.random.rand(batch_size, 1, 28, 28).astype(np.float32)
            output = predictor.predict(data)
            assert output.shape == (batch_size, 10)
        else:
            image_url = (
                "https://raw.githubusercontent.com/aws/amazon-sagemaker-examples/master/"
                "sagemaker_neo_compilation_jobs/pytorch_torchvision/cat.jpg"
            )
            img_data = requests.get(image_url).content
            with open("cat.jpg", "wb") as file_obj:
                file_obj.write(img_data)
            with open("cat.jpg", "rb") as f:
                payload = f.read()
                payload = bytearray(payload)
            response = predictor.predict(payload)
            result = json.loads(response.decode())
            assert len(result) == 1000
