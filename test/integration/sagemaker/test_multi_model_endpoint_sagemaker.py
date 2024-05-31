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
import boto3
import pytest
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.predictor import Predictor
from sagemaker.multidatamodel import MultiDataModel
import requests

from integration import resnet18_tar, traced_resnet18_tar, resnet18_script


@pytest.fixture(scope='module', name='resnet18_filename')
def fixture_resnet18_filename():
    return resnet18_tar.split('/')[-1]


@pytest.fixture(scope='module', name='traced_resnet18_filename')
def fixture_traced_resnet18_filename():
    return traced_resnet18_tar.split('/')[-1]


@pytest.fixture(scope='module', name='s3')
def fixture_s3():
    return boto3.client('s3')


@pytest.fixture(scope='module', name='bucket')
def fixture_bucket(region, aws_id):
    return 'sagemaker-{}-{}'.format(region, aws_id)


@pytest.fixture(scope='module', name='mme')
def fixture_mme_endpoint(
    sagemaker_session, image_uri, region, use_gpu, resnet18_filename, traced_resnet18_filename, s3, bucket
):
    try:
        instance_type = 'ml.g4dn.xlarge' if use_gpu else 'ml.c5.xlarge'
        endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-pytorch-serving")

        model_data = sagemaker_session.upload_data(
            path=resnet18_tar,
            key_prefix="sagemaker-pytorch-serving/mme_models"
        )

        model = PyTorchModel(
            model_data=model_data,
            role='SageMakerRole',
            entry_point=resnet18_script,
            image_uri=image_uri,
            sagemaker_session=sagemaker_session
        )

        model_data_prefix = 's3://{}/sagemaker-pytorch-serving/mme_models/'.format(bucket)

        mme = MultiDataModel(
            name=endpoint_name,
            model_data_prefix=model_data_prefix,
            model=model,
            sagemaker_session=sagemaker_session
        )

        mme.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            serializer=sagemaker.serializers.JSONSerializer(),
            deserializer=sagemaker.deserializers.JSONDeserializer()
        )

        yield mme

    finally:

        delete_models([resnet18_filename, traced_resnet18_filename], s3, bucket)
        sagemaker_session.delete_endpoint(endpoint_name)


def delete_models(filenames, s3, bucket):
    for filename in filenames:
        s3.delete_object(Bucket=bucket, Key='sagemaker-pytorch-serving/mme_models/{}'.format(filename))


def add_models(model_tar_files, mme):
    for model_tar in model_tar_files:
        mme.add_model(model_tar)


@pytest.fixture(scope='module', name='predictor', autouse=True)
def fixture_predictor(mme, sagemaker_session):
    return Predictor(endpoint_name=mme.endpoint_name, sagemaker_session=sagemaker_session)

# This test checks that only the ResNet-18 model in present in the multi-model endpoint.


def test_check_only_resnet18_in_mme(mme, resnet18_filename):
    model_list = list(mme.list_models())
    assert resnet18_filename in model_list
    assert len(model_list) == 1

# This test checks that only the traced ResNet-18 model in present in the multi-model endpoint.


def test_check_only_traced_resnet18_in_mme(resnet18_filename, traced_resnet18_filename, mme, s3, bucket):
    delete_models([resnet18_filename], s3, bucket)
    add_models([traced_resnet18_tar], mme)
    model_list = list(mme.list_models())
    assert traced_resnet18_filename in model_list
    assert len(model_list) == 1

# This test checks that both the ResNet-18 and traced ResNet-18 models are present in the multi-model endpoint.


def test_check_both_models_in_mme(resnet18_filename, traced_resnet18_filename, mme):
    add_models([resnet18_tar], mme)
    model_list = list(mme.list_models())
    for filename in [resnet18_filename, traced_resnet18_filename]:
        assert filename in model_list
    assert len(model_list) == 2

# This test checks that no models are present in the multi-model endpoint.


def test_no_models_in_mme(resnet18_filename, traced_resnet18_filename, mme, s3, bucket):
    delete_models([resnet18_filename, traced_resnet18_filename], s3, bucket)
    model_list = list(mme.list_models())
    assert len(model_list) == 0

# This test checks the invocation outputs from both the ResNet-18 and traced ResNet-18 models


def test_invocation(resnet18_filename, traced_resnet18_filename, mme, predictor):
    add_models([resnet18_tar, traced_resnet18_tar], mme)
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

    resnet18_response = predictor.predict(payload, target_model=resnet18_filename)
    resnet18_result = json.loads(resnet18_response.decode())
    assert len(resnet18_result) == 1000

    traced_resnet18_response = predictor.predict(payload, target_model=traced_resnet18_filename)
    traced_resnet18_result = json.loads(traced_resnet18_response.decode())
    assert len(traced_resnet18_result) == 1000
