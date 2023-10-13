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
import subprocess
import sys
import time

import pytest
import requests

from integration import resnet18_dir, traced_resnet18_dir

BASE_URL = "http://0.0.0.0:8080/"
PING_URL = BASE_URL + "ping"
INVOCATION_URL = BASE_URL + "models/{}/invoke"
MODELS_URL = BASE_URL + "models"
DELETE_MODEL_URL = BASE_URL + "models/{}"


@pytest.fixture(scope="module", autouse=True)
def container(image_uri, use_gpu):
    try:
        gpu_option = "--gpus device=0" if use_gpu else ""

        command = (
            "docker run {} "
            "--name sagemaker-pytorch-inference-toolkit-mme-test "
            "-p 8080:8080 "
            "-v {}:/resnet18 "
            "-v {}:/traced_resnet18 "
            "-e SAGEMAKER_MULTI_MODEL=true {} serve"
        ).format(gpu_option, resnet18_dir, traced_resnet18_dir, image_uri)

        proc = subprocess.Popen(command.split(), stdout=sys.stdout, stderr=subprocess.STDOUT)

        attempts = 0
        while attempts < 10:
            time.sleep(3)
            try:
                requests.get(PING_URL)
                break
            except Exception:
                attempts += 1
                pass
        yield proc.pid

    finally:
        subprocess.check_call("docker rm -f sagemaker-pytorch-inference-toolkit-mme-test".split())


def make_list_model_request():
    response = requests.get(MODELS_URL)
    return response.status_code, json.loads(response.content.decode("utf-8"))


def make_load_model_request(data, content_type="application/json"):
    headers = {"Content-Type": content_type}
    response = requests.post(MODELS_URL, data=data, headers=headers)
    return response.status_code, json.loads(response.content.decode("utf-8"))


def make_unload_model_request(model_name):
    response = requests.delete(DELETE_MODEL_URL.format(model_name))
    return response.status_code, json.loads(response.content.decode("utf-8"))


def make_invocation_request(model_name, data, content_type="application/octet-stream"):
    headers = {"Content-Type": content_type}
    response = requests.post(INVOCATION_URL.format(model_name), data=data, headers=headers)
    return response.status_code, json.loads(response.content.decode("utf-8"))


def test_ping():
    res = requests.get(PING_URL)
    assert res.status_code == 200


def test_list_models_empty():
    code, models = make_list_model_request()
    assert code == 200
    assert models["models"] == []


def test_load_models():
    data1 = {"model_name": "resnet18", "url": "/resnet18"}
    code1, content1 = make_load_model_request(data=json.dumps(data1))
    assert code1 == 200
    assert content1["status"] == 'Model "resnet18" Version: 1.0 registered with 1 initial workers'

    code2, content2 = make_list_model_request()
    assert code2 == 200
    assert content2["models"] == [{"modelName": "resnet18", "modelUrl": "/resnet18"}]

    data2 = {"model_name": "traced_resnet18", "url": "/traced_resnet18"}
    code3, content3 = make_load_model_request(data=json.dumps(data2))
    assert code3 == 200
    assert content3["status"] == 'Model "traced_resnet18" Version: 1.0 registered with 1 initial workers'

    code4, content4 = make_list_model_request()
    assert code4 == 200
    assert content4["models"] == [
        {"modelName": "resnet18", "modelUrl": "/resnet18"},
        {"modelName": "traced_resnet18", "modelUrl": "/traced_resnet18"},
    ]


def test_unload_models():
    code1, content1 = make_unload_model_request("resnet18")
    assert code1 == 200
    assert content1["status"] == 'Model "resnet18" unregistered'

    code2, content2 = make_list_model_request()
    assert code2 == 200
    assert content2["models"] == [{"modelName": "traced_resnet18", "modelUrl": "/traced_resnet18"}]


def test_load_non_existing_model():
    data = {"model_name": "banana", "url": "/banana"}
    code, content = make_load_model_request(data=json.dumps(data))
    assert code == 404


def test_unload_non_existing_model():
    # resnet18 is already unloaded
    code, content = make_unload_model_request("resnet18")
    assert code == 404


def test_load_model_multiple_times():
    # traced_resnet18 is already loaded
    data = {"model_name": "traced_resnet18", "url": "traced_resnet18"}
    code, content = make_load_model_request(data=json.dumps(data))
    assert code == 409


def test_invocation():
    data = {"model_name": "resnet18", "url": "/resnet18"}
    code, content = make_load_model_request(data=json.dumps(data))

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

    code1, predictions1 = make_invocation_request("resnet18", payload)
    assert code1 == 200
    assert len(predictions1) == 1000

    code2, predictions2 = make_invocation_request("traced_resnet18", payload)
    assert code2 == 200
    assert len(predictions2) == 1000
