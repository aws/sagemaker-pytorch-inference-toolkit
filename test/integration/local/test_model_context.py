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

import os
import subprocess
import sys
import time

import pytest
import requests
import torch
from concurrent.futures import ThreadPoolExecutor
import csv

from integration import model_gpu_context_dir

BASE_URL = "http://0.0.0.0:8080/"
PING_URL = BASE_URL + "ping"
INVOCATION_URL = BASE_URL + "models/model/invoke"
GPU_COUNT = torch.cuda.device_count()
DEVICE_IDS_EXPECTED = [i for i in range(GPU_COUNT)]


def send_request(input_data, headers):
    requests.post(INVOCATION_URL, data=input_data, headers=headers)


def read_csv(filename):
    data = {}
    with open(os.path.join(model_gpu_context_dir, 'code', filename), 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            device_id, pid, threadid = row
            if device_id in data:
                continue
            data[int(device_id)] = {'pid': pid, 'threadid': threadid}
    return data


@pytest.fixture(scope="module", autouse=True)
def container(image_uri):
    try:
        if 'cpu' in image_uri:
            pytest.skip("Skipping because tests running on CPU instance")

        command = (
            "docker run --gpus=all -p 8080:8080 "
            "--name sagemaker-pytorch-inference-toolkit-context-test "
            "-v {}:/opt/ml/model "
            "{} serve"
        ).format(model_gpu_context_dir, image_uri)

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
        time.sleep(60)
        yield proc.pid

    finally:
        if 'cpu' in image_uri:
            pytest.skip("Skipping because tests running on CPU instance")
        subprocess.check_call("docker rm -f sagemaker-pytorch-inference-toolkit-context-test".split())


@pytest.fixture(scope="module", autouse=True)
def inference_requests():
    headers = {"Content-Type": "application/json"}
    with ThreadPoolExecutor(max_workers=GPU_COUNT) as executor:
        for i in range(32):
            executor.submit(send_request, b'input', headers)
    time.sleep(60)
    yield


@pytest.fixture(scope="module", name="model_fn_device_info")
def model_fn_device_info():
    return read_csv("model_fn_device_info.csv")


@pytest.fixture(scope="module", name="input_fn_device_info")
def input_fn_device_info():
    return read_csv("input_fn_device_info.csv")


@pytest.fixture(scope="module", name="predict_fn_device_info")
def predict_fn_device_info():
    return read_csv("predict_fn_device_info.csv")


@pytest.fixture(scope="module", name="output_fn_device_info")
def output_fn_device_info():
    return read_csv("output_fn_device_info.csv")


def test_context_all_device_ids(
    model_fn_device_info, input_fn_device_info, predict_fn_device_info, output_fn_device_info
):
    for device_id in DEVICE_IDS_EXPECTED:
        assert device_id in model_fn_device_info
        assert device_id in input_fn_device_info
        assert device_id in predict_fn_device_info
        assert device_id in output_fn_device_info


def test_same_pid_threadid(
    model_fn_device_info, input_fn_device_info, predict_fn_device_info, output_fn_device_info
):
    for device_id in DEVICE_IDS_EXPECTED:
        pid_model_fn = model_fn_device_info[device_id]['pid']
        threadid_model_fn = model_fn_device_info[device_id]['threadid']

        pid_input_fn = input_fn_device_info[device_id]['pid']
        threadid_input_fn = input_fn_device_info[device_id]['threadid']

        pid_predict_fn = predict_fn_device_info[device_id]['pid']
        threadid_predict_fn = predict_fn_device_info[device_id]['threadid']

        pid_output_fn = output_fn_device_info[device_id]['pid']
        threadid_output_fn = output_fn_device_info[device_id]['threadid']

        assert pid_model_fn == pid_input_fn == pid_output_fn == pid_predict_fn
        assert threadid_model_fn == threadid_input_fn == threadid_output_fn == threadid_predict_fn
