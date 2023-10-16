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
import torch
import csv
import threading

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_device_info_csv(filename, context):
    file_path = os.path.join(SCRIPT_DIR, filename)
    pid = os.getpid()
    threadid = threading.current_thread().ident
    
    device = torch.device("cuda:" + str(context.system_properties.get("gpu_id")))
    device_id = str(device)[-1]

    data = [device_id, pid, threadid]
    
    with open(file_path, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(data)

    return


def model_fn(model_dir, context):
    create_device_info_csv("model_fn_device_info.csv", context)
    return 'model'


def input_fn(data, content_type , context):
    create_device_info_csv("input_fn_device_info.csv", context)
    return data


def predict_fn(data, model, context):
    create_device_info_csv("predict_fn_device_info.csv", context)
    return b'output'


def output_fn(prediction, accept, context):
    create_device_info_csv("output_fn_device_info.csv", context)
    return prediction
