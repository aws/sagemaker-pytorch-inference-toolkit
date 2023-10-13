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
import json

from utils import file_utils

resources_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources'))
model_gpu_context_dir = os.path.join(resources_path, 'model_gpu_context')
mnist_path = os.path.join(resources_path, 'mnist')
data_dir = os.path.join(mnist_path, 'data')
training_dir = os.path.join(data_dir, 'training')

all_models_info_json = os.path.abspath(os.path.join(os.path.dirname(__file__), 'all_models_info.json'))

with open(all_models_info_json, 'r') as json_file:
    all_models_info = json.load(json_file)

for model_name in all_models_info.keys():
    model_info = all_models_info[model_name]

    dir_path = model_info['dir_path']
    model_dir = os.path.join(resources_path, *dir_path)
    setattr(__import__('integration'), model_name + '_dir', model_dir)

    script_name = model_info['script_name']
    model = model_info['model']
    if 'filename' in model_info:
        filename = model_info['filename']
    else:
        filename = 'model.tar.gz'
    if 'code_path' in model_info:
        code_path = model_info['code_path']
        script_path = model_info['code_path']
    else:
        code_path = ''
        script_path = None
    if 'requirements' in model_info:
        requirements = os.path.join(model_dir, code_path, 'requirements.txt')
    else:
        requirements = None

    model_script = os.path.join(model_dir, code_path, script_name)
    model_tar = file_utils.make_tarfile(model_script,
                                        os.path.join(model_dir, model),
                                        model_dir,
                                        filename=filename,
                                        script_path=script_path,
                                        requirements=requirements)

    setattr(__import__('integration'), model_name + '_script', model_script)
    setattr(__import__('integration'), model_name + '_tar', model_tar)

ROLE = 'dummy/unused-role'
DEFAULT_TIMEOUT = 20
PYTHON3 = 'py3'

RESOURCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources'))

# These regions have some p2 and p3 instances, but not enough for automated testing
NO_P2_REGIONS = ['ca-central-1', 'eu-central-1', 'eu-west-2', 'us-west-1', 'eu-west-3',
                 'eu-north-1', 'sa-east-1', 'ap-east-1']
NO_P3_REGIONS = ['ap-southeast-1', 'ap-southeast-2', 'ap-south-1', 'ca-central-1',
                 'eu-central-1', 'eu-west-2', 'us-west-1', 'eu-west-3', 'eu-north-1',
                 'sa-east-1', 'ap-east-1']
