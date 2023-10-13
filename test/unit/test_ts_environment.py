# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import os
from mock import patch

from sagemaker_pytorch_serving_container import ts_environment, ts_parameters


@patch.dict(
    os.environ,
    {
        ts_parameters.MODEL_SERVER_BATCH_SIZE: "1",
        ts_parameters.MODEL_SERVER_MAX_BATCH_DELAY: "100",
        ts_parameters.MODEL_SERVER_MIN_WORKERS: "1",
        ts_parameters.MODEL_SERVER_MAX_WORKERS: "4",
        ts_parameters.MODEL_SERVER_RESPONSE_TIMEOUT: "60",
    },
    clear=True,
)
def test_ts_env():
    ts_env = ts_environment.TorchServeEnvironment()

    assert ts_env._batch_size == 1
    assert ts_env._max_batch_delay == 100
    assert ts_env._min_workers == 1
    assert ts_env._max_workers == 4
    assert ts_env._response_timeout == 60
    assert ts_env.is_env_set() is True
