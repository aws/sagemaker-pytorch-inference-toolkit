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
"""This module contains functionality that provides access to system
characteristics, environment variables and configuration settings.
"""

from __future__ import absolute_import

from sagemaker_pytorch_serving_container import ts_parameters

import os
import logging

logger = logging.getLogger()

DEFAULT_TS_BATCH_SIZE = 1
DEFAULT_TS_MAX_BATCH_DELAY = 100
DEFAULT_TS_MIN_WORKERS = 1
DEFAULT_TS_MAX_WORKERS = 1
DEFAULT_TS_RESPONSE_TIMEOUT = 60


class TorchServeEnvironment():
    """Provides access to aspects of the torchserve environment relevant to serving containers,
    including system characteristics, environment variables and configuration settings.

    The Environment is a read-only snapshot of the container environment.
    It is a dictionary-like object, allowing any builtin function that works with dictionary.

    Attributes:
        batch_size (int): This is the maximum batch size in ms that a model is expected to handle
        max_batch_delay (int): This is the maximum batch delay time TorchServe waits to receive
        batch_size number of requests. If TorchServe doesn’t receive batch_size number of requests
        before this timer time’s out, it sends what ever requests that were received to the model handler
        min_workers (int): Minimum number of workers that torchserve is allowed to scale down to
        max_workers (int): Minimum number of workers that torchserve is allowed to scale up to
        response_timeout (int): Time delay after which inference will timeout in absence of a response
    """
    def __init__(self):
        self._batch_size = int(os.environ.get(ts_parameters.MODEL_SERVER_BATCH_SIZE, DEFAULT_TS_BATCH_SIZE))
        self._max_batch_delay = int(os.environ.get(ts_parameters.MODEL_SERVER_MAX_BATCH_DELAY,
                                    DEFAULT_TS_MAX_BATCH_DELAY))
        self._min_workers = int(os.environ.get(ts_parameters.MODEL_SERVER_MIN_WORKERS, DEFAULT_TS_MIN_WORKERS))
        self._max_workers = int(os.environ.get(ts_parameters.MODEL_SERVER_MAX_WORKERS, DEFAULT_TS_MAX_WORKERS))
        self._response_timeout = int(os.environ.get(ts_parameters.MODEL_SERVER_RESPONSE_TIMEOUT,
                                                    DEFAULT_TS_RESPONSE_TIMEOUT))

    def is_env_set(self):  # type: () -> bool
        """bool: whether or not the environment variables have been set"""
        ts_env_list = [ts_parameters.MODEL_SERVER_BATCH_SIZE, ts_parameters.MODEL_SERVER_MAX_BATCH_DELAY,
                       ts_parameters.MODEL_SERVER_MIN_WORKERS, ts_parameters.MODEL_SERVER_MAX_WORKERS,
                       ts_parameters.MODEL_SERVER_RESPONSE_TIMEOUT]
        if any(env in ts_env_list for env in os.environ):
            return True

    @property
    def batch_size(self):  # type: () -> int
        """int: number of requests to batch before running inference on the server"""
        return self._batch_size

    @property
    def max_batch_delay(self):  # type: () -> int
        """int: time delay in milliseconds, to wait for incoming requests to be batched,
        before running inference on the server
        """
        return self._max_batch_delay

    @property
    def min_workers(self):  # type:() -> int
        """int: minimum number of worker for model
        """
        return self._min_workers

    @property
    def max_workers(self):  # type() -> int
        """int: maximum number of workers for model
        """
        return self._max_workers

    @property
    def response_timeout(self):  # type() -> int
        """int: time delay after which inference will timeout in absense of a response
        """
        return self._response_timeout
