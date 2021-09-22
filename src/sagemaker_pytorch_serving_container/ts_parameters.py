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
"""This module contains string constants that define inference toolkit
parameters."""
from __future__ import absolute_import

MODEL_SERVER_BATCH_SIZE = "SAGEMAKER_TS_BATCH_SIZE"  # type: str
MODEL_SERVER_MAX_BATCH_DELAY = "SAGEMAKER_TS_MAX_BATCH_DELAY"  # type: str
MODEL_SERVER_MIN_WORKERS = "SAGEMAKER_TS_MIN_WORKERS"  # type: str
MODEL_SERVER_MAX_WORKERS = "SAGEMAKER_TS_MAX_WORKERS"  # type: str
MODEL_SERVER_RESPONSE_TIMEOUT = "SAGEMAKER_TS_RESPONSE_TIMEOUT"  # type: str
