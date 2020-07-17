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

from mock import patch


@patch('sagemaker_pytorch_serving_container.torchserve.start_torchserve')
def test_hosting_start(start_torchserve):
    from sagemaker_pytorch_serving_container import serving

    serving.main()
    start_torchserve.assert_called()


def test_retry_if_error():
    from sagemaker_pytorch_serving_container import serving
    serving._retry_if_error(Exception)
