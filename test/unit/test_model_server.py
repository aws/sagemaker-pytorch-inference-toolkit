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
import subprocess
import sys

import botocore.session
from botocore.stub import Stubber
from mock import MagicMock, patch
import pytest

from sagemaker_inference import model_server


@patch("subprocess.check_call")
def test_install_requirements(check_call):
    model_server._install_requirements()

    install_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        "/opt/ml/model/code/requirements.txt",
    ]
    check_call.assert_called_once_with(install_cmd)


@patch("subprocess.check_call", side_effect=subprocess.CalledProcessError(0, "cmd"))
def test_install_requirements_installation_failed(check_call):
    with pytest.raises(ValueError) as e:
        model_server._install_requirements()

    assert "failed to install required packages" in str(e.value)


@patch.dict(os.environ, {"CA_REPOSITORY_ARN": "invalid_arn"}, clear=True)
def test_install_requirements_codeartifact_invalid_arn_installation_failed():
    with pytest.raises(Exception) as e:
        model_server._install_requirements()

    assert "invalid CodeArtifact repository arn invalid_arn" in str(e.value)


@patch("subprocess.check_call")
@patch.dict(
    os.environ,
    {
        "CA_REPOSITORY_ARN": "arn:aws:codeartifact:my_region:012345678900:repository/my_domain/my_repo"
    },
    clear=True,
)
def test_install_requirements_codeartifact(check_call):
    # mock/stub codeartifact client and its responses
    endpoint = "https://domain-012345678900.d.codeartifact.region.amazonaws.com/pypi/my_repo/"
    codeartifact = botocore.session.get_session().create_client(
        "codeartifact", region_name="myregion"
    )
    stubber = Stubber(codeartifact)
    stubber.add_response("get_authorization_token", {"authorizationToken": "the-auth-token"})
    stubber.add_response("get_repository_endpoint", {"repositoryEndpoint": endpoint})
    stubber.activate()

    with patch("boto3.client", MagicMock(return_value=codeartifact)):
        model_server._install_requirements()

    install_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        "/opt/ml/model/code/requirements.txt",
        "-i",
        "https://aws:the-auth-token@domain-012345678900.d.codeartifact.region.amazonaws.com/pypi/my_repo/simple/",
    ]
    check_call.assert_called_once_with(install_cmd)
