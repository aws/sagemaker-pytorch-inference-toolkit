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
"""This module contains functionality to configure and start the
multi-model server."""
from __future__ import absolute_import

import os
import re
import subprocess
import sys

import boto3

from sagemaker_inference import logging
from sagemaker_inference.environment import code_dir

logging.configure_logger()
logger = logging.get_logger()

REQUIREMENTS_PATH = os.path.join(code_dir, "requirements.txt")


def _install_requirements():
    logger.info("installing packages from requirements.txt...")
    pip_install_cmd = [sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_PATH]
    if os.getenv("CA_REPOSITORY_ARN"):
        index = _get_codeartifact_index()
        pip_install_cmd.append("-i")
        pip_install_cmd.append(index)
    try:
        subprocess.check_call(pip_install_cmd)
    except subprocess.CalledProcessError:
        logger.error("failed to install required packages, exiting")
        raise ValueError("failed to install required packages")


def _get_codeartifact_index():
    """
    Build the authenticated codeartifact index url
    https://docs.aws.amazon.com/codeartifact/latest/ug/python-configure-pip.html
    https://docs.aws.amazon.com/service-authorization/latest/reference/list_awscodeartifact.html#awscodeartifact-resources-for-iam-policies
    :return: authenticated codeartifact index url
    """
    repository_arn = os.getenv("CA_REPOSITORY_ARN")
    arn_regex = (
        "arn:(?P<partition>[^:]+):codeartifact:(?P<region>[^:]+):(?P<account>[^:]+)"
        ":repository/(?P<domain>[^/]+)/(?P<repository>.+)"
    )
    m = re.match(arn_regex, repository_arn)
    if not m:
        raise Exception("invalid CodeArtifact repository arn {}".format(repository_arn))
    domain = m.group("domain")
    owner = m.group("account")
    repository = m.group("repository")
    region = m.group("region")

    logger.info(
        "configuring pip to use codeartifact "
        "(domain: %s, domain owner: %s, repository: %s, region: %s)",
        domain,
        owner,
        repository,
        region,
    )
    try:
        client = boto3.client("codeartifact", region_name=region)
        auth_token_response = client.get_authorization_token(domain=domain, domainOwner=owner)
        token = auth_token_response["authorizationToken"]
        endpoint_response = client.get_repository_endpoint(
            domain=domain, domainOwner=owner, repository=repository, format="pypi"
        )
        unauthenticated_index = endpoint_response["repositoryEndpoint"]
        return re.sub(
            "https://",
            "https://aws:{}@".format(token),
            re.sub(
                "{}/?$".format(repository),
                "{}/simple/".format(repository),
                unauthenticated_index,
            ),
        )
    except Exception:
        logger.error("failed to configure pip to use codeartifact")
        raise Exception("failed to configure pip to use codeartifact")
