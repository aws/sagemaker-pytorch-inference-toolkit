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
"""This module contains functionality to configure and start Torchserve."""
from __future__ import absolute_import

import os
import signal
import subprocess
import sys

import pkg_resources
import psutil
import logging
from retrying import retry

import sagemaker_pytorch_serving_container
from sagemaker_pytorch_serving_container import ts_environment
from sagemaker_inference import default_handler_service, environment, utils
from sagemaker_inference.environment import code_dir

logger = logging.getLogger()

TS_CONFIG_FILE = os.path.join("/etc", "sagemaker-ts.properties")
DEFAULT_HANDLER_SERVICE = default_handler_service.__name__
DEFAULT_TS_CONFIG_FILE = pkg_resources.resource_filename(
    sagemaker_pytorch_serving_container.__name__, "/etc/default-ts.properties"
)
MME_TS_CONFIG_FILE = pkg_resources.resource_filename(
    sagemaker_pytorch_serving_container.__name__, "/etc/mme-ts.properties"
)
DEFAULT_TS_LOG_FILE = pkg_resources.resource_filename(
    sagemaker_pytorch_serving_container.__name__, "/etc/log4j2.xml"
)
DEFAULT_TS_MODEL_DIRECTORY = os.path.join(os.getcwd(), ".sagemaker", "ts", "models")
DEFAULT_TS_MODEL_NAME = "model"
DEFAULT_TS_CODE_DIR = "code"
DEFAULT_HANDLER_SERVICE = "sagemaker_pytorch_serving_container.handler_service"

ENABLE_MULTI_MODEL = os.getenv("SAGEMAKER_MULTI_MODEL", "false") == "true"
MODEL_STORE = "/" if ENABLE_MULTI_MODEL else DEFAULT_TS_MODEL_DIRECTORY

PYTHON_PATH_ENV = "PYTHONPATH"
REQUIREMENTS_PATH = os.path.join(code_dir, "requirements.txt")
TS_NAMESPACE = "org.pytorch.serve.ModelServer"


def start_torchserve(handler_service=DEFAULT_HANDLER_SERVICE):
    """Configure and start the model server.

    Args:
        handler_service (str): Python path pointing to a module that defines
            a class with the following:

                - A ``handle`` method, which is invoked for all incoming inference
                    requests to the model server.
                - A ``initialize`` method, which is invoked at model server start up
                    for loading the model.

            Defaults to ``sagemaker_pytorch_serving_container.default_handler_service``.

    """

    if ENABLE_MULTI_MODEL:
        if "SAGEMAKER_HANDLER" not in os.environ:
            os.environ["SAGEMAKER_HANDLER"] = handler_service
        _set_python_path()
    else:
        _adapt_to_ts_format(handler_service)

    _create_torchserve_config_file()

    if os.path.exists(REQUIREMENTS_PATH):
        _install_requirements()

    ts_torchserve_cmd = [
        "torchserve",
        "--start",
        "--model-store",
        MODEL_STORE,
        "--ts-config",
        TS_CONFIG_FILE,
        "--log-config",
        DEFAULT_TS_LOG_FILE,
        "--models",
        "model.mar"
    ]

    print(ts_torchserve_cmd)

    logger.info(ts_torchserve_cmd)
    subprocess.Popen(ts_torchserve_cmd)

    ts_process = _retrieve_ts_server_process()

    _add_sigterm_handler(ts_process)

    ts_process.wait()


def _adapt_to_ts_format(handler_service):
    if not os.path.exists(DEFAULT_TS_MODEL_DIRECTORY):
        os.makedirs(DEFAULT_TS_MODEL_DIRECTORY)

    model_archiver_cmd = [
        "torch-model-archiver",
        "--model-name",
        DEFAULT_TS_MODEL_NAME,
        "--handler",
        handler_service,
        "--export-path",
        DEFAULT_TS_MODEL_DIRECTORY,
        "--version",
        "1",
        "--extra-files",
        os.path.join(environment.model_dir)
    ]

    logger.info(model_archiver_cmd)
    subprocess.check_call(model_archiver_cmd)

    _set_python_path()


def _set_python_path():
    # Torchserve handles code execution by appending the export path, provided
    # to the model archiver, to the PYTHONPATH env var.
    # The code_dir has to be added to the PYTHONPATH otherwise the
    # user provided module can not be imported properly.
    if PYTHON_PATH_ENV in os.environ:
        os.environ[PYTHON_PATH_ENV] = "{}:{}".format(environment.code_dir, os.environ[PYTHON_PATH_ENV])
    else:
        os.environ[PYTHON_PATH_ENV] = environment.code_dir


def _create_torchserve_config_file():
    configuration_properties = _generate_ts_config_properties()

    utils.write_file(TS_CONFIG_FILE, configuration_properties)


def _generate_ts_config_properties():
    env = environment.Environment()
    user_defined_configuration = {
        "default_response_timeout": env.model_server_timeout,
        "default_workers_per_model": env.model_server_workers,
        "inference_address": "http://0.0.0.0:{}".format(env.inference_http_port),
        "management_address": "http://0.0.0.0:{}".format(env.management_http_port),
    }

    ts_env = ts_environment.TorchServeEnvironment()

    if ts_env.is_env_set() and not ENABLE_MULTI_MODEL:
        models_string = f'''{{\\
        "{DEFAULT_TS_MODEL_NAME}": {{\\
            "1.0": {{\\
                "defaultVersion": true,\\
                "marName": "{DEFAULT_TS_MODEL_NAME}.mar",\\
                "minWorkers": {ts_env._min_workers},\\
                "maxWorkers": {ts_env._max_workers},\\
                "batchSize": {ts_env._batch_size},\\
                "maxBatchDelay": {ts_env._max_batch_delay},\\
                "responseTimeout": {ts_env._response_timeout}\\
                }}\\
            }}\\
        }}'''
        user_defined_configuration["models"] = models_string
        logger.warn("Sagemaker TS environment variables have been set and will be used "
                    "for single model endpoint.")

    custom_configuration = str()

    for key in user_defined_configuration:
        value = user_defined_configuration.get(key)
        if value:
            custom_configuration += "{}={}\n".format(key, value)

    if ENABLE_MULTI_MODEL:
        default_configuration = utils.read_file(MME_TS_CONFIG_FILE)
    else:
        default_configuration = utils.read_file(DEFAULT_TS_CONFIG_FILE)

    return default_configuration + custom_configuration


def _add_sigterm_handler(ts_process):
    def _terminate(signo, frame):  # pylint: disable=unused-argument
        try:
            os.kill(ts_process.pid, signal.SIGTERM)
        except OSError:
            pass

    signal.signal(signal.SIGTERM, _terminate)


def _install_requirements():
    logger.info("installing packages from requirements.txt...")
    pip_install_cmd = [sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_PATH]

    try:
        subprocess.check_call(pip_install_cmd)
    except subprocess.CalledProcessError:
        logger.exception("failed to install required packages, exiting")
        raise ValueError("failed to install required packages")


# retry for 10 seconds
@retry(stop_max_delay=10 * 1000)
def _retrieve_ts_server_process():
    ts_server_processes = list()

    for process in psutil.process_iter():
        if TS_NAMESPACE in process.cmdline():
            ts_server_processes.append(process)

    if not ts_server_processes:
        raise Exception("Torchserve model server was unsuccessfully started")

    if len(ts_server_processes) > 1:
        raise Exception("multiple ts model servers are not supported")

    return ts_server_processes[0]
