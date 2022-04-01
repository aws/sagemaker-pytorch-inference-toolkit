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
import signal
import subprocess
import types

from mock import Mock, patch
import pytest

from sagemaker_inference import environment
from sagemaker_pytorch_serving_container import torchserve
from sagemaker_pytorch_serving_container.torchserve import TS_NAMESPACE, REQUIREMENTS_PATH

PYTHON_PATH = "python_path"
DEFAULT_CONFIGURATION = "default_configuration"


@patch("subprocess.call")
@patch("subprocess.Popen")
@patch("sagemaker_pytorch_serving_container.torchserve._retrieve_ts_server_process")
@patch("sagemaker_pytorch_serving_container.torchserve._add_sigterm_handler")
@patch("sagemaker_pytorch_serving_container.torchserve._install_requirements")
@patch("sagemaker_pytorch_serving_container.torchserve._create_torchserve_config_file")
@patch("sagemaker_pytorch_serving_container.torchserve._set_python_path")
@patch("os.path.exists", return_value=True)
def test_start_torchserve_default_service_handler(
    exists,
    set_python_path,
    create_config,
    install_requirements,
    sigterm,
    retrieve,
    subprocess_popen,
    subprocess_call,
):
    torchserve.start_torchserve()

    exists.assert_called_once_with(torchserve.MODEL_STORE)
    set_python_path.assert_called_once_with()
    create_config.assert_called_once_with()
    exists.assert_called_once_with(REQUIREMENTS_PATH)
    install_requirements.assert_called_once_with()

    ts_model_server_cmd = [
        "torchserve",
        "--start",
        "--model-store",
        torchserve.MODEL_STORE,
        "--ts-config",
        torchserve.TS_CONFIG_FILE,
        "--log-config",
        torchserve.DEFAULT_TS_LOG_FILE,
        "--models",
        "model=/opt/ml/model"
    ]

    subprocess_popen.assert_called_once_with(ts_model_server_cmd)
    sigterm.assert_called_once_with(retrieve.return_value)


@patch("subprocess.call")
@patch("subprocess.Popen")
@patch("sagemaker_pytorch_serving_container.torchserve._retrieve_ts_server_process")
@patch("sagemaker_pytorch_serving_container.torchserve._add_sigterm_handler")
@patch("sagemaker_pytorch_serving_container.torchserve._install_requirements")
@patch("os.path.exists", return_value=True)
@patch("sagemaker_pytorch_serving_container.torchserve._create_torchserve_config_file")
@patch("sagemaker_pytorch_serving_container.torchserve._set_python_path")
def test_start_torchserve_default_service_handler_multi_model(
    set_python_path,
    create_config,
    exists,
    install_requirements,
    sigterm,
    retrieve,
    subprocess_popen,
    subprocess_call,
):
    torchserve.ENABLE_MULTI_MODEL = True
    torchserve.start_torchserve()
    torchserve.ENABLE_MULTI_MODEL = False

    set_python_path.assert_called_once_with()
    create_config.assert_called_once_with()
    exists.assert_called_once_with(REQUIREMENTS_PATH)
    install_requirements.assert_called_once_with()

    ts_model_server_cmd = [
        "torchserve",
        "--start",
        "--model-store",
        torchserve.MODEL_STORE,
        "--ts-config",
        torchserve.TS_CONFIG_FILE,
        "--log-config",
        torchserve.DEFAULT_TS_LOG_FILE,
        "--models",
        "model=/opt/ml/model"
    ]

    subprocess_popen.assert_called_once_with(ts_model_server_cmd)
    sigterm.assert_called_once_with(retrieve.return_value)


@patch.dict(os.environ, {torchserve.PYTHON_PATH_ENV: PYTHON_PATH}, clear=True)
def test_set_existing_python_path():
    torchserve._set_python_path()

    code_dir_path = "{}:{}".format(environment.code_dir, PYTHON_PATH)

    assert os.environ[torchserve.PYTHON_PATH_ENV] == code_dir_path


@patch.dict(os.environ, {}, clear=True)
def test_new_python_path():
    torchserve._set_python_path()

    code_dir_path = environment.code_dir

    assert os.environ[torchserve.PYTHON_PATH_ENV] == code_dir_path


@patch("sagemaker_pytorch_serving_container.torchserve._generate_ts_config_properties")
@patch("sagemaker_inference.utils.write_file")
def test_create_torchserve_config_file(write_file, generate_ts_config_props):
    torchserve._create_torchserve_config_file()

    write_file.assert_called_once_with(
        torchserve.TS_CONFIG_FILE, generate_ts_config_props.return_value
    )


@patch("sagemaker_inference.utils.read_file", return_value=DEFAULT_CONFIGURATION)
@patch("sagemaker_inference.environment.Environment")
def test_generate_ts_config_properties(env, read_file):
    model_server_timeout = "torchserve_timeout"
    model_server_workers = "torchserve_workers"
    http_port = "http_port"

    env.return_value.model_server_timeout = model_server_timeout
    env.return_value.model_sever_workerse = model_server_workers
    env.return_value.inference_http_port = http_port

    ts_config_properties = torchserve._generate_ts_config_properties()

    inference_address = "inference_address=http://0.0.0.0:{}\n".format(http_port)
    server_timeout = "default_response_timeout={}\n".format(model_server_timeout)

    read_file.assert_called_once_with(torchserve.DEFAULT_TS_CONFIG_FILE)

    assert ts_config_properties.startswith(DEFAULT_CONFIGURATION)
    assert inference_address in ts_config_properties
    assert server_timeout in ts_config_properties


@patch("sagemaker_inference.utils.read_file", return_value=DEFAULT_CONFIGURATION)
@patch("sagemaker_inference.environment.Environment")
def test_generate_ts_config_properties_default_workers(env, read_file):
    env.return_value.model_server_workers = None

    ts_config_properties = torchserve._generate_ts_config_properties()

    workers = "default_workers_per_model={}".format(None)

    read_file.assert_called_once_with(torchserve.DEFAULT_TS_CONFIG_FILE)

    assert ts_config_properties.startswith(DEFAULT_CONFIGURATION)
    assert workers not in ts_config_properties


@patch("sagemaker_inference.utils.read_file", return_value=DEFAULT_CONFIGURATION)
@patch("sagemaker_inference.environment.Environment")
def test_generate_ts_config_properties_multi_model(env, read_file):
    env.return_value.model_server_workers = None

    torchserve.ENABLE_MULTI_MODEL = True
    ts_config_properties = torchserve._generate_ts_config_properties()
    torchserve.ENABLE_MULTI_MODEL = False

    workers = "default_workers_per_model={}".format(None)

    read_file.assert_called_once_with(torchserve.MME_TS_CONFIG_FILE)

    assert ts_config_properties.startswith(DEFAULT_CONFIGURATION)
    assert workers not in ts_config_properties


@patch("signal.signal")
def test_add_sigterm_handler(signal_call):
    ts = Mock()

    torchserve._add_sigterm_handler(ts)

    mock_calls = signal_call.mock_calls
    first_argument = mock_calls[0][1][0]
    second_argument = mock_calls[0][1][1]

    assert len(mock_calls) == 1
    assert first_argument == signal.SIGTERM
    assert isinstance(second_argument, types.FunctionType)


@patch("subprocess.check_call")
def test_install_requirements(check_call):
    torchserve._install_requirements()
    for i in ['pip', 'install', '-r', '/opt/ml/model/code/requirements.txt']:
        assert i in check_call.call_args.args[0]


@patch("subprocess.check_call", side_effect=subprocess.CalledProcessError(0, "cmd"))
def test_install_requirements_installation_failed(check_call):
    with pytest.raises(ValueError) as e:
        torchserve._install_requirements()
    assert "failed to install required packages" in str(e.value)


@patch("retrying.Retrying.should_reject", return_value=False)
@patch("psutil.process_iter")
def test_retrieve_ts_server_process(process_iter, retry):
    server = Mock()
    server.cmdline.return_value = TS_NAMESPACE

    processes = list()
    processes.append(server)

    process_iter.return_value = processes

    process = torchserve._retrieve_ts_server_process()

    assert process == server


@patch("retrying.Retrying.should_reject", return_value=False)
@patch("psutil.process_iter", return_value=list())
def test_retrieve_ts_server_process_no_server(process_iter, retry):
    with pytest.raises(Exception) as e:
        torchserve._retrieve_ts_server_process()

    assert "Torchserve model server was unsuccessfully started" in str(e.value)


@patch("retrying.Retrying.should_reject", return_value=False)
@patch("psutil.process_iter")
def test_retrieve_ts_server_process_too_many_servers(process_iter, retry):
    server = Mock()
    second_server = Mock()
    server.cmdline.return_value = TS_NAMESPACE
    second_server.cmdline.return_value = TS_NAMESPACE

    processes = list()
    processes.append(server)
    processes.append(second_server)

    process_iter.return_value = processes

    with pytest.raises(Exception) as e:
        torchserve._retrieve_ts_server_process()

    assert "multiple ts model servers are not supported" in str(e.value)
