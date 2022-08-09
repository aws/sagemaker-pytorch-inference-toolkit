# Copyright 2019-2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from mock import Mock, patch
import pytest

from sagemaker_inference import environment
from sagemaker_pytorch_serving_container.default_pytorch_inference_handler import DefaultPytorchInferenceHandler
from sagemaker_pytorch_serving_container.transformer import PyTorchTransformer


INPUT_DATA = "input_data"
CONTENT_TYPE = "content_type"
ACCEPT = "accept"
RESULT = "result"
MODEL = "foo"

PREPROCESSED_DATA = "preprocessed_data"
PREDICT_RESULT = "prediction_result"
PROCESSED_RESULT = "processed_result"


def test_default_transformer():
    transformer = PyTorchTransformer()

    assert isinstance(transformer._default_inference_handler, DefaultPytorchInferenceHandler)
    assert transformer._initialized is False
    assert transformer._environment is None
    assert transformer._pre_model_fn is None
    assert transformer._model_warmup_fn is None
    assert transformer._model is None
    assert transformer._model_fn is None
    assert transformer._transform_fn is None
    assert transformer._input_fn is None
    assert transformer._predict_fn is None
    assert transformer._output_fn is None
    assert transformer._context is None


def test_transformer_with_custom_default_inference_handler():
    default_inference_handler = Mock()

    transformer = PyTorchTransformer(default_inference_handler)

    assert transformer._default_inference_handler == default_inference_handler
    assert transformer._initialized is False
    assert transformer._environment is None
    assert transformer._pre_model_fn is None
    assert transformer._model_warmup_fn is None
    assert transformer._model is None
    assert transformer._model_fn is None
    assert transformer._transform_fn is None
    assert transformer._input_fn is None
    assert transformer._predict_fn is None
    assert transformer._output_fn is None
    assert transformer._context is None


@pytest.mark.parametrize("accept_key", ["Accept", "accept"])
@patch("sagemaker_inference.utils.retrieve_content_type_header", return_value=CONTENT_TYPE)
@patch("sagemaker_pytorch_serving_container.transformer.PyTorchTransformer.validate_and_initialize")
def test_transform(validate, retrieve_content_type_header, accept_key):
    data = [{"body": INPUT_DATA}]
    context = Mock()
    request_processor = Mock()
    transform_fn = Mock(return_value=RESULT)

    context.request_processor = [request_processor]
    request_property = {accept_key: ACCEPT}
    request_processor.get_request_properties.return_value = request_property

    transformer = PyTorchTransformer()
    transformer._model = MODEL
    transformer._transform_fn = transform_fn
    transformer._context = context

    result = transformer.transform(data, context)

    validate.assert_called_once()
    retrieve_content_type_header.assert_called_once_with(request_property)
    transform_fn.assert_called_once_with(MODEL, INPUT_DATA, CONTENT_TYPE, ACCEPT, context)
    context.set_response_content_type.assert_called_once_with(0, ACCEPT)
    assert isinstance(result, list)
    assert result[0] == RESULT


@patch("sagemaker_pytorch_serving_container.transformer.PyTorchTransformer._validate_user_module_and_set_functions")
@patch("sagemaker_inference.environment.Environment")
def test_validate_and_initialize(env, validate_user_module):
    transformer = PyTorchTransformer()

    model_fn = Mock()
    context = Mock()
    transformer._model_fn = model_fn

    assert transformer._initialized is False
    assert transformer._context is None

    transformer.validate_and_initialize(context=context)

    assert transformer._initialized is True
    assert transformer._context == context

    transformer.validate_and_initialize()

    model_fn.assert_called_once_with(environment.model_dir, context)
    env.assert_called_once_with()
    validate_user_module.assert_called_once_with()


def test_default_transform_fn():
    transformer = PyTorchTransformer()
    context = Mock()
    transformer._context = context

    input_fn = Mock(return_value=PREPROCESSED_DATA)
    predict_fn = Mock(return_value=PREDICT_RESULT)
    output_fn = Mock(return_value=PROCESSED_RESULT)

    transformer._input_fn = input_fn
    transformer._predict_fn = predict_fn
    transformer._output_fn = output_fn

    result = transformer._default_transform_fn(MODEL, INPUT_DATA, CONTENT_TYPE, ACCEPT)

    input_fn.assert_called_once_with(INPUT_DATA, CONTENT_TYPE, context)
    predict_fn.assert_called_once_with(PREPROCESSED_DATA, MODEL, context)
    output_fn.assert_called_once_with(PREDICT_RESULT, ACCEPT, context)
    assert result == PROCESSED_RESULT


def test_run_handle_function():
    def three_inputs_func(a, b, c):
        pass

    three_inputs_mock = Mock(spec=three_inputs_func)
    a = Mock()
    b = Mock()
    context = Mock()

    transformer = PyTorchTransformer()
    transformer._context = context
    transformer._run_handle_function(three_inputs_mock, a, b)
    three_inputs_mock.assert_called_with(a, b, context)
