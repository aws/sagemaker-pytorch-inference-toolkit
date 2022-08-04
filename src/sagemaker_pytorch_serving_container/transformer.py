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

import logging
import traceback

from six.moves import http_client
from sagemaker_inference.transformer import Transformer
from sagemaker_inference import content_types, environment, utils
from sagemaker_inference.errors import BaseInferenceToolkitError, GenericInferenceToolkitError


class PTTransformer(Transformer):
    """Represents the execution workflow for handling pytorch inference requests
    sent to the model server.
    """
    def __init__(self, default_inference_handler=None):
        super().__init__(default_inference_handler)
        self._context = None

    def transform(self, data, context):
        """Take a request with input data, deserialize it, make a prediction, and return a
        serialized response.
        Args:
            data (obj): the request data.
            context (obj): metadata on the incoming request data.
        Returns:
            list[obj]: The serialized prediction result wrapped in a list if
                inference is successful. Otherwise returns an error message
                with the context set appropriately.
        """

        try:
            properties = context.system_properties
            model_dir = properties.get("model_dir")
            self.validate_and_initialize(model_dir=model_dir, context=self._context)

            input_data = data[0].get("body")

            request_processor = context.request_processor[0]

            request_property = request_processor.get_request_properties()
            content_type = utils.retrieve_content_type_header(request_property)
            accept = request_property.get("Accept") or request_property.get("accept")

            if not accept or accept == content_types.ANY:
                accept = self._environment.default_accept

            if content_type in content_types.UTF8_TYPES:
                input_data = input_data.decode("utf-8")

            result = self._run_handle_function(self._transform_fn, *(self._model, input_data, content_type, accept))

            response = result
            response_content_type = accept

            if isinstance(result, tuple):
                # handles tuple for backwards compatibility
                response = result[0]
                response_content_type = result[1]

            context.set_response_content_type(0, response_content_type)
            return [response]
        except Exception as e:  # pylint: disable=broad-except
            trace = traceback.format_exc()
            if isinstance(e, BaseInferenceToolkitError):
                return super().handle_error(context, e, trace)
            else:
                return super().handle_error(
                    context,
                    GenericInferenceToolkitError(http_client.INTERNAL_SERVER_ERROR, str(e)),
                    trace,
                )
    
    def validate_and_initialize(self, model_dir=environment.model_dir, context=None):
        """Validates the user module against the SageMaker inference contract.
        Load the model as defined by the ``model_fn`` to prepare handling predictions.
        """
        if not self._initialized:
            self._context = context
            self._environment = environment.Environment()
            self._validate_user_module_and_set_functions()

            if self._pre_model_fn is not None:
                self._run_handle_function(self._pre_model_fn, *(model_dir, ))

            self._model = self._run_handle_function(self._model_fn, *(model_dir, ))

            if self._model_warmup_fn is not None:
                self._run_handle_function(self._model_warmup_fn, *(model_dir, self._model))

            self._initialized = True

    def _default_transform_fn(self, model, input_data, content_type, accept):
        """Make predictions against the model and return a serialized response.
        This serves as the default implementation of transform_fn, used when the
        user has not provided an implementation.
        Args:
            model (obj): model loaded by model_fn.
            input_data (obj): the request data.
            content_type (str): the request content type.
            accept (str): accept header expected by the client.
        Returns:
            obj: the serialized prediction result or a tuple of the form
                (response_data, content_type)
        """
        data = self._run_handle_function(self._input_fn, *(input_data, content_type))
        prediction = self._run_handle_function(self._predict_fn, *(data, model))
        result = self._run_handle_function(self._output_fn, *(prediction, accept))

        return result
    
    def _run_handle_function(self, func, *argv):
        """Wrapper to call the handle function which covers 2 cases:
        1. context passed to the handle function
        2. context not passed to the handle function
        """
        try:
            argv_context = argv + (self._context, )
            result = func(*argv_context)
        except TypeError:
            result = func(*argv)
        
        return result
