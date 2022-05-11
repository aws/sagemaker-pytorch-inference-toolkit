from six.moves import http_client

from sagemaker_inference.transformer import Transformer
from sagemaker_inference import content_types, environment, utils
from sagemaker_inference.default_inference_handler import DefaultInferenceHandler
from sagemaker_inference.errors import BaseInferenceToolkitError, GenericInferenceToolkitError


class PyTorchTransformer(Transformer):
    def transform(self, data, context):
        """
        Take a request with input data, deserialize it, make a prediction, and return a
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
            self.validate_and_initialize(model_dir=model_dir)

            response_list = []

            for i in range(len(data)):
                print(f"Processing Data: {data[i]}")
                input_data = data[i].get("body")

                request_processor = context.request_processor[0]

                request_property = request_processor.get_request_properties()
                content_type = utils.retrieve_content_type_header(request_property)
                accept = request_property.get("Accept") or request_property.get("accept")

                if not accept or accept == content_types.ANY:
                    accept = self._environment.default_accept

                if content_type in content_types.UTF8_TYPES:
                    input_data = input_data.decode("utf-8")

                result = self._transform_fn(self._model, input_data, content_type, accept)

                response = result
                response_content_type = accept

                if isinstance(result, tuple):
                    # handles tuple for backwards compatibility
                    response = result[0]
                    response_content_type = result[1]

                context.set_response_content_type(0, response_content_type)

                response_list.append(response)

            return response_list
        except Exception as e:  # pylint: disable=broad-except
            trace = traceback.format_exc()
            if isinstance(e, BaseInferenceToolkitError):
                return self.handle_error(context, e, trace)
            else:
                return self.handle_error(
                    context,
                    GenericInferenceToolkitError(http_client.INTERNAL_SERVER_ERROR, str(e)),
                    trace,
                )
