from .openapiparser import OpenApiParser
from llama_index.core.schema import TextNode

class EndpointParser(OpenApiParser):
    def _create_endpoint_node(self, node: TextNode, endpoint: object) -> None:
        pass

    def _filter_endpoint(self, endpoint: object) -> None:
        pass
