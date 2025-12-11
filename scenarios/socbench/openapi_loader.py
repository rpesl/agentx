import json
import os
import random

class OpenAPILoader:

    def __init__(self, benchmark_root: str):
        self.benchmark_root = benchmark_root

    @staticmethod
    def load_openapi_specs(domain_path: str) -> list[dict]:
        openapis = []
        for entry in os.listdir(domain_path):
            service_path = os.path.join(domain_path, entry)
            if os.path.isdir(service_path):
                openapi_file = os.path.join(service_path, "openapi.json")
                if os.path.exists(openapi_file):
                    with open(openapi_file, "r") as file:
                        openapis.append(json.load(file))
        return openapis

    @staticmethod
    def load_query(domain_path: str) -> tuple[str, list[str]]:
        query_file = os.path.join(domain_path, "queries.json")
        with open(query_file, "r") as file:
            query_data = json.load(file)
        query = random.choice(query_data["queries"])
        return query["query"], query["endpoints"]

    def get_random_domain_path(self, domains: list[str], instance_range: tuple[int, int] = (1, 5)) -> str:
        instance_id = random.randint(*instance_range)
        domain = random.choice(domains)
        return os.path.join(self.benchmark_root, f"socbenchd_{instance_id}", domain)
