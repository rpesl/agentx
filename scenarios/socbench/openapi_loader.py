import json
import os
from typing import List, Tuple


class OpenAPILoader:
    def __init__(self, benchmark_root: str):
        self.benchmark_root = benchmark_root
        self.domain_idx = 0
        self.instance_idx = 0
        self.query_idx = 0
        self.queries_cache = {}
        self.domains = [
            "01-energy", "02-materials", "03-industrials", "04-consumer discretionary",
            "05-consumer staples", "06-health care", "07-financials", "08-information technology",
            "09-communication services", "10-utilities", "11-real estate"
        ]
        self.instances = list(range(1, 6))

    @staticmethod
    def load_openapi_specs(domain_path: str) -> List[dict]:
        openapis = []
        for entry in sorted(os.listdir(domain_path)):
            service_path = os.path.join(domain_path, entry)
            if os.path.isdir(service_path):
                openapi_file = os.path.join(service_path, "openapi.json")
                if os.path.exists(openapi_file):
                    with open(openapi_file, "r") as file:
                        openapis.append(json.load(file))
        return openapis

    def load_query(self, domain_path: str) -> Tuple[str, List[str]]:
        if domain_path not in self.queries_cache:
            query_file = os.path.join(domain_path, "queries.json")
            with open(query_file, "r") as file:
                query_data = json.load(file)
                self.queries_cache[domain_path] = query_data["queries"]

        queries = self.queries_cache[domain_path]
        query = queries[self.query_idx % len(queries)]
        self.query_idx += 1
        return query["query"], query["endpoints"]

    def get_next_domain_path(self) -> str:
        domain_name = self.domains[self.domain_idx % len(self.domains)]
        instance_id = self.instances[self.instance_idx % len(self.instances)]

        self.domain_idx += 1
        if self.domain_idx % len(self.domains) == 0:
            self.instance_idx += 1
            if self.instance_idx >= len(self.instances):
                self.instance_idx = 0

        domain_path = os.path.join(self.benchmark_root, f"socbenchd_{instance_id}", domain_name)
        return domain_path
