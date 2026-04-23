import json
from pathlib import Path
from typing import List, Tuple

"""
This module provides query loaders for the SOCBench and RestBench benchmarks. 
The SOCBenchQueryLoader class loads queries from the SOCBench dataset, while the RestBenchQueryLoader class loads queries from the RestBench dataset. 
Both classes implement caching to optimize query loading and provide methods to retrieve the next query for evaluation.
"""


class SOCBenchQueryLoader:
    def __init__(self, benchmark_root: str):
        self.benchmark_root = Path(benchmark_root)
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

    def load_query(self, domain_path: Path) -> Tuple[str, List[str], int]:
        """Loads a query from the specified domain path. Caches queries to optimize loading."""
        domain_path_str = str(domain_path.as_posix())
        if domain_path_str not in self.queries_cache:
            query_file = domain_path / "queries.json"
            with open(query_file, "r") as file:
                query_data = json.load(file)
                self.queries_cache[domain_path_str] = query_data["queries"]

        queries = self.queries_cache[domain_path_str]
        query = queries[self.query_idx % len(queries)]
        self.query_idx += 1
        instance_id = int(domain_path.parent.name.split("_")[1])

        return query["query"], query["endpoints"], instance_id

    def get_next_domain(self) -> Path:
        """Returns the path to the next domain for evaluation. Cycles through domains and instances."""
        domain_name = self.domains[self.domain_idx % len(self.domains)]
        instance_id = self.instances[self.instance_idx % len(self.instances)]
        self.domain_idx += 1
        if self.domain_idx % len(self.domains) == 0:
            self.instance_idx = (self.instance_idx + 1) % len(self.instances)
        domain_path = self.benchmark_root / f"socbenchd_{instance_id}" / domain_name

        return domain_path


class RestBenchQueryLoader:
    def __init__(self, restbench_root: str):
        self.restbench_root = Path(restbench_root)
        self.files = [
            self.restbench_root / "datasets" / "spotify.json",
            self.restbench_root / "datasets" / "tmdb.json",
        ]
        self.file_idx = 0
        self.query_idx = 0
        self.cache = {}

    def load_query(self) -> Tuple[str, List[str]]:
        """Loads a query from the RestBench dataset. Caches queries to optimize loading."""
        file_path = self.files[self.file_idx % len(self.files)]
        self.file_idx += 1

        if file_path not in self.cache:
            with open(file_path, "r") as f:
                self.cache[file_path] = json.load(f)

        queries = self.cache[file_path]
        query = queries[self.query_idx % len(queries)]
        self.query_idx += 1

        return query["query"], query["solution"]
