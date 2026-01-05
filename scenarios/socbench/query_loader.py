import json
from pathlib import Path
from typing import List, Tuple

class QueryLoader:
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
        domain_name = self.domains[self.domain_idx % len(self.domains)]
        instance_id = self.instances[self.instance_idx % len(self.instances)]
        self.domain_idx += 1
        if self.domain_idx % len(self.domains) == 0:
            self.instance_idx = (self.instance_idx + 1) % len(self.instances)
        domain_fs = self.benchmark_root / f"socbenchd_{instance_id}" / domain_name
        return domain_fs
