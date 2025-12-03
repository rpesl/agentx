class MetricEvaluator:

    @staticmethod
    def compute_recall(retrieved_endpoints: set[str], expected_endpoints: list[str]) -> float:
        if not expected_endpoints:
            return 0.0
        true_positives = len(retrieved_endpoints.intersection(set(expected_endpoints)))
        recall = true_positives / len(expected_endpoints)
        return float(round(recall, 2))

    @staticmethod
    def compute_precision(retrieved_endpoints: set[str], expected_endpoints: list[str]) -> float:
        if not retrieved_endpoints:
            return 0.0
        true_positives = len(retrieved_endpoints.intersection(set(expected_endpoints)))
        precision = true_positives / len(retrieved_endpoints)
        return float(round(precision, 2))

    @staticmethod
    def compute_f1(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        f1 = 2 * (precision * recall) / (precision + recall)
        return float(round(f1, 2))