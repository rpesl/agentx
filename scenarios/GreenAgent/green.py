import uvicorn, asyncio, logging, argparse, contextlib, subprocess, socket, sys
from dotenv import load_dotenv
from pydantic import HttpUrl
from endpoint_evaluator import normalize_endpoint, normalize_expected_endpoint, match_retrieved_to_expected, compute_f1, \
    compute_pass_at_k
from query_loader import SOCBenchQueryLoader, RestBenchQueryLoader
from scenarios import ScenarioRunner
from socbenchsc.src.socbenchsc.analysis import Analysis
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.tasks import TaskUpdater
from a2a.types import (TaskState, Part, TextPart)
from a2a.utils import (new_agent_text_message)
from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest, EvalResult
from agentbeats.tool_provider import ToolProvider
from models import CodeEval, judge_agent_card, CodeScore

logging.basicConfig(level=logging.INFO)
load_dotenv()
BENCHMARK_ROOT = "scenarios/GreenAgent/benchmark"
RESTBENCH_ROOT = "scenarios/GreenAgent/benchmark/restbench/data"
SCENARIOS = ["easy", "medium", "hard", "rag_easy", "rag_medium", "rag_hard", "restbench"]

"""
This agent orchestrates the code generation and evaluation process for the SOCBench and RestBench benchmarks.
It runs multiple rounds of code generation for each participant agent, evaluates the generated code against expected API endpoints,
and computes recall, precision, F1, and pass@k scores to determine a winner.
"""


class CodeJudge(GreenAgent):
    def __init__(self):
        self._required_config_keys = ["num_rounds"]
        self._tool_provider = ToolProvider()
        self.expected_endpoints = []
        self.restbench_expected_endpoints = []
        self.query_loader = SOCBenchQueryLoader(BENCHMARK_ROOT)
        self.restbench_loader = RestBenchQueryLoader(RESTBENCH_ROOT)
        self.scenario_runner = ScenarioRunner(self._tool_provider)

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """
        Validate the incoming evaluation request to ensure it has all required config keys and participants.
        :param request: EvalRequest containing participants and config for the evaluation
        :return: Tuple of (is_valid: bool, message: str). If is_valid is False, message contains the reason for invalidation.
        """
        missing_config_keys = set(self._required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        participants = request.participants
        if not participants:
            return False, "No participants provided"

        for key in ["num_rounds", "pass_at_k_n", "pass_at_k_k"]:
            try:
                int(request.config.get(key, 1))
            except Exception as e:
                return False, f"Can't parse {key}: {e}"

        n = int(request.config.get("pass_at_k_n", 1))
        k = int(request.config.get("pass_at_k_k", 1))
        if k > n:
            return False, f"pass_at_k_k ({k}) cannot be greater than pass_at_k_n ({n})"

        return True, "ok"

    async def run_eval(self, request: EvalRequest, updater: TaskUpdater) -> None:
        """
        Main evaluation logic: orchestrate code generation for each participant and scenario, then judge the generated code and compute scores.
        :param request: EvalRequest containing participants and config for the evaluation
        :param updater: TaskUpdater to update the task status and add artifacts during the evaluation process
        :return: None. The final results are added as an artifact to the updater.
        """
        try:
            n = int(request.config.get("pass_at_k_n", 1))
            k = int(request.config.get("pass_at_k_k", 1))

            code = await self.orchestrate_code_creation(
                request.participants,
                request.config["num_rounds"],
                n,
                updater
            )
            code_eval: CodeEval = await self.judge_code(code, updater, n=n, k=k)
            result = EvalResult(winner=code_eval.winner, detail=code_eval.model_dump())
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=result.model_dump_json()))],
                name="Result",
            )
        finally:
            self._tool_provider.reset()

    async def orchestrate_code_creation(
            self,
            participants: dict[str, HttpUrl],
            num_rounds: int,
            n: int,
            updater: TaskUpdater,
    ) -> dict[str, dict[str, list[list[str]]]]:
        """
        Orchestrate the code generation process for each participant agent across multiple rounds and scenarios.
        :param participants: Dictionary mapping participant names to their agent URLs
        :param num_rounds: Number of rounds to run for each scenario
        :param n: Number of samples to generate for each scenario and round (used for pass@k evaluation)
        :param updater: TaskUpdater to update the task status and add artifacts during the code generation process
        :return: Nested dictionary containing generated code for each participant, scenario, round, and sample.
        """

        results = {
            role: {scenario: [] for scenario in SCENARIOS}
            for role in participants.keys()
        }

        self.expected_endpoints = []
        self.restbench_expected_endpoints = []

        for round_id in range(int(num_rounds)):

            domain_path = self.query_loader.get_next_domain()
            query_text, endpoints, instance_id = self.query_loader.load_query(domain_path)
            normalized_endpoints = [normalize_expected_endpoint(ep) for ep in endpoints]
            self.expected_endpoints.append(normalized_endpoints)

            restbench_query_text, restbench_endpoints = self.restbench_loader.load_query()
            normalized_restbench_endpoints = [normalize_expected_endpoint(ep) for ep in restbench_endpoints]
            self.restbench_expected_endpoints.append(normalized_restbench_endpoints)

            for role, agent_url in participants.items():
                url_str = str(agent_url)

                easy_samples = []
                medium_samples = []
                hard_samples = []
                rag_easy_samples = []
                rag_medium_samples = []
                rag_hard_samples = []
                restbench_samples = []

                for sample_idx in range(n):
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            f"[Round {round_id + 1}/{num_rounds}][Sample {sample_idx + 1}/{n}][SOCBench-D]\n"
                            f"Query: {query_text}\n"
                            f"Expected Endpoints: {endpoints}\n"
                        )
                    )

                    easy_samples.append(await self.scenario_runner.run_easy(
                        role=role,
                        expected_endpoints=normalized_endpoints,
                        query_text=query_text,
                        agent_url=url_str,
                        updater=updater,
                        instance_id=instance_id
                    ))

                    medium_samples.append(await self.scenario_runner.run_medium(
                        role=role,
                        query_text=query_text,
                        agent_url=url_str,
                        updater=updater,
                        instance_id=instance_id
                    ))

                    hard_samples.append(await self.scenario_runner.run_hard(
                        role=role,
                        query_text=query_text,
                        agent_url=url_str,
                        updater=updater,
                        instance_id=instance_id
                    ))

                    rag_easy_samples.append(await self.scenario_runner.run_rag_easy(
                        role=role,
                        expected_endpoints=normalized_endpoints,
                        query_text=query_text,
                        agent_url=url_str,
                        updater=updater,
                        instance_id=instance_id
                    ))

                    rag_medium_samples.append(await self.scenario_runner.run_rag_medium(
                        role=role,
                        query_text=query_text,
                        agent_url=url_str,
                        updater=updater,
                        instance_id=instance_id
                    ))

                    rag_hard_samples.append(await self.scenario_runner.run_rag_hard(
                        role=role,
                        query_text=query_text,
                        agent_url=url_str,
                        updater=updater,
                        instance_id=instance_id
                    ))

                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            f"[Round {round_id + 1}/{num_rounds}][Sample {sample_idx + 1}/{n}][RestBench]\n"
                            f"Query: {restbench_query_text}\n"
                            f"Expected Endpoints: {restbench_endpoints}\n"
                        )
                    )

                    restbench_samples.append(await self.scenario_runner.run_restbench(
                        role=role,
                        expected_endpoints=normalized_restbench_endpoints,
                        query_text=restbench_query_text,
                        agent_url=url_str,
                        updater=updater
                    ))

                results[role]["easy"].append(easy_samples)
                results[role]["medium"].append(medium_samples)
                results[role]["hard"].append(hard_samples)
                results[role]["rag_easy"].append(rag_easy_samples)
                results[role]["rag_medium"].append(rag_medium_samples)
                results[role]["rag_hard"].append(rag_hard_samples)
                results[role]["restbench"].append(restbench_samples)

        return results

    async def judge_code(
            self,
            code_dict: dict[str, dict[str, list[list[str]]]],
            updater: TaskUpdater,
            n: int = 1,
            k: int = 1
    ) -> CodeEval:
        """
        Judge the generated code by extracting retrieved endpoints, and comparing them against expected endpoints to compute recall, precision, F1, and pass@k scores.
        :param code_dict: Nested dictionary containing generated code for each participant, scenario, round, and sample.
        :param updater: TaskUpdater to update the task status and add artifacts during the judging process
        :param n: Number of samples generated for each scenario and round
        :param k: Value of k for pass@k evaluation
        :return: CodeEval containing the final scores and winner information
        """

        scenario_results = {scenario: {} for scenario in SCENARIOS}
        num_rounds = len(self.expected_endpoints)

        for scenario in SCENARIOS:
            intermediate_results = {
                agent: {
                    "recall": [],
                    "precision": [],
                    "f1": [],
                    "num_correct_per_round": []
                }
                for agent in code_dict
            }

            for round_idx in range(num_rounds):
                if scenario != "restbench":
                    expected = self.expected_endpoints[round_idx]
                else:
                    expected = self.restbench_expected_endpoints[round_idx]

                for agent in code_dict:
                    samples = code_dict[agent][scenario][round_idx]

                    round_recall = []
                    round_precision = []
                    round_f1 = []
                    num_correct = 0

                    for sample_idx, sample_code in enumerate(samples):
                        try:
                            analysis = Analysis(sample_code)
                            retrieved = analysis.perform_analysis()
                        except (SyntaxError, ValueError, NotImplementedError):
                            retrieved = set()

                        normalized_retrieved = {normalize_endpoint(ep) for ep in retrieved}
                        matched = match_retrieved_to_expected(normalized_retrieved, expected)

                        recall = round(len(matched) / len(expected), 2) if expected else 0.0
                        precision = round(len(matched) / len(normalized_retrieved), 2) if normalized_retrieved else 0.0
                        f1 = compute_f1(precision, recall)

                        round_recall.append(recall)
                        round_precision.append(precision)
                        round_f1.append(f1)

                        if len(matched) == len(expected) and len(expected) > 0:
                            num_correct += 1

                        await updater.update_status(
                            TaskState.working,
                            new_agent_text_message(
                                f"Agent {agent} - Round {round_idx + 1} - Sample {sample_idx + 1}/{n} - Scenario {scenario}:\n"
                                f"Retrieved endpoints: {retrieved}\n"
                                f"Matched endpoints: {matched}\n"
                                f"Recall: {recall}, Precision: {precision}, F1: {f1}"
                            )
                        )

                    intermediate_results[agent]["recall"].append(
                        sum(round_recall) / len(round_recall)
                    )
                    intermediate_results[agent]["precision"].append(
                        sum(round_precision) / len(round_precision)
                    )
                    intermediate_results[agent]["f1"].append(
                        sum(round_f1) / len(round_f1)
                    )
                    intermediate_results[agent]["num_correct_per_round"].append(num_correct)

            for agent, metrics in intermediate_results.items():
                pass_at_k_scores = [
                    compute_pass_at_k(num_correct=c, num_total=n, k=k)
                    for c in metrics["num_correct_per_round"]
                ]
                avg_pass_at_k = round(sum(pass_at_k_scores) / len(pass_at_k_scores), 2) if pass_at_k_scores else 0.0

                scenario_results[scenario][agent] = {
                    "recall": min(sum(metrics["recall"]) / len(metrics["recall"]), 1.0),
                    "precision": min(sum(metrics["precision"]) / len(metrics["precision"]), 1.0),
                    "f1": min(sum(metrics["f1"]) / len(metrics["f1"]), 1.0),
                    "pass_at_k": avg_pass_at_k,
                }

        participants_scores = {}
        num_scenarios = len(SCENARIOS)

        for agent in code_dict:
            avg_recall = round(sum(scenario_results[s][agent]["recall"] for s in SCENARIOS) / num_scenarios, 2)
            avg_precision = round(sum(scenario_results[s][agent]["precision"] for s in SCENARIOS) / num_scenarios, 2)
            avg_f1 = round(sum(scenario_results[s][agent]["f1"] for s in SCENARIOS) / num_scenarios, 2)
            avg_pass = round(sum(scenario_results[s][agent]["pass_at_k"] for s in SCENARIOS) / num_scenarios, 2)

            participants_scores[agent] = CodeScore(
                recall=avg_recall,
                precision=avg_precision,
                f1=avg_f1,
                pass_at_1=avg_pass
            )
        winner = max(participants_scores, key=lambda a: participants_scores[a].recall)

        return CodeEval(
            participants=participants_scores,
            winner=f"Winner: {winner} (combined Recall = {participants_scores[winner].recall})"
        )


def is_port_open(port):
    """Check if a given port is open on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


async def main():
    parser = argparse.ArgumentParser(description="Run the A2A code generation agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument("--cloudflare-quick-tunnel", action="store_true",
                        help="Use a Cloudflare quick tunnel. Requires cloudflared. This will override --card-url")
    args = parser.parse_args()

    logging.info("Starting MCP Server on port 8000...")
    mcp_process = subprocess.Popen(
        [sys.executable, "scenarios/GreenAgent/mcp_server.py"],
        stdout=None,
        stderr=None
    )

    retries = 0
    while not is_port_open(8000) and retries < 10:
        logging.info("Waiting for MCP Server to bind to port 8000...")
        await asyncio.sleep(1)
        retries += 1

    if not is_port_open(8000):
        logging.error("MCP Server failed to start in time!")
        mcp_process.terminate()
        return
    logging.info("MCP Server is up and running on port 8000.")

    try:
        if args.cloudflare_quick_tunnel:
            from agentbeats.cloudflare import quick_tunnel
            agent_url_cm = quick_tunnel(f"http://{args.host}:{args.port}")
        else:
            agent_url_cm = contextlib.nullcontext(args.card_url or f"http://{args.host}:{args.port}/")

        async with agent_url_cm as agent_url:
            agent = CodeJudge()
            executor = GreenExecutor(agent)
            agent_card = judge_agent_card("CodeJudge", str(agent_url))
            request_handler = DefaultRequestHandler(
                agent_executor=executor,
                task_store=InMemoryTaskStore(),
            )

            server = A2AStarletteApplication(
                agent_card=agent_card,
                http_handler=request_handler,
            )

            uvicorn_config = uvicorn.Config(server.build(), host=args.host, port=args.port)
            uvicorn_server = uvicorn.Server(uvicorn_config)
            await uvicorn_server.serve()
    finally:
        logging.info("Shutting down MCP Server...")
        mcp_process.terminate()


if __name__ == '__main__':
    asyncio.run(main())
