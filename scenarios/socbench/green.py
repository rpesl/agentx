import uvicorn, asyncio, logging, argparse, contextlib
from dotenv import load_dotenv
from pydantic import HttpUrl
from endpoint_evaluator import normalize_endpoint, normalize_expected_endpoint, match_retrieved_to_expected, compute_f1
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
BENCHMARK_ROOT = "scenarios/socbench/benchmark"
RESTBENCH_ROOT = "scenarios/socbench/benchmark/restbench/data"
SCENARIOS = ["easy", "medium", "hard", "rag_easy", "rag_medium", "rag_hard", "restbench"]


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
        missing_config_keys = set(self._required_config_keys) - set(request.config.keys())

        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        participants = request.participants
        if not participants:
            return False, "No participants provided"
        try:
            int(request.config["num_rounds"])
        except Exception as e:
            return False, f"Can't parse num_rounds: {e}"
        return True, "ok"

    async def run_eval(self, request: EvalRequest, updater: TaskUpdater) -> None:
        try:
            code = await self.orchestrate_code_creation(
                request.participants,
                request.config["num_rounds"],
                updater
            )
            code_eval: CodeEval = await self.judge_code(code, updater)
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
            updater: TaskUpdater,
    ) -> dict[str, dict[str, list[str]]]:

        results = {
            role: {scenario: [] for scenario in SCENARIOS}
            for role in participants.keys()
        }

        self.expected_endpoints = []
        self.restbench_expected_endpoints = []

        for round_id in range(num_rounds):

            domain_path = self.query_loader.get_next_domain()
            query_text, endpoints, instance_id = self.query_loader.load_query(domain_path)
            normalized_endpoints = [normalize_expected_endpoint(ep) for ep in endpoints]
            self.expected_endpoints.append(normalized_endpoints)

            restbench_query_text, restbench_endpoints = self.restbench_loader.load_query()
            normalized_restbench_endpoints = [normalize_expected_endpoint(ep) for ep in restbench_endpoints]
            self.restbench_expected_endpoints.append(normalized_restbench_endpoints)

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"[Round {round_id + 1}][SOCBench-D] \n"
                    f"Query: {query_text}\n"
                    f"Expected Endpoints: {endpoints}\n"
                )
            )

            for role, agent_url in participants.items():

                url_str = str(agent_url)

                easy_code = await self.scenario_runner.run_easy(
                    role=role,
                    expected_endpoints=normalized_endpoints,
                    query_text=query_text,
                    agent_url=url_str,
                    updater=updater,
                    instance_id=instance_id
                )
                results[role]["easy"].append(easy_code)

                medium_code = await self.scenario_runner.run_medium(
                    role=role,
                    query_text=query_text,
                    agent_url=url_str,
                    updater=updater,
                    instance_id=instance_id
                )
                results[role]["medium"].append(medium_code)

                hard_code = await self.scenario_runner.run_hard(
                    role=role,
                    query_text=query_text,
                    agent_url=url_str,
                    updater=updater,
                    instance_id=instance_id
                )
                results[role]["hard"].append(hard_code)

                rag_easy_code = await self.scenario_runner.run_rag_easy(
                    role=role,
                    expected_endpoints=normalized_endpoints,
                    query_text=query_text,
                    agent_url=url_str,
                    updater=updater,
                    instance_id=instance_id
                )
                results[role]["rag_easy"].append(rag_easy_code)

                rag_medium_code = await self.scenario_runner.run_rag_medium(
                    role=role,
                    query_text=query_text,
                    agent_url=url_str,
                    updater=updater,
                    instance_id=instance_id
                )
                results[role]["rag_medium"].append(rag_medium_code)

                rag_hard_code = await self.scenario_runner.run_rag_hard(
                    role=role,
                    query_text=query_text,
                    agent_url=url_str,
                    updater=updater,
                    instance_id=instance_id
                )
                results[role]["rag_hard"].append(rag_hard_code)

                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"[Round {round_id + 1}][RestBench]\n"
                        f"Query: {restbench_query_text}\n"
                        f"Expected Endpoints: {restbench_endpoints}\n"
                    )
                )

                restbench_code = await self.scenario_runner.run_restbench(
                    role=role,
                    expected_endpoints=normalized_restbench_endpoints,
                    query_text=restbench_query_text,
                    agent_url=url_str,
                    updater=updater
                )
                results[role]["restbench"].append(restbench_code)

        return results

    async def judge_code(self, code_dict: dict[str, dict[str, list[str]]], updater: TaskUpdater) -> CodeEval:

        scenario_results = {scenario: {} for scenario in SCENARIOS}
        num_rounds = len(self.expected_endpoints)

        for scenario in SCENARIOS:
            intermediate_results = {
                agent: {"recall": [], "precision": [], "f1": []}
                for agent in code_dict
            }

            for round_idx in range(num_rounds):
                if scenario != "restbench":
                    expected = self.expected_endpoints[round_idx]
                else:
                    expected = self.restbench_expected_endpoints[round_idx]

                for agent in code_dict:
                    code = code_dict[agent][scenario][round_idx]

                    try:
                        analysis = Analysis(code)
                        retrieved = analysis.perform_analysis()
                    except (SyntaxError, ValueError, NotImplementedError):
                        retrieved = set()

                    normalized_retrieved = {normalize_endpoint(ep) for ep in retrieved}

                    matched = match_retrieved_to_expected(normalized_retrieved, expected)
                    recall = round(len(matched) / len(expected), 2) if expected else 0.0
                    precision = round(len(matched) / len(normalized_retrieved), 2) if normalized_retrieved else 0.0
                    f1 = compute_f1(precision, recall)

                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            f"Agent {agent} - Round {round_idx + 1} - Scenario {scenario}:\n"
                            f"Retrieved endpoints: {retrieved}\n"
                            f"Matched endpoints: {matched}\n"
                            f"Recall: {recall}, Precision: {precision}, F1: {f1}"
                        )
                    )

                    intermediate_results[agent]["recall"].append(recall)
                    intermediate_results[agent]["precision"].append(precision)
                    intermediate_results[agent]["f1"].append(f1)

            for agent, metrics in intermediate_results.items():
                scenario_results[scenario][agent] = {
                    "recall": min(sum(metrics["recall"]) / len(metrics["recall"]), 1.0),
                    "precision": min(sum(metrics["precision"]) / len(metrics["precision"]), 1.0),
                    "f1": min(sum(metrics["f1"]) / len(metrics["f1"]), 1.0),
                }

        final_recall_scores = {
            agent: round(
                sum(scenario_results[scenario][agent]["recall"] for scenario in SCENARIOS) / len(SCENARIOS),
                2
            )
            for agent in code_dict
        }

        winner = max(final_recall_scores, key=final_recall_scores.get)

        return CodeEval(
            participants={
                agent: CodeScore(
                    recall=round(
                        sum(scenario_results[scenario][agent]["recall"] for scenario in SCENARIOS) / len(SCENARIOS),
                        2
                    ),
                    precision=round(
                        sum(scenario_results[scenario][agent]["precision"] for scenario in SCENARIOS) / len(SCENARIOS),
                        2
                    ),
                    f1=round(
                        sum(scenario_results[scenario][agent]["f1"] for scenario in SCENARIOS) / len(SCENARIOS),
                        2
                    )
                )
                for agent in code_dict
            },
            winner=f"Winner: {winner} (combined Recall = {final_recall_scores[winner]})"
        )


async def main():
    parser = argparse.ArgumentParser(description="Run the A2A code generation agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument("--cloudflare-quick-tunnel", action="store_true",
                        help="Use a Cloudflare quick tunnel. Requires cloudflared. This will override --card-url")
    args = parser.parse_args()

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


if __name__ == '__main__':
    asyncio.run(main())
