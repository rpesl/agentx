import argparse
import contextlib
import uvicorn
import asyncio
import logging
from dotenv import load_dotenv
from pydantic import HttpUrl
from endpoint_evaluator import normalize_endpoint, normalize_expected_endpoint, match_retrieved_to_expected, compute_f1
from openapi_loader import OpenAPILoader
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

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluation_agent")

BENCHMARK_ROOT = "scenarios/socbench/benchmark"
SCENARIOS = ["easy", "medium", "hard"]


class CodeJudge(GreenAgent):
    def __init__(self, num_agents=2):
        self._required_roles = [f"PurpleAgent_{i}" for i in range(num_agents)]
        self._required_config_keys = ["num_rounds"]
        self._tool_provider = ToolProvider()
        self.expected_endpoints = []
        self.openapi_loader = OpenAPILoader(BENCHMARK_ROOT)
        self.scenario_runner = ScenarioRunner(self._tool_provider)

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self._required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"
        missing_config_keys = set(self._required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"
        try:
            int(request.config["num_rounds"])
        except Exception as e:
            return False, f"Can't parse num_rounds: {e}"
        return True, "ok"

    async def run_eval(self, request: EvalRequest, updater: TaskUpdater) -> None:
        logger.info(f"Starting code creation orchestration: {request}")

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

        for round_id in range(num_rounds):
            domain_path = self.openapi_loader.get_next_domain_path()
            openapis = self.openapi_loader.load_openapi_specs(domain_path)
            query_text, endpoints = self.openapi_loader.load_query(domain_path)
            self.expected_endpoints.append(endpoints)

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"[Round {round_id + 1}] Selected query: {query_text}")
            )
            for role, agent_url in participants.items():
                url_str = str(agent_url)

                easy_code = await self.scenario_runner.run_easy(
                    role=role,
                    openapis=openapis,
                    expected_endpoints=endpoints,
                    query_text=query_text,
                    agent_url=url_str,
                    updater=updater
                )
                results[role]["easy"].append(easy_code)

                medium_code = await self.scenario_runner.run_medium(
                    role=role,
                    openapis=openapis,
                    query_text=query_text,
                    agent_url=url_str,
                    updater=updater
                )
                results[role]["medium"].append(medium_code)

                hard_code = await self.scenario_runner.run_hard(
                    role=role,
                    openapis=openapis,
                    query_text=query_text,
                    agent_url=url_str,
                    updater=updater
                )
                results[role]["hard"].append(hard_code)

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
                expected = self.expected_endpoints[round_idx]

                for agent in code_dict:
                    code = code_dict[agent][scenario][round_idx]
                    try:
                        analysis = Analysis(code)
                        retrieved = analysis.perform_analysis()
                    except (SyntaxError, ValueError):
                        retrieved = set()

                    normalized_retrieved = {normalize_endpoint(ep) for ep in retrieved}
                    normalized_expected = [normalize_expected_endpoint(ep) for ep in expected]

                    matched = match_retrieved_to_expected(normalized_retrieved, normalized_expected)
                    recall = round(len(matched) / len(normalized_expected), 2) if normalized_expected else 0.0
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
                    "recall": sum(metrics["recall"]) / len(metrics["recall"]),
                    "precision": sum(metrics["precision"]) / len(metrics["precision"]),
                    "f1": sum(metrics["f1"]) / len(metrics["f1"]),
                }

        final_recall_scores = {
            agent: round(
                sum(scenario_results[sc][agent]["recall"] for sc in SCENARIOS) / len(SCENARIOS),
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
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
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
