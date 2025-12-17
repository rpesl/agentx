import json
from dataclasses import dataclass
from typing import Callable, Optional

from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from a2a.utils import new_agent_text_message

from socbenchsc.src.socbenchsc import Analysis
from endpoint_evaluator import normalize_endpoint


@dataclass
class ScenarioConfig:
    level: str
    max_attempts: int
    requires_confirmation: bool
    prompt_builder: Callable[..., str]
    expected_endpoints: Optional[list[str]] = None


class ScenarioRunner:
    def __init__(self, tool_provider):
        self._tool_provider = tool_provider

    @staticmethod
    async def log(
            updater: TaskUpdater,
            role: str,
            scenario: str,
            msg: str
    ) -> None:
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"[{role}][{scenario}] {msg}")
        )

    @staticmethod
    def extract_endpoints(code: str) -> set[str]:
        try:
            analysis = Analysis(code)
            retrieved = analysis.perform_analysis()
        except (SyntaxError, ValueError, NotImplementedError):
            return set()

        return {normalize_endpoint(ep) for ep in retrieved}

    async def request_code(
            self,
            agent_url: str,
            prompt: str,
            new_conversation: bool
    ) -> str:
        return await self._tool_provider.talk_to_agent(
            message=json.dumps({
                "prompt": prompt
            }),
            url=agent_url,
            new_conversation=new_conversation
        )

    async def request_confirmation(
            self,
            agent_url: str,
            query_text: str,
            extracted: set[str],
    ) -> str:
        prompt = (
            f"You generated code for the query: {query_text}"
            f"The static analysis extracted the following endpoints from your code:"
            f"{list(extracted)}"
            f"Are these the correct endpoints you intended to use?"
            f"Please answer with 'Yes' or 'No' and briefly explain."
        )

        return await self._tool_provider.talk_to_agent(
            message=json.dumps({
                "mode": "confirmation",
                "prompt": prompt
            }),
            url=agent_url,
            new_conversation=False
        )

    @staticmethod
    def confirmation_is_positive(text: str) -> bool:
        text = text.lower().strip()
        return text.startswith("yes") or text.startswith("**yes**")

    async def run_core(
            self,
            role: str,
            agent_url: str,
            updater: TaskUpdater,
            config: ScenarioConfig,
            query_text: str,
            openapis: list[dict]
    ) -> str:

        best_code = ""
        best_endpoint_count = -1

        await self.log(
            updater,
            role,
            config.level,
            f"Starting scenario ({config.max_attempts} attempts)"
        )

        for attempt in range(1, config.max_attempts + 1):

            await self.log(
                updater,
                role,
                config.level,
                f"Attempt {attempt}/{config.max_attempts}"
            )

            prompt = config.prompt_builder(
                query_text=query_text,
                openapis=openapis,
                expected_endpoints=config.expected_endpoints
            )

            code = await self.request_code(
                agent_url=agent_url,
                prompt=prompt,
                new_conversation=True
            )

            endpoints = self.extract_endpoints(code)

            await self.log(
                updater,
                role,
                config.level,
                f"Extracted endpoints: {endpoints}"
            )

            if len(endpoints) > best_endpoint_count:
                best_code = code
                best_endpoint_count = len(endpoints)

            if not config.requires_confirmation:
                return code

            confirmation = await self.request_confirmation(
                agent_url,
                query_text,
                endpoints,
            )

            await self.log(
                updater,
                role,
                config.level,
                f"Confirmation: {confirmation}"
            )

            if self.confirmation_is_positive(confirmation):
                await self.log(
                    updater,
                    role,
                    config.level,
                    "Accepted"
                )
                return code
            await self.log(
                updater,
                role,
                config.level,
                "Rejected - retrying"
            )

        await self.log(
            updater,
            role,
            config.level,
            "Max attempts reached - returning best code"
        )

        return best_code

    @staticmethod
    def easy_prompt(
            *,
            expected_endpoints: list[str],
            query_text: str,
            openapis: list[dict],
            **_
    ) -> str:
        return f"""
               Generate Python code for the this query:
               Query: {query_text}
               Use the OpenAPI specs:
               OpenAPI specs: {json.dumps(openapis, indent=2)}
               Expected endpoints to retrieve:
               {expected_endpoints}
               """

    @staticmethod
    def task_prompt(
            *,
            query_text: str,
            openapis: list[dict],
            **_
    ) -> str:
        return f"""
               Generate Python code for this query:
               {query_text}
               Use the OpenAPI specs:
               {json.dumps(openapis, indent=2)}
               """

    async def run_easy(
            self,
            role: str,
            openapis: list[dict],
            expected_endpoints: list[str],
            query_text: str,
            agent_url: str,
            updater: TaskUpdater
    ) -> str:
        config = ScenarioConfig(
            level="easy",
            max_attempts=3,
            requires_confirmation=True,
            expected_endpoints=expected_endpoints,
            prompt_builder=self.easy_prompt
        )
        return await self.run_core(
            role, agent_url, updater, config, query_text, openapis
        )

    async def run_medium(
            self,
            role: str,
            openapis: list[dict],
            query_text: str,
            agent_url: str,
            updater: TaskUpdater
    ) -> str:
        config = ScenarioConfig(
            level="medium",
            max_attempts=3,
            requires_confirmation=True,
            prompt_builder=self.task_prompt
        )
        return await self.run_core(
            role, agent_url, updater, config, query_text, openapis
        )

    async def run_hard(
            self,
            role: str,
            openapis: list[dict],
            query_text: str,
            agent_url: str,
            updater: TaskUpdater
    ) -> str:
        config = ScenarioConfig(
            level="hard",
            max_attempts=1,
            requires_confirmation=False,
            prompt_builder=self.task_prompt
        )
        return await self.run_core(
            role, agent_url, updater, config, query_text, openapis
        )
