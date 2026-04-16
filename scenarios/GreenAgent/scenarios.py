import json
from dataclasses import dataclass
from re import Pattern
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
    expected_endpoints: Optional[list[Pattern]] = None


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

    @staticmethod
    def confirmation_is_positive(text: str) -> bool:
        text = text.lower().strip()
        return text.startswith("yes") or text.startswith("**yes**") or text.startswith("'yes'")

    async def request_code(
            self,
            agent_url: str,
            task_description: str,
            context: dict | None = None
    ) -> str:
        message = task_description
        if context:
            message = f"{task_description}\n\n[Context: {json.dumps(context)}]"

        return await self._tool_provider.talk_to_agent(
            message=message,
            url=agent_url,
            new_conversation=True
        )

    async def request_confirmation(
            self,
            agent_url: str,
            query_text: str,
            extracted: set[str],
            context: dict | None = None
    ) -> str:
        task_description = f"""
You generated code for this task:
{query_text}

Static analysis extracted these API endpoints from your code:
{list(extracted)}

Question: Are these the correct endpoints you intended to use?
Respond with 'Yes' or 'No' followed by a brief explanation.
"""
        if context:
            task_description += f"\n\n[Context: {json.dumps(context)}]"
        return await self._tool_provider.talk_to_agent(
            message=task_description,
            url=agent_url,
            new_conversation=False
        )

    @staticmethod
    def _easy_prompt(*, expected_endpoints: list[str], query_text: str, **_) -> str:
        return f"""
Generate Python code to accomplish the following task:

{query_text}

Your code should use these API endpoints:
{chr(10).join(f"  - {ep}" for ep in expected_endpoints)}

Requirements:
- Use the 'requests' library for HTTP calls
- Generate functional, executable Python code
- Focus on the endpoints listed above

Note: You may use any available tools or resources to discover API specifications.
""".strip()

    @staticmethod
    def _task_prompt(*, query_text: str, **_) -> str:
        return f"""
Generate Python code to accomplish the following task:

{query_text}

Requirements:
- Use the 'requests' library for HTTP calls
- Generate functional, executable Python code
- Discover and use relevant APIs based on the task description

Note: You may use any available tools or resources to discover API specifications.
""".strip()

    @staticmethod
    def _rag_prompt(*, query_text: str, expected_endpoints: list[str] | None = None, **_) -> str:
        base = f"""
Generate Python code to accomplish the following task:

{query_text}

Requirements:
- Use the 'requests' library for HTTP calls
- Generate functional, executable Python code

Hint: This task may benefit from semantic search to find relevant API specifications.
"""

        if expected_endpoints:
            base += f"""
Your code should use these API endpoints:
{chr(10).join(f"  - {ep}" for ep in expected_endpoints)}
"""

        return base.strip()

    @staticmethod
    def _restbench_prompt(*, expected_endpoints: list[str], query_text: str, **_) -> str:
        return f"""
Generate Python code to accomplish the following task:

{query_text}

Context: This task uses public APIs (Spotify or TMDB).

Your code should use these API endpoints:
{chr(10).join(f"  - {ep}" for ep in expected_endpoints)}

Requirements:
- Use the 'requests' library for HTTP calls
- Generate functional, executable Python code
""".strip()

    async def run_core(
            self,
            role: str,
            agent_url: str,
            updater: TaskUpdater,
            config: ScenarioConfig,
            query_text: str,
            instance_id: str | int,
            mode: str = "code"
    ) -> str:

        best_code = ""
        best_endpoint_count = -1

        await self.log(
            updater,
            role,
            config.level,
            f"Starting {mode.upper()} scenario ({config.max_attempts} attempts)"
        )

        for attempt in range(1, config.max_attempts + 1):
            await self.log(
                updater,
                role,
                config.level,
                f"Attempt {attempt}/{config.max_attempts}"
            )

            task_description = config.prompt_builder(
                query_text=query_text,
                expected_endpoints=config.expected_endpoints
            )

            context = {
                "benchmark": "socbench",
                "instance_id": instance_id,
                "scenario": config.level,
                "mode": mode,
                "attempt": attempt,
                "max_attempts": config.max_attempts
            }

            code = await self.request_code(
                agent_url=agent_url,
                task_description=task_description,
                context=context
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

            context_conformation = {
                "mode": "confirm"
            }
            confirmation = await self.request_confirmation(
                agent_url,
                query_text,
                endpoints,
                context=context_conformation
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


    async def run_easy(
            self,
            role: str,
            expected_endpoints: list[Pattern],
            query_text: str,
            agent_url: str,
            updater: TaskUpdater,
            instance_id: int
    ) -> str:
        config = ScenarioConfig(
            level="easy",
            max_attempts=3,
            requires_confirmation=True,
            expected_endpoints=expected_endpoints,
            prompt_builder=self._easy_prompt
        )
        return await self.run_core(role, agent_url, updater, config, query_text, instance_id, mode="code")

    async def run_medium(
            self,
            role: str,
            query_text: str,
            agent_url: str,
            updater: TaskUpdater,
            instance_id: int
    ) -> str:
        config = ScenarioConfig(
            level="medium",
            max_attempts=3,
            requires_confirmation=True,
            prompt_builder=self._task_prompt
        )
        return await self.run_core(role, agent_url, updater, config, query_text, instance_id, mode="code")

    async def run_hard(
            self,
            role: str,
            query_text: str,
            agent_url: str,
            updater: TaskUpdater,
            instance_id: int
    ) -> str:
        config = ScenarioConfig(
            level="hard",
            max_attempts=1,
            requires_confirmation=False,
            prompt_builder=self._task_prompt
        )
        return await self.run_core(role, agent_url, updater, config, query_text, instance_id, mode="code")

    async def run_rag_easy(
            self,
            role: str,
            expected_endpoints: list[Pattern],
            query_text: str,
            agent_url: str,
            updater: TaskUpdater,
            instance_id: int
    ) -> str:
        config = ScenarioConfig(
            level="rag_easy",
            max_attempts=3,
            requires_confirmation=True,
            expected_endpoints=expected_endpoints,
            prompt_builder=self._rag_prompt
        )
        return await self.run_core(role, agent_url, updater, config, query_text, instance_id, mode="rag")

    async def run_rag_medium(
            self,
            role: str,
            query_text: str,
            agent_url: str,
            updater: TaskUpdater,
            instance_id: int
    ) -> str:
        config = ScenarioConfig(
            level="rag_medium",
            max_attempts=3,
            requires_confirmation=True,
            prompt_builder=self._rag_prompt
        )
        return await self.run_core(role, agent_url, updater, config, query_text, instance_id, mode="rag")

    async def run_rag_hard(
            self,
            role: str,
            query_text: str,
            agent_url: str,
            updater: TaskUpdater,
            instance_id: int
    ) -> str:
        config = ScenarioConfig(
            level="rag_hard",
            max_attempts=1,
            requires_confirmation=False,
            prompt_builder=self._rag_prompt
        )
        return await self.run_core(role, agent_url, updater, config, query_text, instance_id, mode="rag")

    async def run_restbench(
            self,
            role: str,
            expected_endpoints: list[Pattern],
            query_text: str,
            agent_url: str,
            updater: TaskUpdater
    ) -> str:
        config = ScenarioConfig(
            level="restbench",
            max_attempts=3,
            requires_confirmation=True,
            expected_endpoints=expected_endpoints,
            prompt_builder=self._restbench_prompt
        )
        return await self.run_core(role, agent_url, updater, config, query_text, instance_id="restbench", mode="rag")