import json
import os
import re

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import TaskState
from a2a.utils import new_task, new_agent_text_message
from openai import OpenAI


class PurpleExecutor(AgentExecutor):
    def __init__(self, api_key_env: str = "NEBIUS_API_KEY"):
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(f"Environment variable {api_key_env} not set")
        self.client = OpenAI(
            base_url="https://api.tokenfactory.nebius.com/v1/",
            api_key=self.api_key
        )

    @staticmethod
    def clean_generated_code(text: str) -> str:
        text = text.encode("ascii", "ignore").decode()
        code_blocks = re.findall(r"```(?:[\w+-]*)?\n(.*?)```", text, re.DOTALL)
        if code_blocks:
            filtered_blocks = [
                block.strip() for block in code_blocks
                if block.strip() and not block.strip().startswith("pip install")
            ]
            if filtered_blocks:
                cleaned = max(filtered_blocks, key=len)
                cleaned = re.sub(r"^#.*\n", "", cleaned)
                return cleaned
            return ""
        return text.strip()

    def generate_code(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": "You are a code generation assistant specialized in creating Python code based on OpenAPI specifications."
                },
                {"role": "user", "content": prompt}
            ]
        )
        generated_code = response.choices[0].message.content
        return self.clean_generated_code(generated_code)

    def generate_text(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a reviewer agent. "
                        "Your task is to confirm or reject whether extracted API endpoints "
                        "match the user's intent. "
                        "Respond with 'Yes' or 'No' followed by a brief explanation."
                    )
                },
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content.strip()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        request_text = context.get_user_input()

        try:
            request_data = json.loads(request_text)
            prompt = request_data.get("prompt", "")
            mode = request_data.get("mode", "code")
        except json.JSONDecodeError as e:
            msg = context.message
            if msg:
                task = new_task(msg)
                await event_queue.enqueue_event(task)
                from a2a.server.tasks import TaskUpdater
                updater = TaskUpdater(event_queue, task.id, task.context_id)
                await updater.failed(
                    new_agent_text_message(f"Invalid JSON request: {e}", context_id=context.context_id)
                )
            return

        msg = context.message
        if msg:
            task = new_task(msg)
            await event_queue.enqueue_event(task)
        else:
            return

        from a2a.server.tasks import TaskUpdater
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            if mode == "confirm":
                result = self.generate_text(prompt)
            else:
                result = self.generate_code(prompt)

            response_message = new_agent_text_message(
                result,
                context_id=context.context_id
            )

            await updater.update_status(
                TaskState.working,
                response_message
            )

            await updater.complete(
                message=new_agent_text_message(
                    result,
                    context_id=context.context_id
                )
            )

        except Exception as e:
            await updater.failed(
                new_agent_text_message(
                    f"Code generation failed: {str(e)}",
                    context_id=context.context_id
                )
            )

    async def cancel(self, request: RequestContext, event_queue: EventQueue):
        return None
