import json
import os
import re
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from a2a.utils import new_agent_text_message
from openai import OpenAI


class CodeGenerator:

    def __init__(self, api_key_env: str = "NEBIUS_API_KEY"):
        self.api_key = os.getenv(api_key_env)

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

    def run_openai_query(self, prompt: str) -> str:
        client = OpenAI(base_url="https://api.tokenfactory.nebius.com/v1/", api_key=self.api_key)
        response = client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct",
            messages=[
                {"role": "system", "content": "You are a code generation assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        generated_code = response.choices[0].message.content
        return self.clean_generated_code(generated_code)

    async def generate_code_for_query(self, role: str, scenario: str, openapis: list[dict], query_text: str,
                                      updater: TaskUpdater, dictionary: dict) -> str:
        prompt = f"""
        You are a code generation agent. Generate Python code for the following query using the OpenAPI specs.
        Query: {query_text}
        OpenAPI Specifications: {json.dumps(openapis)}
        """
        generated_code = self.run_openai_query(prompt)
        dictionary[role][scenario].append(generated_code)
        await updater.update_status(TaskState.working, new_agent_text_message(f"[{role}][{scenario}]: {generated_code}"))
        return generated_code

    async def confirm_endpoints(
            self, role: str, scenario: str, expected_endpoints: list[str],
            updater: TaskUpdater
    ) -> str:

        endpoints_json = json.dumps(expected_endpoints)
        prompt = f"""
        You generated code for a query. Please confirm:
        Are these the normalized endpoints you intended to cover? {endpoints_json}
        Respond with 'Yes' or 'No' and explain.
        """

        confirmation = self.run_openai_query(prompt)


        return confirmation
