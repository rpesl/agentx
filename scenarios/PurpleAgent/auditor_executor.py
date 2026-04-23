import json
import logging
import re

from base_executor import BaseExecutor
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam,
    Function as ToolCallFunction,
)

logger = logging.getLogger("AuditorExecutor")


class AuditorExecutor(BaseExecutor):
    """
    Dedicated auditor agent that:
    1. Extracts all API endpoints from generated code.
    2. Fetches real OpenAPI specs via MCP to verify each endpoint.
    3. Returns a structured JSON audit report with confirmed/wrong/missing endpoints.
    """

    async def run_logic(self, task_description: str, context: dict) -> str:
        """Main entry point for the auditor executor."""
        mode = context.get("mode", "audit")
        instance_id = context.get("instance_id", 1)

        if mode == "confirm":
            return await self._lightweight_confirm(task_description)

        code = context.get("code", "")
        if not code:
            code = self._extract_code_from_message(task_description)

        return await self._mcp_audit(task_description, code, instance_id, context)

    @staticmethod
    def _extract_code_from_message(message: str) -> str:
        match = re.search(r"CODE[:\s]*\n(.*)", message, re.DOTALL)
        if match:
            return match.group(1).strip()
        return message

    async def _lightweight_confirm(self, prompt: str) -> str:
        """A quick check to see if the generated endpoints even remotely match the task, without calling any tools."""
        system_msg: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": (
                "You are a code reviewer. "
                "Respond with 'Yes' if the API endpoints match the task intent, "
                "or 'No' with a brief explanation if they don't."
            ),
        }
        user_msg: ChatCompletionUserMessageParam = {"role": "user", "content": prompt}

        response = self.client.chat.completions.create(
            model="NousResearch/Hermes-4-405B",
            messages=[system_msg, user_msg],
        )
        return self.clean_text(response.choices[0].message.content.strip())

    async def _mcp_audit(
            self,
            task_description: str,
            code: str,
            instance_id: int | str,
            context: dict,
    ) -> str:
        """Performs a detailed audit of the generated code against real OpenAPI specs via MCP tools."""
        use_rag = context.get("mode") == "rag"
        tools = await self._list_mcp_tools(use_rag=use_rag)

        system_msg: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": f"""
You are a High-Recall API Auditor. Your goal is to ensure EVERY endpoint needed for the task is present. Instance ID: {instance_id}.

RECALL STRATEGY:
1. EXPLORE: Use list_available_domains. Don't just look at the code's domains; look at ALL domains that might fit the task description.
2. SEARCH: For each domain, use load_openapi_specs. Search for keywords from the task (e.g., if the task is 'get user energy', search for 'user', 'energy', 'meter', 'reading').
3. VALIDATE: If the code uses 'GET /data', but the spec shows 'GET /v2/data' is the correct one, mark it as 'wrong' AND list the correct one.
4. FILL THE GAPS: If the task requires 3 steps (Auth -> List -> Get) but the code only does 2, list the missing one in "missing".

OUTPUT JSON:
{{
  "confirmed": ["METHOD /path"],
  "wrong": [{{"used": "...", "correct": "...", "reason": "..."}}],
  "missing": ["METHOD /path (Reason why it is needed)"],
  "summary": "..."
}}
""".strip(),
        }

        user_msg: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": (
                f"Task: {task_description}\n\n"
                f"Code to audit:\n```python\n{code}\n```"
            ),
        }

        messages: list[ChatCompletionMessageParam] = [system_msg, user_msg]

        response = self.client.chat.completions.create(
            model="NousResearch/Hermes-4-405B",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        for _ in range(4):
            if not response.choices[0].message.tool_calls:
                break

            assistant_message = response.choices[0].message
            tool_calls_param = [
                ChatCompletionMessageToolCallParam(
                    id=tc.id,
                    type="function",
                    function=ToolCallFunction(
                        name=tc.function.name, arguments=tc.function.arguments
                    ),
                )
                for tc in (assistant_message.tool_calls or [])
            ]
            assistant_msg: ChatCompletionAssistantMessageParam = {
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": tool_calls_param,
            }
            messages.append(assistant_msg)

            for tool_call in assistant_message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                logger.info(f"Auditor → MCP tool: {name}({args})")
                result = await self._call_mcp_tool(name, args)
                tool_msg: ChatCompletionToolMessageParam = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result if isinstance(result, str) else json.dumps(result),
                }
                messages.append(tool_msg)

            response = self.client.chat.completions.create(
                model="NousResearch/Hermes-4-405B",
                messages=messages,
                tools=tools,
            )

        raw = response.choices[0].message.content or "{}"
        logger.info(f"Audit raw report:\n{raw[:500]}")

        clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        try:
            json.loads(clean)
            return clean
        except json.JSONDecodeError:
            return json.dumps({"confirmed": [], "wrong": [], "missing": [], "summary": clean})
