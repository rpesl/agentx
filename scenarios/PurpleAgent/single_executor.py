import json
import logging
from base_executor import BaseExecutor
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam,
    Function as ToolCallFunction,
)

logger = logging.getLogger("SingleExecutor")


class SingleExecutor(BaseExecutor):
    """
    A straightforward executor that:
    In "confirm" mode: directly asks the model to confirm if the generated endpoints match the task.
    In "code" mode: generates code using MCP tools with a simple workflow and returns the final code without any iterative validation or refinement.
    """

    async def run_logic(self, task_description: str, context: dict) -> str:
        """Main entry point for the single executor."""
        mode = context.get("mode", "")
        instance_id = context.get("instance_id", 1)
        if mode == "confirm":
            logger.info("Executing confirmation logic...")
            return await self.generate_confirmation(task_description)
        else:
            use_rag = (mode == "rag")
            logger.info(f"Generating code with use_rag={use_rag} for instance {instance_id}")
            return await self.generate_code_with_mcp(task_description, instance_id, use_rag)

    async def generate_confirmation(self, prompt: str) -> str:
        """A simple confirmation step that asks the model to evaluate if the generated endpoints match the user's intent."""
        system_msg: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": (
                "You are a reviewer agent. "
                "Confirm or reject whether extracted API endpoints match the user's intent. "
                "Respond with 'Yes' or 'No' followed by a brief explanation."
            ),
        }
        user_msg: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": prompt,
        }

        response = self.client.chat.completions.create(
            model="NousResearch/Hermes-4-405B",
            messages=[system_msg, user_msg],
        )
        return self.clean_text(response.choices[0].message.content.strip())

    async def generate_code_with_mcp(self, prompt: str, instance_id: int, use_rag: bool = False) -> str:
        """Generates Python code using MCP tools based on the provided prompt and instance context."""
        if use_rag:
            instruction = f"You have access to MCP tools INCLUDING RAG. ONLY use domains from instance {instance_id}."
        else:
            instruction = f"You have access to MCP tools. ONLY use domains from instance {instance_id}."

        system_message = f"""
You are a code generation assistant specialized in creating Python code based on OpenAPI specifications.
{instruction}
WORKFLOW:
1. Call list_available_domains(instance_id={instance_id}) ONCE.
2. Choose EXACTLY ONE domain path.
3. Call {'retrieve_relevant_specs_with_rag' if use_rag else 'load_openapi_specs'} ONCE.
4. Generate Python code using 'requests' library.
5. Return ONLY Python code, NO explanations.
"""

        tools = await self._list_mcp_tools(use_rag=use_rag)

        system_msg: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": system_message,
        }
        user_msg: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": prompt,
        }
        messages: list[ChatCompletionMessageParam] = [system_msg, user_msg]

        response = self.client.chat.completions.create(
            model="NousResearch/Hermes-4-405B",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        iteration = 0
        while response.choices[0].message.tool_calls and iteration < 5:
            iteration += 1
            assistant_message = response.choices[0].message

            tool_calls_param = [
                ChatCompletionMessageToolCallParam(
                    id=tc.id,
                    type="function",
                    function=ToolCallFunction(name=tc.function.name, arguments=tc.function.arguments),
                ) for tc in (assistant_message.tool_calls or [])
            ]
            assistant_msg: ChatCompletionAssistantMessageParam = {
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": tool_calls_param,
            }
            messages.append(assistant_msg)
            for tool_call in assistant_message.tool_calls:
                tool_result = await self._call_mcp_tool(tool_call.function.name,
                                                        json.loads(tool_call.function.arguments))
                if isinstance(tool_result, str):
                    result_str = tool_result
                else:
                    result_str = json.dumps(tool_result)
                tool_msg: ChatCompletionToolMessageParam = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                }
                messages.append(tool_msg)

            response = self.client.chat.completions.create(
                model="NousResearch/Hermes-4-405B",
                messages=messages,
                tools=tools
            )

        final_content = response.choices[0].message.content or ""
        return self.clean_generated_code(final_content)
