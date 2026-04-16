# import json, os, re, logging
# from openai import OpenAI
# from openai.types.chat import (
#     ChatCompletionMessageParam,
#     ChatCompletionSystemMessageParam,
#     ChatCompletionUserMessageParam,
#     ChatCompletionAssistantMessageParam,
#     ChatCompletionToolMessageParam,
# )
# from openai.types.chat.chat_completion_message_tool_call_param import (
#     ChatCompletionMessageToolCallParam,
#     Function as ToolCallFunction,
# )
# from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
# from openai.types.shared_params import FunctionDefinition
# from a2a.server.agent_execution import AgentExecutor, RequestContext
# from a2a.server.events import EventQueue
# from a2a.utils import new_task, new_agent_text_message
# from mcp import ClientSession
# from mcp.client.sse import sse_client
#
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("PurpleExecutor")
#
# class PurpleExecutor(AgentExecutor):
#
#     def __init__(self, api_key_env: str = "NEBIUS_API_KEY"):
#         self.api_key = os.getenv(api_key_env)
#         if not self.api_key:
#             raise ValueError(f"Environment variable {api_key_env} not set")
#         self.mcp_url = "http://localhost:8000/sse"
#         self.client = OpenAI(
#             base_url="https://api.tokenfactory.nebius.com/v1/",
#             api_key=self.api_key
#         )
#         self.mcp_session: ClientSession | None = None
#         self._sse_context = None
#         self._session_context = None
#         self.mcp_initialized = False
#
#     async def _ensure_mcp_connected(self):
#         if self.mcp_initialized:
#             logging.info("MCP session already initialized")
#             return
#
#         try:
#             logger.info(f"Connecting to shared MCP Server at {self.mcp_url}")
#
#             self._sse_context = sse_client(self.mcp_url)
#             streams = await self._sse_context.__aenter__()
#             stdio, write = streams
#
#             self._session_context = ClientSession(stdio, write)
#             self.mcp_session = await self._session_context.__aenter__()
#
#             await self.mcp_session.initialize()
#             self.mcp_initialized = True
#             logger.info("Successfully connected to shared MCP session")
#
#         except Exception as e:
#             logger.error(f"Failed to connect to central MCP: {e}")
#             self.mcp_initialized = False
#             raise
#
#     async def _cleanup_mcp(self):
#         if self._session_context is not None:
#             try:
#                 await self._session_context.__aexit__(None, None, None)
#             except Exception as e:
#                 logger.warning(f"Error closing MCP session: {e}")
#             self._session_context = None
#             self.mcp_session = None
#
#         if self._sse_context is not None:
#             try:
#                 await self._sse_context.__aexit__(None, None, None)
#             except Exception as e:
#                 logger.warning(f"Error closing SSE connection: {e}")
#             self._sse_context = None
#
#         self.mcp_initialized = False
#
#     async def _call_mcp_tool(self, tool_name: str, arguments: dict):
#         await self._ensure_mcp_connected()
#
#         try:
#             result = await self.mcp_session.call_tool(tool_name, arguments)
#             if result.content:
#                 if hasattr(result.content[0], 'text'):
#                     return result.content[0].text
#                 return str(result.content[0])
#             return {}
#         except Exception as e:
#             logger.error(f"MCP Tool call failed: {str(e)}", exc_info=True)
#             return {"error": str(e)}
#
#     async def _list_mcp_tools(self, use_rag: bool = False) -> list[ChatCompletionToolParam]:
#         await self._ensure_mcp_connected()
#
#         tools_list = await self.mcp_session.list_tools()
#         openai_tools: list[ChatCompletionToolParam] = []
#         for tool in tools_list.tools:
#             if not use_rag and tool.name == "retrieve_relevant_specs_with_rag":
#                 continue
#             if use_rag and tool.name == "load_openapi_specs":
#                 continue
#             openai_tool: ChatCompletionToolParam = {
#                 "type": "function",
#                 "function": FunctionDefinition(
#                      name=tool.name,
#                      description=tool.description or "",
#                      parameters=tool.inputSchema
#                 )
#             }
#             openai_tools.append(openai_tool)
#
#         logger.info(f"Available OpenAI MCP tools: {[tool['function']['name'] for tool in openai_tools]}")
#         return openai_tools
#
#     @staticmethod
#     def extract_context(message: str) -> tuple[str, dict | None]:
#         match = re.search(r'\[Context:\s*(\{.*?})]', message, re.DOTALL)
#
#         if match:
#             try:
#                 context = json.loads(match.group(1))
#                 clean_task = re.sub(r'\[Context:.*?]', '', message, flags=re.DOTALL).strip()
#                 return clean_task, context
#             except json.JSONDecodeError:
#                 logger.warning("Found context marker but couldn't parse JSON")
#
#         return message, None
#
#     @staticmethod
#     def clean_generated_code(text: str) -> str:
#         text = text.encode("ascii", "ignore").decode()
#         code_blocks = re.findall(r"```(?:[\w+-]*)?\n(.*?)```", text, re.DOTALL)
#         if code_blocks:
#             filtered_blocks = [
#                 block.strip() for block in code_blocks
#                 if block.strip() and not block.strip().startswith("pip install")
#             ]
#
#             if filtered_blocks:
#                 cleaned = max(filtered_blocks, key=len)
#                 cleaned = re.sub(r"^#.*\n", "", cleaned)
#                 return cleaned
#             return ""
#         return text.strip()
#
#     @staticmethod
#     def clean_text(text: str) -> str:
#         result = text.encode("ascii", "ignore").decode().strip()
#         return result
#
#     async def generate_code_with_mcp(self, prompt: str, instance_id: int, use_rag: bool = False) -> str:
#         if use_rag:
#             system_message = f"""
# You are a code generation assistant specialized in creating Python code based on OpenAPI specifications.
#
# You have access to MCP (Model Context Protocol) tools INCLUDING RAG capabilities for loading OpenAPI specifications.
#
# WORKFLOW:
# 1. You must ONLY use domains from instance {instance_id}
# 2. Call list_available_domains(instance_id={instance_id}) ONCE to see available domains
# 3. Choose EXACTLY ONE domain from the list (the domain path like "socbenchd_1/01-energy")
# 4. Call retrieve_relevant_specs_with_rag(domain_path, query) ONCE with the chosen domain
# 5. Generate Python code using 'requests' library based on the retrieved specs to answer the user's request
# 6. Return ONLY Python code, NO explanations
# """
#         else:
#             system_message = f"""
# You are a code generation assistant specialized in creating Python code based on OpenAPI specifications.
#
# You have access to MCP (Model Context Protocol) tools to load OpenAPI specifications.
#
# WORKFLOW:
# 1. You must ONLY use domains from instance {instance_id}
# 2. Call list_available_domains(instance_id={instance_id}) ONCE to see available domains
# 3. Choose EXACTLY ONE domain from the list (the domain path like "socbenchd_1/01-energy")
# 4. Call load_openapi_specs(domain_path) ONCE with the chosen domain path
# 5. Generate Python code using 'requests' library based on the loaded specs to answer the user's request
# 6. Return ONLY Python code, NO explanations
# """
#
#         tools = await self._list_mcp_tools(use_rag=use_rag)
#
#         system_msg: ChatCompletionSystemMessageParam = {
#             "role": "system",
#             "content": system_message,
#         }
#         user_msg: ChatCompletionUserMessageParam = {
#             "role": "user",
#             "content": prompt,
#         }
#         messages: list[ChatCompletionMessageParam] = [system_msg, user_msg]
#
#         try:
#             logger.info("Calling LLM (1st call)...")
#             response = self.client.chat.completions.create(
#                 model="moonshotai/Kimi-K2-Instruct",
#                 messages=messages,
#                 tools=tools,
#                 tool_choice="auto"
#             )
#             logger.info("LLM responded successfully")
#         except Exception as e:
#             logger.error(f"LLM call failed: {str(e)}", exc_info=True)
#             raise
#
#         iteration = 0
#         max_iterations = 5
#
#         while response.choices[0].message.tool_calls and iteration < max_iterations:
#             iteration += 1
#             logger.info(f"Tool iteration {iteration}/{max_iterations}")
#
#             assistant_message = response.choices[0].message
#             tool_calls_param: list[ChatCompletionMessageToolCallParam] = [
#                 ChatCompletionMessageToolCallParam(
#                     id=tc.id,
#                     type="function",
#                     function=ToolCallFunction(
#                         name=tc.function.name,
#                         arguments=tc.function.arguments,
#                     ),
#                 )
#                 for tc in (assistant_message.tool_calls or [])
#             ]
#             assistant_msg: ChatCompletionAssistantMessageParam = {
#                 "role": "assistant",
#                 "content": assistant_message.content,
#                 "tool_calls": tool_calls_param,
#             }
#             messages.append(assistant_msg)
#
#             for tool_call in assistant_message.tool_calls:
#                 tool_name = tool_call.function.name
#
#                 try:
#                     tool_args = json.loads(tool_call.function.arguments)
#                     logger.info(f"Calling MCP tool: {tool_name}")
#                     logger.info(f"Tool args: {json.dumps(tool_args, indent=2)}")
#                     tool_result = await self._call_mcp_tool(tool_name, tool_args)
#                     if isinstance(tool_result, str):
#                         result_str = tool_result
#                     else:
#                         result_str = json.dumps(tool_result)
#
#                 except Exception as e:
#                    result_str = json.dumps({"error": str(e)})
#
#                 tool_msg: ChatCompletionToolMessageParam = {
#                     "role": "tool",
#                     "tool_call_id": tool_call.id,
#                     "content": result_str,
#                 }
#                 messages.append(tool_msg)
#
#             try:
#                 logger.info(f"Calling LLM with tool results (iteration {iteration})...")
#                 response = self.client.chat.completions.create(
#                     model="moonshotai/Kimi-K2-Instruct",
#                     messages=messages,
#                     tools=tools,
#                     tool_choice="auto"
#                 )
#             except Exception as e:
#                 logger.error(f"LLM call failed: {str(e)}", exc_info=True)
#                 raise
#
#         final_response = response.choices[0].message.content
#         if final_response is None:
#             final_response = ""
#
#         cleaned_code = self.clean_generated_code(final_response)
#         return cleaned_code
#
#     async def generate_text(self, prompt: str) -> str:
#         system_msg: ChatCompletionSystemMessageParam = {
#             "role": "system",
#             "content": (
#                 "You are a reviewer agent. "
#                 "Confirm or reject whether extracted API endpoints match the user's intent. "
#                 "Respond with 'Yes' or 'No' followed by a brief explanation."
#             ),
#         }
#         user_msg: ChatCompletionUserMessageParam = {
#             "role": "user",
#             "content": prompt,
#         }
#         messages: list[ChatCompletionMessageParam] = [system_msg, user_msg]
#
#         response = self.client.chat.completions.create(
#             model="moonshotai/Kimi-K2-Instruct",
#             messages=messages,
#         )
#
#         result = self.clean_text(response.choices[0].message.content.strip())
#         return result
#
#     async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
#         request_text = context.get_user_input()
#         task_description, json_context = self.extract_context(request_text)
#
#         try:
#             instance_id = json_context.get("instance_id", 1)
#             scenario = json_context.get("scenario", "")
#             mode = json_context.get("mode", "")
#
#             if mode == "confirm":
#                 logger.info("Execution mode: confirm")
#             else:
#                 logger.info(f"Using context hints: scenario={scenario}, instance={instance_id}, mode={mode}")
#
#         except (json.JSONDecodeError, AttributeError) as e:
#             logger.error(f"Failed to parse context JSON: {str(e)}", exc_info=True)
#             msg = context.message
#             if msg:
#                 task = new_task(msg)
#                 await event_queue.enqueue_event(task)
#                 from a2a.server.tasks import TaskUpdater
#                 updater = TaskUpdater(event_queue, task.id, task.context_id)
#                 await updater.failed(
#                     new_agent_text_message(
#                         f"Error parsing context JSON: {str(e)}",
#                         context_id=context.context_id
#                     )
#                 )
#             return
#
#         msg = context.message
#         if msg:
#             task = new_task(msg)
#             await event_queue.enqueue_event(task)
#             logger.info(f"Task created: {task.id}")
#         else:
#             logger.error("No message context")
#             return
#
#         from a2a.server.tasks import TaskUpdater
#         updater = TaskUpdater(event_queue, task.id, task.context_id)
#
#         try:
#             logger.info(f"{task_description}")
#
#             if mode == "confirm":
#                 result = await self.generate_text(task_description)
#             else:
#                 use_rag = (mode == "rag")
#                 logger.info(f"Generating code with MCP (use_rag={use_rag})...")
#                 result = await self.generate_code_with_mcp(task_description, instance_id, use_rag)
#
#             logger.info(f"Result preview:\n{result[:1000]}...")
#
#             await updater.complete(
#                 message=new_agent_text_message(result, context_id=context.context_id)
#             )
#             logger.info("Task completed successfully")
#
#         except Exception as e:
#             logger.error(f"Execution failed: {str(e)}", exc_info=True)
#             await updater.failed(
#                 new_agent_text_message(
#                     f"Error during execution: {str(e)}",
#                     context_id=context.context_id
#                 )
#             )
#         finally:
#             await self._cleanup_mcp()
#     async def cancel(self, request: RequestContext, event_queue: EventQueue):
#         await self._cleanup_mcp()
#         return None