from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from workflow.rag_workflow import rag_graph
import os
import logging

load_dotenv()

class State(TypedDict):
    input: str  # user query input
    output: str  # model output answer
    conversation_history: list  # store all conversation history
    messages: list  # LangChain message history for tool calling
    task_completed: bool  # flag to check if task is completed
    retrieved_answers: NotRequired[int]  # count of retrieved answers, defaults to 5
    retrieved_docs: NotRequired[list]  # raw retrieved documents with similarity scores

# Global agent instance
agent = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat")

@tool
async def llm_chat(query: str) -> str:
    """
    MCP-based tool for common conversations
    """
    result = await agent.ainvoke([HumanMessage(content=query)])
    # ainvoke returns AIMessage directly
    if isinstance(result, AIMessage):
        return result.content
    return "No response"

@tool
async def llm_query(query: str) -> str:
    """
    MCP-based tool for calculator / weather / general query
    """
    client = MultiServerMCPClient(
        {
            "calculator_service": {
                "transport": "stdio",
                "command": "python",
                "args": ["mcp_tool/calculator_server.py"],
            },
            "weather_service": {
                "transport": "stdio",
                "command": "python",
                "args": ["mcp_tool/weather_server.py"],
            }
        }
    )

    try:
        mcp_tools = await client.get_tools()

        agent = create_agent(
            "deepseek-chat",
            mcp_tools,
            system_prompt=SystemMessage(
                content=f"You are a helpful AI assistant. Answer the user's query: {query}"
            )
        )

        result = await agent.ainvoke({
            "messages": [HumanMessage(content=query)]
        })

        ai_message = None
        if "messages" in result:
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    ai_message = msg
                    break

        return ai_message.content if ai_message else "No response"

    except Exception as e:
        logging.error(f"llm_query error: {e}")
        return f"Error: {e}"

@tool
async def llm_rag(
    query: str,
    retrieved_answers: int = 5
) -> str:
    """
    RAG tool: Retrieve documents and generate summary based on them.
    Returns JSON string with summary and retrieved documents.
    """
    try:
        result = await rag_graph.ainvoke({
            "input": query,
            "retrieved_answers": retrieved_answers,
            "messages": [],
            "conversation_history": [],
            "output": "",
            "task_completed": False,
        })

        # Extract the summary output and retrieved documents
        summary = result.get("output", "No summary generated")
        retrieved_docs = result.get("retrieved_docs", [])

        # Return as JSON string
        import json
        return json.dumps({
            "summary": summary,
            "retrieved_docs": retrieved_docs
        }, ensure_ascii=False)

    except Exception as e:
        logging.error(f"llm_rag error: {e}")
        import json
        return json.dumps({
            "summary": f"RAG error: {e}",
            "retrieved_docs": []
        }, ensure_ascii=False)

# def team_leader(state: State) -> State:
#     """
#     Use agent to understand user intent and decide workflow path by calling appropriate tool
#     """

#     prompt = f"""
#     You are a helpful AI assistant. Based on the user's query: {state['input']},
#     decide whether to use llm_chat for common conversations, llm_query for calculations, weather info,
#     or llm_rag for document retrieval or research.

#     If the query is about common chat (for example, common greeting ,.etc), call the llm_chat tool.
#     If the query is related to calculations, math operations, weather information or common conversations, call the llm_query tool.
#     If it requires document retrieval, research, or detailed information retrieval, call the llm_rag tool.
    
    
#     Additionally, if you choose weather realated tool in llm_query, you should translate the query city name into English with lowercase before calling llm_query.
#     If you call llm_rag, you MUST pass:
#     - query = the user query
#     - retrieved_answers = {state.get("retrieved_answers")}
#     """
#     logging.debug(f"team_leader state: {state}")
#     # Bind the tools to the client
#     model_with_tool = agent.bind_tools([llm_chat, llm_query, llm_rag])

#     response = model_with_tool.invoke([
#         HumanMessage(content=prompt)
#     ])

#     new_state = state.copy()
#     new_state["messages"] = [
#         HumanMessage(content=state["input"]),
#         response
#     ]

async def team_leader(state: State) -> State:
    """Route user query to appropriate tool using LLM."""
    # Get existing messages or initialize with user input
    messages = state.get("messages", [])

    # If this is the first call, add the user's message
    if not messages or len(messages) == 0:
        messages = [HumanMessage(content=state["input"])]

    # Build the prompt for the router
    prompt = f"""
You are a router agent.

User query:
{state['input']}

Decide which tool to call:

- llm_chat(query: str) - for common conversations
- llm_query(query: str) - for calculations, weather info, general queries
- llm_rag(query: str, retrieved_answers: int) - for document retrieval or research

IMPORTANT:
You MUST call exactly ONE tool.
If you choose llm_rag, you MUST pass retrieved_answers={state.get("retrieved_answers", 5)}.
"""

    # Add the prompt as a human message
    messages.append(HumanMessage(content=prompt))

    model_with_tools = agent.bind_tools(
        [llm_chat, llm_query, llm_rag]
    )

    ai_message = await model_with_tools.ainvoke(messages)

    # Update state with the AI response (which should contain tool_calls)
    new_state = state.copy()
    new_state["messages"] = messages + [ai_message]

    return new_state


async def check_completion(state: State) -> State:
    """
    Determines if the task has been completed based on the user query and AI responses.
    Also extracts the tool output and retrieved_docs to the state.
    """
    # Extract tool output from messages
    output = ""
    retrieved_docs = []
    messages = state.get("messages", [])

    # Look for the last tool message (ToolMessage)
    for msg in reversed(messages):
        if hasattr(msg, 'content') and msg.content:
            content = msg.content
            # Try to parse as JSON (from llm_rag)
            try:
                import json
                if isinstance(content, str) and content.startswith('{'):
                    parsed = json.loads(content)
                    if "summary" in parsed:
                        # This is from llm_rag
                        output = parsed.get("summary", content)
                        retrieved_docs = parsed.get("retrieved_docs", [])
                    else:
                        output = content
                else:
                    output = content
            except (json.JSONDecodeError, TypeError):
                output = content
            break

    new_state = state.copy()
    new_state["output"] = output
    new_state["retrieved_docs"] = retrieved_docs

    # For now, always mark as completed after one tool execution
    # This prevents infinite loops
    new_state["task_completed"] = True

    logging.debug(f"check_completion output: {output[:200] if output else 'No output'}...")
    return new_state


# Create ToolNode with all tools
tool_node = ToolNode([llm_chat, llm_query, llm_rag])


# Create workflow graph
workflow = StateGraph(State)

# Define nodes
workflow.add_node("team_leader", team_leader)
workflow.add_node("tools", tool_node)
workflow.add_node("check_completion", check_completion)

# Add entry point
workflow.add_edge(START, "team_leader")

# Add conditional edges from team_leader
# If tool_calls exist, go to tools node, otherwise go to check_completion
workflow.add_conditional_edges(
    "team_leader",
    # If there are tool calls, route to tools; otherwise END
    lambda state: "tools" if any(
        msg.tool_calls for msg in state.get("messages", []) if hasattr(msg, "tool_calls")
    ) else "check_completion",
    {
        "tools": "tools",
        "check_completion": "check_completion"
    }
)

# After tools execute, go to check_completion
workflow.add_edge("tools", "check_completion")

# Check completion and decide whether to end or continue
workflow.add_conditional_edges(
    "check_completion",
    lambda state: "team_leader" if not (state.get('task_completed', False) in [True, 'True', 'true', 'TRUE']) else END,
    {
        "team_leader": "team_leader",
        END: END
    }
)

# Compile the graph (async-compatible)
graph = workflow.compile()