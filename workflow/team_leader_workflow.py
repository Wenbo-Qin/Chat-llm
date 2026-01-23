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
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage):
            return msg.content
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
    RAG tool: only cares about query + retrieved_answers
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

        ai_message = None
        if "output" in result:
            ai_message = result["output"]

        return ai_message if ai_message else "No RAG result"

    except Exception as e:
        logging.error(f"llm_rag error: {e}")
        return f"RAG error: {e}"

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

def team_leader(state: State) -> State:
    prompt = f"""
You are a router agent.

User query:
{state['input']}

Decide which tool to call:

- llm_chat(query: str)
- llm_query(query: str)
- llm_rag(query: str, retrieved_answers: int)

IMPORTANT:
You MUST call exactly ONE tool.
If you choose llm_rag, you MUST answer based on documents that retrieved.
"""
    model_with_tools = agent.bind_tools(
        [llm_chat, llm_query, llm_rag]
    )

    ai_message = model_with_tools.invoke(
        [HumanMessage(content=prompt)]
    )
    # ⚠️ 关键：ai_message 必须包含 tool_calls
    new_state = state.copy()
    new_state["messages"] = [
        HumanMessage(content=state["input"]),
        ai_message,
    ]

    return new_state

def check_completion(state: State) -> State:
    """
    Determines if the task has been completed based on the user query and AI responses
    """
    
    # Combine all responses for analysis
    #all_responses = " ".join([item['content'] for item in conversation_history if item['role'] == 'assistant'])

    prompt = f"""
    You are a helpful AI assistant. Based on the AI answer: {state['output']} and user query: {state['input']}
    decide whether to end conversation or still use the model to generate a new answer. 
    
    If the answer is corresponding to the user query and the task is complete, respond with True.
    If the answer is not corresponding to the user query and the task is not complete, respond with False.
    If user did not ask a question, respond with True.
    
    Do not respond with anything else.

    For example, in these two cases, the tasks are complete, so you should respond with True:
    ### Case 1:
    User: My name is John.
    AI: I am a helpful AI assistant. How can I help you?
    
    ### Case 2:
    User: What's the weather like in New York City?
    AI: The weather in New York City is sunny with a high of 75 degrees.
    """
    
    # # Simple heuristic to determine if the task is complete
    # completion_indicators = [
    #     "calculate" in user_query and ("equals" in all_responses.lower() or "result" in all_responses.lower()),
    #     "weather" in user_query and "degree" in all_responses.lower(),
    #     "thank" in user_query,  # User thanked, implying satisfaction
    #     state.get('iteration_count', 0) >= 5  # Prevent infinite loops
    # ]
    
    # Return updated state with completion status
    response = agent.invoke([
        HumanMessage(content=prompt)
    ])
    logging.debug(f"check_completion response: {response.content}")
    logging.debug(f"state: {state}")
    new_state = state.copy()
    new_state["task_completed"] = response.content
    return new_state


def route_based_on_decision(state: State) -> str:
    """
    Route to llm_chat, llm_query or llm_rag based on team leader decision
    """
    decision = team_leader(state)
    return decision

tool_node = ToolNode([llm_chat, llm_query, llm_rag])

async def retrieve(state: State) -> State:
    logging.debug(f"[retrieve] before tool: {state}")

    result_state = await tool_node.ainvoke(state)

    output = ""
    for msg in reversed(result_state.get("messages", [])):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            output = msg.content
            break

    new_state = state.copy()
    new_state["output"] = output
    new_state["messages"] = result_state.get("messages", [])
    return new_state


# Create workflow graph  
workflow = StateGraph(State)

# Define nodes
workflow.add_node("team_leader", team_leader)
workflow.add_node("retrieve", retrieve)
workflow.add_node("check_completion", check_completion)

# Add entry point
workflow.add_edge(START, "team_leader")

# Add conditional edges
workflow.add_conditional_edges(
    "team_leader",
    tools_condition,
    {
        "tools": "retrieve",
        # END: END
    }
)

workflow.add_edge("retrieve", "check_completion")

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