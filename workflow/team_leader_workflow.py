from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from openai import OpenAI
from typing import Literal

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import logging
from model import model_choose

from pydantic import Field
load_dotenv()
model = ChatOpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

class State(TypedDict):
    input: str  # user query input
    output: str  # model output answer
    conversation_history: list  # store all conversation history
    messages: list  # LangChain message history for tool calling
    task_completed: bool  # flag to check if task is completed
    # iteration_count: int=Field(default=0)  # counter to prevent infinite loops

@tool
async def llm_chat(state: State) -> State:
    """Create an answer to the user's common chat."""
    try:
        agent = ChatOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            model="deepseek-chat")
        result = await agent.ainvoke({"messages": [HumanMessage(content=state['input'])]})
        print(result)
        logging.debug(result)
        # Extract the AI's response from the result
        ai_message = None
        if 'messages' in result:
            # Find the last AI message in the conversation
            for msg in reversed(result['messages']):
                if isinstance(msg, AIMessage):
                    ai_message = msg
                    break
        
        response = ai_message.content if ai_message else "Could not generate a response"
    except Exception as e:
        logging.error(f"Error in llm_query: {str(e)}")
        response = f"An error occurred while processing your request: {str(e)}"
    
    # Only return the response string
    return response
@tool
async def llm_query(state: State) -> State:
    """Create an MCP client that connects to both calculator and weather servers"""
    # Create an MCP client that connects to both calculator and weather servers
    client = MultiServerMCPClient(
        {
            "calculator_service": {
                "transport": "stdio",
                "command": "python",
                "args": ["mcp_tool/calculate_server.py"],
            },
            "weather_service": {
                "transport": "stdio",
                "command": "python",
                "args": ["mcp_tool/weather_server.py"],
            }
        }
    )

    try:
        # Get the tools from the server
        mcp_tools = await client.get_tools()

        # Create an agent using the create_agent function
        agent = create_agent(
            "deepseek-chat", 
            mcp_tools,
            system_prompt=SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": f"You are a helpful AI assistant. Answer the user's query: {state['input']}",
                    }
                ]
            )
        )
        
        result = await agent.ainvoke({"messages": [HumanMessage(content=state['input'])]})
        print(result)
        logging.debug(result)
        # Extract the AI's response from the result
        ai_message = None
        if 'messages' in result:
            # Find the last AI message in the conversation
            for msg in reversed(result['messages']):
                if isinstance(msg, AIMessage):
                    ai_message = msg
                    break
        
        response = ai_message.content if ai_message else "Could not generate a response"
    except Exception as e:
        logging.error(f"Error in llm_query: {str(e)}")
        response = f"An error occurred while processing your request: {str(e)}"
    
    # Only return the response string
    return response

@tool
async def llm_rag(state: State) -> str:
    """RAG implementation placeholder"""
    # Placeholder for RAG implementation (will be implemented later)
    response = {"output": f"RAG functionality not yet implemented Response to: {state['input']}"}
    # Only return the response string
    return response

def team_leader(state: State) -> State:
    """
    Use agent to understand user intent and decide workflow path by calling appropriate tool
    """
    client = ChatOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        model="deepseek-chat")

    prompt = f"""
    You are a helpful AI assistant. Based on the user's query: {state['input']},
    decide whether to use llm_chat for common conversations, llm_query for calculations, weather info,
    or llm_rag for document retrieval or research.
    
    If the query is about common chat (for example, common greeting ,.etc), call the llm_chat tool.
    If the query is related to calculations, math operations, weather information or common conversations, call the llm_query tool.
    If it requires document retrieval, research, or detailed information retrieval, call the llm_rag tool.
    
    Additionally, if you choose llm_query, you should translate the query city name into English with lowercase before calling llm_query.
    """
    
    # Bind the tools to the client
    model_with_tool = client.bind_tools([llm_chat, llm_query, llm_rag])
    logging.debug(f"team_leader input: {state['input']}")

    # Use the tool-bound model to decide and return a response with tool call
    response = model_with_tool.invoke([
        HumanMessage(content=prompt)
    ])
    logging.debug(f"team_leader response: {response}")

    # Return state with the response containing the tool call
    state = state.copy()
    state["messages"] = [HumanMessage(content=state['input']), response]

    return state


def check_completion(state: State) -> State:
    """
    Determines if the task has been completed based on the user query and AI responses
    """
    
    # Combine all responses for analysis
    #all_responses = " ".join([item['content'] for item in conversation_history if item['role'] == 'assistant'])
    
    client = ChatOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        model="deepseek-chat")

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
    response = client.invoke([
        HumanMessage(content=prompt)
    ])
    logging.debug(f"check_completion response: {response.content}")
    logging.debug(f"state: {state}")
    state['task_completed'] = response.content
    # state['task_completed'] = True # for debug only
    return state


def route_based_on_decision(state: State) -> str:
    """
    Route to llm_chat, llm_query or llm_rag based on team leader decision
    """
    decision = team_leader(state)
    return decision

tool_node = ToolNode([llm_chat, llm_query, llm_rag])

async def retrieve(state: State) -> State:
    """Execute tool and update state with result"""
    # Call ToolNode to execute the tool
    result_state = await tool_node.ainvoke(state)
    
    # Extract the tool result from messages
    messages = result_state.get("messages", [])
    output = state.get("output", "")
    
    # The last message should be ToolMessage with the tool result
    for msg in reversed(messages):
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            output = msg.content
            break
    
    # Update conversation history with the output
    updated_history = state.get('conversation_history', []).copy()
    updated_history.append({"role": "assistant", "content": output})
    
    # Return updated state
    return {
        "input": state["input"],
        "output": output,
        "conversation_history": updated_history,
        "messages": messages,
        "task_completed": state.get("task_completed", False)
    }

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