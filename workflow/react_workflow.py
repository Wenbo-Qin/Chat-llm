"""
ReAct Workflow Implementation

This module implements a ReAct (Reasoning + Acting) agent pattern that:
1. THOUGHT: Analyzes the current situation and decides what to do
2. ACTION: Executes a tool or provides a final answer
3. OBSERVATION: Observes the result of the action
4. Iterates until the task is complete

The RAG subgraph (rag_workflow.py) remains unchanged and is called as a tool.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path for imports
# This allows the file to be run directly from the workflow directory
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging
from typing_extensions import TypedDict, NotRequired, Literal
from typing import List, Any
from langgraph.graph import StateGraph, START, END
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Import existing tools
from workflow.team_leader_workflow import llm_chat, llm_query, llm_rag

# Import history loading
from db_service.history_conversations import load_history_conversation

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReActState(TypedDict):
    """
    State for the ReAct workflow.

    Attributes:
        messages: Conversation history including user messages, AI responses, and tool results
        input: Original user input
        output: NotRequired - Final answer when the task is complete
        next: NotRequired - Next step in the workflow ('agent', 'tools', 'end')
        iteration_count: NotRequired - Track iterations to prevent infinite loops
        max_iterations: NotRequired - Maximum number of ReAct iterations allowed
        retrieved_docs: NotRequired - Retrieved documents with similarity scores from RAG
        retrieved_answers: NotRequired - Count of retrieved answers from RAG
    """
    messages: List[Any]
    input: str
    output: NotRequired[str]
    next: NotRequired[str]
    iteration_count: NotRequired[int]
    max_iterations: NotRequired[int]
    retrieved_docs: NotRequired[List[dict]]
    retrieved_answers: NotRequired[int]


# Global LLM instance for the ReAct agent
_react_agent = None

# # GLM 4.7 model
# def get_react_agent():
#     """Get or create global ReAct agent instance."""
#     global _react_agent
#     if _react_agent is None:
#         _react_agent = ChatOpenAI(
#             openai_api_key=os.getenv("ZHIPUAI_API_KEY"),
#             openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
#             model="glm-4.7",
#             temperature=0  # Lower temperature for more deterministic reasoning
#         )
#     return _react_agent

def get_react_agent():
    """Get or create global ReAct agent instance."""
    global _react_agent
    if _react_agent is None:
        _react_agent = ChatOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            temperature=0  # Lower temperature for more deterministic reasoning
        )
    return _react_agent


def create_react_system_prompt(retrieved_answers: int = 5) -> str:
    """
    Create the system prompt for the ReAct agent.

    Args:
        retrieved_answers: Number of documents to retrieve when using RAG

    Returns:
        System prompt that instructs the LLM to follow ReAct pattern
    """
    return f"""You are an intelligent AI assistant that helps users with their questions.

    ## Your Internal Thinking Process (do not include in your response):

    1. Analyze the user's request and the current situation
    2. Decide what to do:
    - If you need more information → call a tool
    - If you have enough information → provide a final answer
    3. After a tool is executed, review the result
    4. If needed, perform another action; otherwise, answer

    ## Available Tools:

    - **llm_chat(query: str)**: Use for general conversations, greetings, casual chat
    - **llm_query(query: str)**: Use for math calculations, weather information, or general factual queries
    - **llm_rag(query: str, retrieved_answers: int)**: If and only if user ask Psychology, camera related question, Use it for document retrieval, research. If you use llm_rag, you are not allowed to use other tools.

    ## Guidelines:

    1. Think step by step internally before taking action
    2. Be specific when calling tools - provide clear and specific queries
    3. Use tools efficiently - don't call tools if you can answer from your knowledge
    4. Iterate if needed - if the first tool call doesn't give you enough information, try another app
    5. Provide clear, natural, and conversational answers to users

    ## CRITICAL - Response Format:

    - DO NOT include "THOUGHT:", "ANSWER:", "ACTION:", or "OBSERVATION:" labels in your responses
    - DO NOT show your internal reasoning process to users
    - When providing your final answer, respond naturally as if you're having a conversation
    - Your responses should be clean, user-friendly text without structured tags or labels
    - Users should see only your final answer, not your thinking process

    ## Important:

    - You MUST call exactly ONE tool at a time
    - If you call llm_rag, you MUST use retrieved_answers={retrieved_answers}
    - After each tool execution, evaluate if you need more actions
    - When you're ready to answer, provide the final response naturally WITHOUT calling another tool

    Remember: Keep your responses conversational and professional. Users should not see any internal reasoning labels."""


async def react_agent_node(state: ReActState) -> ReActState:
    """
    ReAct agent node that performs reasoning and decides actions.

    This node:
    1. Analyzes the conversation history
    2. Decides whether to call a tool or provide a final answer
    3. Returns the AI's response (which may include tool_calls)

    Args:
        state: Current ReAct state containing messages and iteration info

    Returns:
        Updated state with the AI's response added to messages
    """
    messages = state["messages"]
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)
    retrieved_answers = state.get("retrieved_answers", 5)

    # Check iteration limit to prevent infinite loops
    if iteration_count >= max_iterations:
        logger.warning(f"Max iterations ({max_iterations}) reached, forcing completion")

        # Generate final answer based on all previous tool results
        final_prompt = """Based on all the information gathered from tool calls in this conversation,
        please provide a comprehensive final answer to the user's original question.
        Summarize all findings and give a clear, structured response."""

        final_messages = messages + [HumanMessage(content=final_prompt)]
        response = await get_react_agent().ainvoke(final_messages)

        new_state = state.copy()
        new_state["messages"] = messages + [response]
        new_state["output"] = response.content
        new_state["next"] = "end"

        return new_state

    # Prepare messages for LLM
    # Only add system prompt on first iteration
    if iteration_count == 0:
        # Check if system prompt already exists
        has_system = any(isinstance(msg, SystemMessage) for msg in messages)
        if not has_system:
            system_msg = SystemMessage(content=create_react_system_prompt(retrieved_answers=retrieved_answers))
            messages_with_system = [system_msg] + messages
        else:
            messages_with_system = messages
    else:
        # On subsequent iterations, use messages as-is (they already have the full history)
        messages_with_system = messages

    # Call the LLM to get reasoning and action decisionroach
    try:
        # Bind tools to the agent so it can call them
        agent_with_tools = get_react_agent().bind_tools([llm_chat, llm_query, llm_rag])

        response = await agent_with_tools.ainvoke(messages_with_system)

        new_state = state.copy()
        new_state["messages"] = messages_with_system + [response]
        new_state["iteration_count"] = iteration_count + 1

        # Log the reasoning (if present in response content)
        if hasattr(response, 'content') and response.content:
            logger.info(f"ReAct iteration {iteration_count + 1}: {response.content[:200]}...")

        # Log tool calls if present
        if hasattr(response, 'tool_calls') and response.tool_calls:
            logger.info(f"Tool calls: {[call['name'] for call in response.tool_calls]}")

        return new_state

    except Exception as e:
        logger.error(f"Error in react_agent_node: {e}")

        # Return error state
        error_state = state.copy()
        error_state["messages"] = messages + [
            AIMessage(content=f"I encountered an error: {str(e)}. Please try again.")
        ]
        error_state["next"] = "end"
        return error_state


def should_continue(state: ReActState) -> Literal["tools", "end"]:
    """
    Determine whether to continue the ReAct loop or end.

    This function checks the last message in the conversation:
    - If it has tool_calls → continue to tools node
    - Otherwise → end the conversation

    Args:
        state: Current ReAct state

    Returns:
        "tools" if the agent wants to call tools, "end" otherwise
    """
    messages = state["messages"]

    if not messages:
        return "end"

    last_message = messages[-1]

    # Check if the last message has tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info(f"Agent decided to call tools: {[call['name'] for call in last_message.tool_calls]}")
        return "tools"

    # No tool calls, meaning the agent provided a final answer
    if isinstance(last_message, AIMessage):
        state["output"] = last_message.content
        logger.info("Agent provided final answer, ending ReAct loop")

    return "end"


async def custom_tool_node(state: ReActState) -> ReActState:
    """
    Custom tool execution node that properly maintains message history.

    This node:
    1. Extracts tool calls from the last AI message
    2. Executes each tool
    3. Appends tool results as ToolMessages to the messages list
    4. Extracts retrieved_docs from RAG tool results

    Args:
        state: Current ReAct state with messages containing tool calls

    Returns:
        Updated state with tool results added to messages and retrieved_docs
    """
    messages = state["messages"]

    # Find the last AI message with tool calls
    last_ai_message = None
    tool_calls = []

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            last_ai_message = msg
            tool_calls = msg.tool_calls
            break

    if not tool_calls:
        logger.warning("No tool calls found in messages")
        return state

    # Execute each tool call
    tool_results = []
    tool_map = {
        "llm_chat": llm_chat,
        "llm_query": llm_query,
        "llm_rag": llm_rag
    }

    # Track retrieved documents from RAG
    retrieved_docs = []
    retrieved_answers = state.get("retrieved_answers", 5)

    for tool_call in tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id", "")

        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

        try:
            # Get the tool
            tool = tool_map.get(tool_name)

            if not tool:
                raise ValueError(f"Unknown tool: {tool_name}")

            # Execute the tool using .ainvoke() method for LangChain tools
            result = await tool.ainvoke(tool_args)

            # Special handling for llm_rag to extract retrieved_docs
            if tool_name == "llm_rag":
                try:
                    # Parse JSON response from RAG
                    import json
                    if isinstance(result, str):
                        parsed_result = json.loads(result)
                    else:
                        parsed_result = result

                    # Extract retrieved_docs and summary
                    if isinstance(parsed_result, dict):
                        retrieved_docs = parsed_result.get("retrieved_docs", [])
                        # Update retrieved_answers from args
                        retrieved_answers = tool_args.get("retrieved_answers", 5)
                        logger.info(f"Extracted {len(retrieved_docs)} retrieved documents from RAG")

                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse RAG result as JSON: {e}")
                    retrieved_docs = []

            # Create ToolMessage with the result
            tool_result = ToolMessage(
                content=str(result),
                tool_call_id=tool_id,
                name=tool_name
            )
            tool_results.append(tool_result)

            logger.info(f"Tool {tool_name} executed successfully")

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")

            # Create error ToolMessage
            tool_result = ToolMessage(
                content=f"Error: {str(e)}",
                tool_call_id=tool_id,
                name=tool_name
            )
            tool_results.append(tool_result)

    # Return state with tool results appended to messages
    new_state = state.copy()
    new_state["messages"] = messages + tool_results

    # Add retrieved_docs to state if available
    if retrieved_docs:
        new_state["retrieved_docs"] = retrieved_docs
        new_state["retrieved_answers"] = retrieved_answers
        logger.info(f"Added {len(retrieved_docs)} retrieved documents to state")

    return new_state


def create_react_graph(max_iterations: int = 10):
    """
    Create and compile the ReAct workflow graph.

    The graph structure:
        START → react_agent_node → [conditional] → tools → react_agent_node → ...
                                      ↓
                                     end

    Args:
        max_iterations: Maximum number of ReAct iterations to prevent infinite loops

    Returns:
        Compiled LangGraph StateGraph
    """
    # Create the workflow graph
    workflow = StateGraph(ReActState)
    logging.debug(workflow.state_schema)
    # Add nodes
    workflow.add_node("agent", react_agent_node)
    workflow.add_node("tools", custom_tool_node)  # Use custom tool node instead of ToolNode

    # Set entry point
    workflow.add_edge(START, "agent")

    # Add conditional edge from agent to either tools or end
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )

    # After tools execution, always go back to agent for next iteration
    workflow.add_edge("tools", "agent")

    # Compile the graph
    react_graph = workflow.compile()

    logger.info("ReAct workflow graph created successfully")
    return react_graph


# Create the default ReAct graph instance
react_graph = create_react_graph(max_iterations=10)


# Helper function to run ReAct workflow
async def run_react(input_message: str, max_iterations: int = 10, retrieved_answers: int = 5, session_id: str = None) -> dict:
    """
    Run the ReAct workflow with a user input.

    Args:
        input_message: User's query or request
        max_iterations: Maximum number of ReAct iterations
        retrieved_answers: Number of documents to retrieve when using RAG
        session_id: Optional session ID for loading conversation history

    Returns:
        Dictionary containing:
            - messages: Full conversation history
            - output: Final answer
            - iteration_count: Number of iterations performed
            - retrieved_docs: Retrieved documents with similarity scores
            - retrieved_answers: Count of retrieved answers
    """
    # Load conversation history if session_id is provided
    if session_id:
        messages = load_history_conversation(input_message, session_id)
        logger.info(f"Message {messages} loaded")
        logger.info(f"Loaded {len(messages) - 1} historical messages for session {session_id}")
    else:
        messages = [HumanMessage(content=input_message)]

    initial_state: ReActState = {
        "messages": messages,
        "input": input_message,
        "max_iterations": max_iterations,
        "iteration_count": 0,
        "retrieved_answers": retrieved_answers
    }

    try:
        result = await react_graph.ainvoke(initial_state)

        return {
            "messages": result.get("messages", []),
            "output": result.get("output", ""),
            "iteration_count": result.get("iteration_count", 0),
            "retrieved_docs": result.get("retrieved_docs", []),
            "retrieved_answers": result.get("retrieved_answers", 5),
            "success": True
        }

    except Exception as e:
        logger.error(f"Error running ReAct workflow: {e}")

        return {
            "messages": initial_state["messages"],
            "output": f"An error occurred: {str(e)}",
            "iteration_count": 0,
            "retrieved_docs": [],
            "retrieved_answers": 5,
            "success": False,
            "error": str(e)
        }


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def test_react():
        """Test the ReAct workflow with various queries."""

        test_queries = [
            "你好",  # Simple greeting - should use llm_chat
            "帮我计算 125 + 37的结果再乘以5",  # Calculation - should use llm_query
            "从众心理如何产生的？",  # Knowledge retrieval - should use llm_rag
            "今天北京的天气怎么样？",  # Weather - should use llm_query
        ]

        print("\n" + "="*70)
        print(" "*20 + "ReAct Workflow Test Suite")
        print("="*70)

        for i, query in enumerate(test_queries, 1):
            print(f"\n{'─'*70}")
            print(f"Test {i}/4: {query}")
            print(f"{'─'*70}")

            result = await run_react(query, max_iterations=5)

            # Show iterations
            iterations = result['iteration_count']
            print(f"✓ Completed in {iterations} iteration{'s' if iterations > 1 else ''}")

            # Show final answer
            if result.get('output'):
                print(f"\nFinal Answer:\n{result['output'][:300]}")
                if len(result['output']) > 300:
                    print("...")

            # Show status
            status = "✅ Success" if result['success'] else "❌ Failed"
            print(f"\nStatus: {status}")

            # Show conversation summary
            if result.get("success"):
                messages = result['messages']
                tool_calls_count = sum(1 for m in messages if isinstance(m, AIMessage) and hasattr(m, 'tool_calls') and m.tool_calls)
                tool_results_count = sum(1 for m in messages if isinstance(m, ToolMessage))
                print(f"Conversation: {len(messages)} messages ({tool_calls_count} tool calls, {tool_results_count} tool results)")

        print(f"\n{'='*70}")
        print(" "*25 + "All Tests Completed")
        print("="*70 + "\n")

    # Run tests
    asyncio.run(test_react())
