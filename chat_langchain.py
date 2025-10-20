import getpass
import os

from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage

import env
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

workflow = StateGraph(state_schema=MessagesState)

# add memory in the next step

if not os.getenv("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = env.deepseek_api_key

# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = env.langchain_api_key


client = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
    # other params...
)


def chat_with_llm(state: MessagesState):
    model = client
    system_prompt = (
        "You are a helpful assistant. "
        "Answer all questions to the best of your ability. "
        "The provided chat history includes a summary of the earlier conversation."
    )
    system_message = SystemMessage(content=system_prompt)
    message_history = state["messages"][:-1]  # exclude the most recent user input
    if len(message_history) >= 4:
        last_human_message = state["messages"][-1]
        # Invoke the model to generate conversation summary
        summary_prompt = (
            "Distill the above chat messages into a single summary message. "
            "Include as many specific details as you can."
            "Language of summary must in Chinese."
        )
        summary_message = model.invoke(
            input=message_history + [HumanMessage(content=summary_prompt)]
        )

        # Delete messages that we no longer want to show up
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
        # Re-add user message
        human_message = HumanMessage(content=last_human_message.content)
        # Call the model with summary & response
        response = model.invoke([system_message, summary_message, human_message])
        message_updates = [summary_message, human_message, response] + delete_messages
    else:
        message_updates = model.invoke([system_message] + state["messages"])

    return {"messages": message_updates}
    # ai_msg = model.invoke(messages)
    # print(ai_msg.content)
    # print(model.model_name)


# Define the node and edge
workflow.add_node("model", chat_with_llm)
workflow.add_edge(START, "model")

# Add simple in-memory checkpointer
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    demo_ephemeral_chat_history = [
        HumanMessage(content="Hey there! I'm Nemo."),
        AIMessage(content="Hello!"),
        HumanMessage(content="How are you today?"),
        AIMessage(content="Fine thanks!"),
    ]
    result = app.invoke(
        {
            "messages": demo_ephemeral_chat_history
                        + [HumanMessage("What did I say my name was?")]
        },
        config={"configurable": {"thread_id": "4"}},
    )
    print(result['messages'])
    # question = input("Enter a sentence in English: ")
    # chat_with_llm(question)

