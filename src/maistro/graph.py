from . import llm_utils
from . import utils
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph, MessagesState, END, START
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from typing import Literal


def task_mAIstro(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Load memories from the store and use them to personalize the chatbot's response."""
    memories = utils.retrieve_all_memories(
        store, user_id=config["configurable"]["user_id"]
    )
    response = llm_utils.find_memory_type(state["messages"], memories)
    return {"messages": [response]}


def update_todos(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and update the memory collection."""
    tool_name = "ToDo"

    # retrieve existing ToDo items
    existing_items = utils.retrieve_todo(store, config["configurable"]["user_id"])

    # Format the existing memories for the Trustcall extractor
    existing_memories = utils.format_existing_memories(existing_items, tool_name)

    # Initialize the spy for visibility into the tool calls made by Trustcall
    spy = utils.Spy()

    # Generate updated ToDos using Trustcall
    result = llm_utils.generate_updated_todos(
        state["messages"][:-1],
        tool_name,
        spy,
        existing_memories,
    )

    # Save the updated ToDos back to the store
    utils.save_memories(
        store,
        result,
        namespace=("todo", config["configurable"]["user_id"]),
    )

    # Respond to the tool call made in task_mAIstro, confirming the update
    tool_calls = state["messages"][-1].tool_calls

    # Extract the changes made by Trustcall and add the the ToolMessage returned to task_mAIstro
    todo_update_msg = utils.extract_tool_info(spy.called_tools, tool_name)
    return {
        "messages": [
            {
                "role": "tool",
                "content": todo_update_msg,
                "tool_call_id": tool_calls[0]["id"],
            }
        ]
    }


def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and update the memory collection."""
    tool_name = "Profile"

    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Define the namespace for the memories
    namespace = ("profile", user_id)

    # retrieve existing profile items
    existing_items = utils.retrieve_user_profile(store, user_id, namespace)

    # Format the existing memories for the Trustcall extractor
    existing_memories = utils.format_existing_memories(existing_items, tool_name)

    # Generate updated profile using Trustcall
    result = llm_utils.generate_updated_profile(
        state["messages"][:-1], existing_memories
    )

    # Save the memories from Trustcall to the store
    utils.save_memories(store, result, namespace)

    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            {
                "role": "tool",
                "content": "updated profile",
                "tool_call_id": tool_calls[0]["id"],
            }
        ]
    }


def update_instructions(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and update the memory collection."""

    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Define the namespace for the memories
    namespace = ("instructions", user_id)

    # retrieve existing instruction memory
    existing_memory = utils.retrieve_instructions(store, user_id)

    # generate updated instructions
    new_todo_instruction = llm_utils.generate_updated_todo_instructions(
        state["messages"][:-1],
        existing_memory,
    )

    # Overwrite the existing memory in the store
    utils.overwrite_existing_memory(
        store,
        namespace,
        new_todo_instruction,
        key="user_instructions",
    )
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            {
                "role": "tool",
                "content": "updated instructions",
                "tool_call_id": tool_calls[0]["id"],
            }
        ]
    }


# Conditional edge
def route_message(
    state: MessagesState,
    config: RunnableConfig,
    store: BaseStore,
) -> Literal[END, "update_todos", "update_instructions", "update_profile"]:
    """Reflect on the memories and chat history to decide whether to update the memory collection."""
    message = state["messages"][-1]
    if len(message.tool_calls) == 0:
        return END
    else:
        tool_call = message.tool_calls[0]
        if tool_call["args"]["update_type"] == "user":
            return "update_profile"
        elif tool_call["args"]["update_type"] == "todo":
            return "update_todos"
        elif tool_call["args"]["update_type"] == "instructions":
            return "update_instructions"
        else:
            raise ValueError


def build_graph():
    # Create the graph + all nodes
    builder = StateGraph(MessagesState)

    # Define the flow of the memory extraction process
    builder.add_node(task_mAIstro)
    builder.add_node(update_todos)
    builder.add_node(update_profile)
    builder.add_node(update_instructions)

    builder.add_edge(START, "task_mAIstro")
    builder.add_conditional_edges("task_mAIstro", route_message)
    builder.add_edge("update_todos", "task_mAIstro")
    builder.add_edge("update_profile", "task_mAIstro")
    builder.add_edge("update_instructions", "task_mAIstro")

    # Store for long-term (across-thread) memory
    across_thread_memory = InMemoryStore()

    # Checkpointer for short-term (within-thread) memory
    within_thread_memory = MemorySaver()

    # We compile the graph with the checkpointer and store
    graph = builder.compile(
        checkpointer=within_thread_memory,
        store=across_thread_memory,
    )
    return graph
