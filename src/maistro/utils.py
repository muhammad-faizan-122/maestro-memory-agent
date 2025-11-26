from langgraph.store.base import BaseStore
from datetime import datetime
from langchain_core.messages import merge_message_runs, SystemMessage
from src.maistro import prompts
import uuid
from langchain_core.messages import AIMessage


# Inspect the tool calls made by Trustcall
class Spy:
    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        # Collect information about the tool calls made by the extractor.
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                self.called_tools.append(
                    r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                )


def extract_tool_info(tool_calls, schema_name="Memory"):
    """Extract information from tool calls for both patches and new memories.

    Args:
        tool_calls: List of tool calls from the model
        schema_name: Name of the schema tool (e.g., "Memory", "ToDo", "Profile")
    """

    # Initialize list of changes
    changes = []

    for call_group in tool_calls:
        for call in call_group:
            args = call.get("args", None)

            if call["name"] == "PatchDoc":

                json_doc_id = args.get("json_doc_id", "")
                planned_edits = args.get("planned_edits", "")
                patches = args.get("patches")
                print("type of patches: ", type(patches), patches)
                if isinstance(patches, list):
                    value = patches[0]["value"]
                elif isinstance(patches, dict):
                    value = patches["value"]
                else:
                    ValueError(f"wrong type of patches: {type(patches)}\n{patches}")
                changes.append(
                    {
                        "type": "update",
                        "doc_id": json_doc_id,
                        "planned_edits": planned_edits,
                        "value": value,
                    }
                )
            elif call["name"] == schema_name:
                changes.append({"type": "new", "value": args})

    # Format results as a single string
    result_parts = []
    for change in changes:
        if change["type"] == "update":
            result_parts.append(
                f"Document {change['doc_id']} updated:\n"
                f"Plan: {change['planned_edits']}\n"
                f"Added content: {change['value']}"
            )
        else:
            result_parts.append(
                f"New {schema_name} created:\n" f"Content: {change['value']}"
            )

    return "\n\n".join(result_parts)


def get_trustcall_message(messages_history):
    TRUSTCALL_INSTRUCTION_FORMATTED = prompts.TRUSTCALL_INSTRUCTION.format(
        time=datetime.now().isoformat()
    )
    trustcall_message = list(
        merge_message_runs(
            messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)]
            + messages_history
        )
    )
    return trustcall_message


def retrieve_user_profile(store: BaseStore, user_id, namespace=None):
    """Retrieve profile memory from the store"""
    try:
        namespace = ("profile", user_id) if not namespace else namespace
        memories = store.search(namespace)
        user_profile = memories[0].value if memories else None
        return user_profile
    except Exception as e:
        print(f"Error retrieving user profile: {e}")
        return None


def retrieve_todo(store: BaseStore, user_id):
    """Retrieve todo memory from the store"""
    try:
        namespace = ("todo", user_id)
        print("namespace todo: ", namespace)
        memories = store.search(namespace)
        todo = "\n".join(f"{mem.value}" for mem in memories)
        print("todo: ")
        return todo
    except Exception as e:
        print(f"Error retrieving todo: {e}")
        return None


def retrieve_instructions(store: BaseStore, user_id):
    """Retrieve instructions memory from the store"""
    try:
        namespace = ("instructions", user_id)
        memories = store.search(namespace)
        instructions = memories[0].value if memories else None
        return instructions
    except Exception as e:
        print(f"Error retrieving instructions: {e}")
        return None


def retrieve_all_memories(store: BaseStore, user_id):
    """Retrieve all memories for a given user ID"""
    return {
        "user_profile": retrieve_user_profile(store, user_id),
        "todo": retrieve_todo(store, user_id),
        "instructions": retrieve_instructions(store, user_id),
    }


def save_memories(store: BaseStore, result, namespace):
    """Save the memories from Trustcall to the store"""
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(
            namespace,
            rmeta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json"),
        )


def format_existing_memories(existing_items, tool_name):
    try:
        memories = []
        if existing_items and isinstance(existing_items, list):
            for existing_item in existing_items:
                if not isinstance(existing_item, str):
                    memories.append((existing_item.key, tool_name, existing_item.value))
        return memories
    except Exception as e:
        msg = f"Failed to format the existing memories due to {e}"
        raise ValueError(msg)


def overwrite_existing_memory(store: BaseStore, namespace, new_memory, key):
    """Overwrite the existing memory in the store"""

    try:
        if isinstance(new_memory, AIMessage):
            new_memory = new_memory.content
        store.put(namespace, key, {"memory": new_memory})
    except Exception as e:
        print(f"Error overwriting existing memory: {e}")
        raise e
