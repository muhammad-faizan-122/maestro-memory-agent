from langchain_core.messages import HumanMessage, SystemMessage
from trustcall import create_extractor
from langchain_google_genai import ChatGoogleGenerativeAI
from .utils import get_trustcall_message
from dotenv import load_dotenv
from . import states
from . import prompts


load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
profile_extractor = create_extractor(
    model,
    tools=[states.Profile],
    tool_choice="Profile",
)


def find_memory_type(messages_history: list, memories: dict):
    try:
        # prepare system message
        system_msg = prompts.MODEL_SYSTEM_MESSAGE.format(
            user_profile=memories["user_profile"],
            todo=memories["todo"],
            instructions=memories["instructions"],
        )
        messages = [SystemMessage(content=system_msg)] + messages_history

        # Respond using memory as well as the chat history
        response = model.bind_tools(
            [states.UpdateMemory],
            parallel_tool_calls=False,
        ).invoke(messages)
    except Exception as e:
        print(f"Error generating response: {e}")
        response = None
    return response


def generate_updated_todo_instructions(messages_history, existing_memory):
    # Format the memory in the system prompt
    system_msg = prompts.CREATE_INSTRUCTIONS.format(
        current_instructions=existing_memory.value if existing_memory else None
    )
    new_memory = model.invoke(
        [SystemMessage(content=system_msg)]
        + messages_history
        + [
            HumanMessage(
                content="Please update the instructions based on the conversation"
            )
        ]
    )
    return new_memory


def generate_updated_todos(messages_history, tool_name, spy, existing_memories):
    """Merge the chat history and the instruction"""
    try:
        message = get_trustcall_message(messages_history)
        # Create the Trustcall extractor for updating the ToDo list
        todo_extractor = create_extractor(
            model,
            tools=[states.ToDo],
            tool_choice=tool_name,
            enable_inserts=True,
        ).with_listeners(on_end=spy)

        # Invoke the extractor
        result = todo_extractor.invoke(
            {"messages": message, "existing": existing_memories}
        )
    except Exception as e:
        print(f"Error generating updated todos: {e}")
        result = None
    return result


def generate_updated_profile(message_history, existing_memories):
    try:
        message = get_trustcall_message(message_history)
        # Invoke the extractor
        result = profile_extractor.invoke(
            {"messages": message, "existing": existing_memories}
        )
        return result
    except Exception as e:
        print(f"Error generating updated profile: {e}")
        return None
