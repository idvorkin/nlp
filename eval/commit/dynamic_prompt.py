import sys
import os

# Add the parent directory to the path so we can import commit.py
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


def get_commit_prompt(context):
    """
    Dynamic prompt function that reuses the actual prompt from commit.py
    """
    vars_dict = context.get("vars", {})
    oneline = vars_dict.get("oneline", False)

    # Import here to avoid issues with dependencies
    from commit import prompt_summarize_diff

    # Get the actual prompt template from commit.py
    prompt_template = prompt_summarize_diff(
        "", oneline
    )  # Pass empty string as placeholder

    # Extract the system message instructions
    system_message = None
    for message in prompt_template.messages:
        if hasattr(message, "content") and hasattr(message, "type"):
            if message.type == "system":
                system_message = message.content
                break

    # Fallback if we can't extract system message
    if not system_message:
        system_message = "You are an expert programmer. Write a commit message for the following git diff."

    # Return as simple chat format
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": vars_dict.get("diff_content", "")},
    ]
