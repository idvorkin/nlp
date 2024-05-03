#!python3
from langchain.prompts import ChatPromptTemplate
from langchain_core import messages


def promptfoo_markdown_test(context: dict) -> str:
    doc1 = """
    # Heading 1
* A point
* B
# Heading 2
    * 2 point
    * 2.2
    """

    doc2 = """
    # Heading 1
* A point d2
* B d2
# Heading 2
    * 2 point d2
    * 2.2 d2
    """
    prompt = prompt_merge_documents([doc1, doc2]).messages

    to_str = "\n----\n".join([m.content for m in prompt])  # type:ignore
    return to_str

    # super annoying, claude can't handle this pattern, must get translsated in LC somewhere.
    # PITA< keeping crappy merged format
    # # cdef to_dict(m):
    # role = "assistant" if isinstance(m, messages.SystemMessage) else "user"
    # return {"role":role , "content": m.content}

    # return [to_dict(m) for m in prompt] #type:ignore


def wrap_document(doc):
    return f"<DOCUMENT>\n {doc}\n</DOCUMENT>\n"


def prompt_merge_documents(documents):
    instructions = """
    You are a brilliant AI.
User will pass multiple markdown documents into a single document. You should

1. Read all documents
2. Identify and compare headings in each document.
3. If a heading exists in one document and not in others, include it in the final merged document.
4. If two headings are very similar, merge these headings into one.
5. Under each merged heading, include the content from the corresponding headings of all documents, separated by a clear delimiter or sub-heading to indicate the source document.
6. Ensure the final document is well-formatted and only contains the final markdown

Please ensure the script handles different levels of headings appropriately and preserves the integrity of the markdown formatting (e.g., lists, code blocks, links).


"""

    merge_documents = "".join([wrap_document(doc) for doc in documents])
    return ChatPromptTemplate.from_messages(
        [
            messages.SystemMessage(content=instructions),
            messages.HumanMessage(content=merge_documents),
        ]
    )


if __name__ == "__main__":
    context = {}
    out = promptfoo_markdown_test(context)
    print(out)
