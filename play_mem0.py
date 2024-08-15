#!python3


import typer

from loguru import logger
from icecream import ic
from mem0 import Memory
from rich.console import Console

app = typer.Typer()
console = Console()

@app.command()
def start():
    client =  Memory() # 1. Add: Store a memory from any unstructured text
    result = client.add("I am working on improving my tennis skills. Suggest some online courses.", user_id="alice", metadata={"category": "hobbies"})
    ic(result)
    from mem0 import MemoryClient
    memoryclient = MemoryClient()

    messages = [
            {"role": "user", "content": "Hi, I'm Alex. I'm a vegetarian and I'm allergic to nuts."},
            {"role": "assistant", "content": "Hello Alex! I've noted that you're a vegetarian and have a nut allergy. I'll keep this in mind for any food-related recommendations or discussions."}
            ]
    memoryclient.add(messages, user_id="igor")

    # Created memory --> 'Improving her tennis skills.' and 'Looking for online suggestions.'
    # 2 . Update: update the memory
    #re sult = m.update(memory_id=<memory_id_1>, data="Likes to play tennis on weekends")
    ic(result)

    # Updated memory --> 'Likes to play tennis on weekends.' and 'Looking for online suggestions.'
    # 3. Search: search related memories
    related_memories = m.search(query="What are Alice's hobbies?", user_id="alice")
    ic(related_memories)

    # Retrieved memory --> 'Likes to play tennis on weekends'
    # 4. Get all memories
    all_memories = m.get_all()
    memory_id = all_memories[0]["id"] # get a memory_id

# All memory items --> 'Likes to play tennis on weekends.' and 'Looking for online suggestions.'

@logger.catch()
def app_wrap_loguru():
    app()

if __name__ == "__main__":
    app_wrap_loguru()
