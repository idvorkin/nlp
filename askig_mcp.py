from fastmcp import FastMCP
import askig_logic
import typer # Added for CLI
from typing_extensions import Annotated # Added for Typer

# from askig_models import BlogPlacementSuggestion # Import if type hints for return need it

# It is good practice to configure logging for the server.
# from loguru import logger
# import sys
# logger.remove()
# logger.add(sys.stderr, level="INFO") # Adjust level as needed

# Renamed mcp_app to mcp_instance for clarity with typer app
mcp_instance = FastMCP(name="AskIG_MCP_Server", title="AskIG MCP Server", description="MCP server for AskIG tools.") 

@mcp_instance.tool()
async def ask_blog(question: str, facts: int = 20, debug: bool = False, model: str = "openai") -> str:
    """Answers questions about Igor's blog using available context."""
    # logger.info(f"Received ask_blog request: question='{question}', facts={facts}, model='{model}')
    response_str = await askig_logic.iask_logic(question, facts=facts, debug=debug, model=model)
    return response_str

@mcp_instance.tool()
async def suggest_blog_placement(topic: str, num_docs: int = 20, debug: bool = False, model: str = "openai") -> str:
    """Suggests where to add new blog content about a topic based on existing content and structure."""
    # logger.info(f"Received suggest_blog_placement request: topic='{topic}', num_docs={num_docs}, model='{model}')
    response_str = await askig_logic.iask_where_logic(topic, num_docs=num_docs, debug=debug, model=model)
    return response_str

# Typer application
cli_app = typer.Typer(no_args_is_help=True, help="AskIG MCP Server CLI - Run with stdio or http (SSE) transport.")

@cli_app.command(name="stdio", help="Run the MCP server with STDIO transport (default).")
def run_stdio():
    """
    Starts the MCP server using the STDIO transport.
    This is often used for local clients or specific MCP integrations.
    """
    print("Starting AskIG MCP Server with STDIO transport...")
    try:
        mcp_instance.run(transport="stdio") # Explicitly specify stdio
    except Exception as e:
        print(f"Error trying to run server with STDIO transport: {e}")
        print("Ensure FastMCP is correctly installed.")

@cli_app.command(name="http", help="Run the MCP server with HTTP (SSE) transport.")
def run_http(
    host: Annotated[str, typer.Option(help="Host to bind the server to.")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="Port to bind the server to.")] = 8000,
):
    """
    Starts the MCP server using HTTP (specifically SSE - Server-Sent Events) transport.
    """
    print(f"Starting AskIG MCP Server with HTTP (SSE) transport on {host}:{port}...")
    try:
        # FastMCP uses 'sse' for its HTTP-based transport in examples
        mcp_instance.run(transport="sse", host=host, port=port)
    except Exception as e:
        print(f"Error trying to run server with HTTP (SSE) transport: {e}")
        print("Ensure FastMCP and its dependencies (like uvicorn for sse) are installed.")
        print("Try: uv pip install 'fastmcp[sse]' or 'uvicorn'")


if __name__ == "__main__":
    # The mcp_instance.run() calls are now within the Typer command functions.
    # The __main__ block just needs to invoke the Typer app.
    cli_app() 