# Coding conventions used in this project

For CLIs, use a Typer app.
Use `ic` for logging.
Use Rich for pretty printing.
Use Loguru for logging.
Use Typer for CLI apps.
Use Pydantic for data validation.
Use types; when using types, prefer using built-ins like `foo | None` vs `foo: Optional[str]`.
When using Typer, use the latest syntax for arguments and options.

```python
    name: Annotated[Optional[str], typer.Argument()] = None
    def main(name: Annotated[str, typer.Argument()] = "Wade Wilson"):
    lastname: Annotated[str, typer.Option(help="Last name of person to greet.")] = "",
    formal: Annotated[bool, typer.Option(help="Say hi formally.")] = False,
```

Prefer returning from a function vs nesting ifs.
Prefer descriptive variable names over comments.
Avoid nesting ifs, return from functions as soon as you can
