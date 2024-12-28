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

### Code Style

Prefer returning from a function vs nesting ifs.
Prefer descriptive variable names over comments.
Avoid nesting ifs, return from functions as soon as you can

### Types

Use types whenever possible.
Use the latest syntax for types (3.12)
Don't use tuples, define pydantic types for return values. Call Them FunctionReturn for the function name
<examples>
For a Single Item Return
Function: get_user_profile()
Return Type: UserProfileResponse
For Multiple Items
Function: list_users()
Return Type: UserListResponse or PaginatedUsersResponse
For Aggregated Data
Function: get_sales_summary()
Return Type: SalesSummaryResult
For Nested or Composite Data
Function: get_order_details()
Return Type: OrderDetailsResponse (which may include nested models like ProductInfo or ShippingDetails).
</examples>

### Testing

When possible update the tests to reflect the new changes.
Tests are in the test directory

### Adding Libraries

When adding a new library, ensure it's in pyproject.toml

### Adding New scripts

When adding new scripts, be sure to add them to the scripts section in pyproject.toml

### When running python,

use python3

### Tracking specification changes

Lets add enduser_specs.md to track specifications

Every time you make changes to expected user behavior, update the specs

If the user changes the specifications change the specifications.
This will be mostly tracking business logic
When the specs aren't clear ask user to clarify

