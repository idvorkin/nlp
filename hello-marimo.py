#!uv run
# /// script
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    # Default Entrypoint - this is pretty nice!
    return


@app.cell
def _():
    print("hello world")
    return


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# My man what's going on""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
