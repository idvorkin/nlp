from setuptools import setup

setup(
    name="gpt",
    version="0.1.0",
    py_modules=["gpt3", "igor_journal", "dump_grateful"],
    install_requires=[],
    entry_points="""
            [console_scripts]
            gpt=gpt3:app_with_loguru
            gpt3=gpt3:app_with_loguru
            ij=igor_journal:app_with_loguru
            dg=dump_grateful:app_with_loguru
        """,
)
