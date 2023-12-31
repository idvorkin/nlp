from setuptools import setup

setup(
    name="idvorkin_nlp",
    version="0.1.0",
    py_modules=[
        "gpt3",
        "life",
        "igor_journal",
        "dump_grateful",
        "improv",
        "openai_wrapper",
        "tts",
        "pbf",
    ],
    install_requires=[],
    entry_points="""
            [console_scripts]
            gpt=gpt3:app
            gpt3=gpt3:app
            ij=igor_journal:app
            dg=dump_grateful:app
            improv=improv:app
            life=life:app
            tts=tts:app
            pbf=pbf:app
        """,
)
