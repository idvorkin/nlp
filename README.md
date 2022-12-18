# My explorations in NLP

From my [blog](https://idvork.in/nlp):

My explorations of NLP, mostly using my corpus of journal entries and other writing. My intent is two fold 1) learning about NLP and sentiment analysis 2) finding latent meaning in my writing, ideally to help me better understand my own psychological processes. I’ve had much more success with the former then the latter

## ClI Tools

OK, the tools here use typer, which to do completion needs to be built as a package. See setup.py

I don't really understand zsh completion, so here's some of the stuff I've been doing:

During active development just run it diretly.
Manually install the package - yuk doesn't take effect till reinstalling.

    pip3 install .

Install completions for the tools

    ij install-completion
    gpt install-completion

Directly source the completors:

    source ~/.zfunc/_gpt
    source ~/.zfunc/_ij

Notes on zsh completors:

<https://github.com/zsh-users/zsh-completions/blob/master/zsh-completions-howto.org>
