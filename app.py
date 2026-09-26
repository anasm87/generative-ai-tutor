"""
Gradio app - the local version of 5_gradio.ipynb.

Same bot as rag_chat.py, wrapped in a web page instead of the terminal:

    python app.py            ->  http://127.0.0.1:7860
    python app.py --share    ->  also a temporary public link you can send someone
    python app.py --plain    ->  no question rewriting, the way the course notebook does it
    python app.py --port 7870 -> a specific port, if 7860 is busy

Two differences from the notebook, both deliberate:

1. A fresh engine is built for every message, filled with that browser's own
   history. The notebook keeps ONE bot with ONE memory, so two people using the
   same link share a conversation - user A's secrets can leak to user B. The
   notebook fixes this at the end by resetting the bot's memory; building a new
   engine per request is the same idea with nothing left to forget. Engines are
   cheap: no model is loaded here, the index and the LLM are shared.

2. No private attributes. The notebook pokes rag_bot._retriever._similarity_top_k
   to change top-k; here the slider value is simply passed to as_retriever().
"""

import os
import sys

import gradio as gr
from dotenv import load_dotenv
from llama_index.core.base.llms.types import ChatMessage

import rag
import rag_chat

# The Gradio pieces used below:
#
#   gr.ChatInterface   a ready-made chat page: message box, send button, the
#                      history display, the example buttons. You supply one
#                      function and it builds the rest
#   fn(message, history, *extras)
#                      the contract for that function. Gradio calls it with the
#                      new message, this browser's conversation, and whatever
#                      extra widgets you declared. Whatever string it returns is
#                      shown as the bot's reply
#   additional_inputs  extra widgets (here a slider) whose current values are
#                      passed to fn as further arguments, in order
#   .launch()          starts a small web server on 127.0.0.1:7860.
#                      share=True also opens a temporary public link

CONDENSE = True   # overridden at startup by --plain

# With additional_inputs, Gradio wants one list per example: [question, top_k].
EXAMPLES = [
    ["What is self-attention and what problem does it solve?", 8],
    ["How does CLIP connect images and text?", 8],
    ["How does a diffusion model generate an image?", 8],
    ["How does retrieval-augmented generation reduce hallucination?", 8],
]


def text_of(message):
    """Gradio 6 may hand content back as a string or as a list of parts."""
    content = message["content"]
    if isinstance(content, str):
        return content
    return " ".join(part.get("text", "") for part in content)


def respond(message, history, top_k):
    """One question. Gradio calls this once per message the user sends.

    message   what was just typed
    history   this browser's conversation so far, as [{"role":..., "content":...}]
    top_k     the slider's current value

    The history arrives from the browser, so it is per-visitor by definition.
    Turning it into ChatMessages and handing it to a brand-new engine is what
    keeps two users from sharing one memory.
    """
    past = [ChatMessage(role=m["role"], content=text_of(m)) for m in history]
    # CONDENSE=True (unless started with --plain) rewrites the question using
    # the conversation before searching, which is what makes follow-ups like
    # "what about the other one?" retrieve anything useful.
    bot = rag_chat.build_bot(INDEX, LLM, top_k=int(top_k), history=past, condense=CONDENSE)

    answer = bot.chat(message)
    sources = ", ".join(
        f"{hit.node.metadata.get('file_name', '?')} ({hit.score:.2f})"
        for hit in answer.source_nodes
    )
    # Showing the sources under every answer is the quickest way to spot the
    # usual failure: a confident answer built from the wrong chunks.
    return f"{answer.response.strip()}\n\n*from: {sources}*"


if __name__ == "__main__":
    # utf-8-sig, because PowerShell writes a BOM that would otherwise turn the
    # first key name into '<BOM>GROQ_API_KEY' and it would silently not be found.
    load_dotenv(os.path.join(rag.HERE, ".env"), encoding="utf-8-sig")
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("no GROQ_API_KEY - put it in generative_ai/.env (free key: console.groq.com)")

    # Loaded once, at startup, and shared by every request: the vector database
    # and the LLM connection. These are the expensive things. The chat engine
    # built per message on top of them costs nothing.
    INDEX = rag.load_or_build_index(rag.build_embeddings())
    LLM = rag_chat.get_llm()
    CONDENSE = "--plain" not in sys.argv

    demo = gr.ChatInterface(
        fn=respond,
        title="Generative AI tutor",
        description="Ask me about transformers, embeddings, LLMs or generative models. "
                    "I answer only from three open-access textbooks "
                    "(Jurafsky & Martin, Prince, Xiao & Zhu) and show you the source of every answer.",
        examples=EXAMPLES,
        additional_inputs=[
            gr.Slider(1, 16, value=rag.TOP_K, step=1, label="Chunks retrieved (top-k)",
                      info="How many pieces of the documents are given to the model."),
        ],
    )
    # Gradio picks 7860, or the next free port above it if that one is taken -
    # usually by an earlier run of this app that is still alive. --port pins it.
    port = None
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])

    demo.launch(share="--share" in sys.argv, server_port=port)
