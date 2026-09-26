"""
Gradio app - the local version of 5_gradio.ipynb, grown up a bit.

    python app.py            ->  http://127.0.0.1:7860
    python app.py --share    ->  also a temporary public link you can send someone
    python app.py --plain    ->  no question rewriting, the way the course notebook does it
    python app.py --port 7870 -> a specific port, if 7860 is busy

What it adds over the notebook:

1. The answer streams in word by word instead of appearing after a silent wait.
2. A sources panel shows the real book titles, page numbers and match scores, so
   any claim can be checked in seconds.
3. A fresh engine is built for every message, filled with that browser's own
   history. The notebook keeps ONE bot with ONE memory, so two people using the
   same link share a conversation - user A's secrets can leak to user B. Engines
   are cheap (0.05 ms) because the index and the LLM are shared.
4. No private attributes. The notebook poked rag_bot._retriever._similarity_top_k
   to change top-k; here the slider value goes straight into as_retriever().
"""

import os
import sys
import time

import gradio as gr
from dotenv import load_dotenv
from llama_index.core.base.llms.types import ChatMessage

import rag
import rag_chat

# The Gradio pieces used below:
#
#   gr.ChatInterface     a ready-made chat page: message box, send button, the
#                        history display, the example buttons
#   fn(message, history, *additional_inputs)
#                        Gradio calls this once per message. If it is a GENERATOR,
#                        every yield updates the page - that is how streaming works
#   additional_inputs    extra widgets whose values are passed to fn, in order
#   additional_outputs   extra components that fn also writes to; the function
#                        then yields a tuple, one item per output
#   .launch()            starts a web server on 127.0.0.1:7860

CONDENSE = True   # overridden at startup by --plain

# File names are ugly in a citation. This is what the reader should see instead.
TITLES = {
    "speech_and_language_processing": ("Speech and Language Processing", "Jurafsky & Martin"),
    "understanding_deep_learning": ("Understanding Deep Learning", "Simon Prince"),
    "foundations_of_large_language_models": ("Foundations of Large Language Models", "Xiao & Zhu"),
    "vision_transformer_vit": ("An Image is Worth 16x16 Words (ViT)", "Dosovitskiy et al."),
    "clip_text_image_pairs": ("Learning Transferable Visual Models (CLIP)", "Radford et al."),
    "llava_visual_instruction_tuning": ("Visual Instruction Tuning (LLaVA)", "Liu et al."),
    "flamingo_visual_language_model": ("Flamingo", "Alayrac et al."),
    "ddpm_denoising_diffusion": ("Denoising Diffusion Probabilistic Models", "Ho et al."),
    "latent_diffusion_stable_diffusion": ("Latent Diffusion Models", "Rombach et al."),
    "whisper_speech_recognition": ("Robust Speech Recognition (Whisper)", "Radford et al."),
    "wav2vec2_audio_representations": ("wav2vec 2.0", "Baevski et al."),
    "ethical_and_social_risks": ("Ethical and Social Risks of Harm from LMs", "Weidinger et al."),
}

# One list per example: [question, top_k, temperature] - a value for the
# message and for every widget in additional_inputs, in the same order.
EXAMPLES = [
    ["What is self-attention and what problem does it solve?", 8, 0.1],
    ["How does CLIP connect images and text?", 8, 0.1],
    ["How does a diffusion model generate an image?", 8, 0.1],
    ["What are the main risks of harm from language models?", 8, 0.1],
    ["What was the final score of the 2022 World Cup final?", 8, 0.1],
]
EXAMPLE_LABELS = ["Self-attention", "CLIP", "Diffusion", "Risks of harm", "Out of scope"]


def text_of(message):
    """Gradio may hand content back as a string or as a list of parts."""
    content = message["content"]
    if isinstance(content, str):
        return content
    return " ".join(part.get("text", "") for part in content)


def pretty(hit):
    """One retrieved chunk as a citation: title, author, page, match strength."""
    stem = hit.node.metadata.get("file_name", "?").replace(".pdf", "")
    title, author = TITLES.get(stem, (stem, ""))
    page = hit.node.metadata.get("page_label")
    filled = max(0, min(10, round(hit.score * 10)))
    bar = "█" * filled + "░" * (10 - filled)
    parts = [p for p in (author, f"p. {page}" if page else "") if p]
    return f"**{title}**  \n{' · '.join(parts)}  \n`{bar}` {hit.score:.2f}"


def sources_panel(hits, seconds=None):
    """The right-hand panel: where this answer came from."""
    if not hits:
        return "### Sources\n_Nothing was retrieved._"
    docs = len({h.node.metadata.get("file_name") for h in hits})
    head = f"### Sources\n_{len(hits)} passages from {docs} document(s)_\n\n"
    tail = f"\n\n---\n_answered in {seconds:.1f} s_" if seconds else ""
    return head + "\n\n".join(pretty(h) for h in hits) + tail


def respond(message, history, top_k, temperature):
    """One question. A generator, so the answer appears word by word.

    The extra arguments arrive in the order the widgets are listed in
    additional_inputs. Each yield is a tuple: (what the chat shows, what the
    sources panel shows). Retrieval finishes before the first token arrives, so
    the citations are on screen while the answer is still being written.
    """
    started = time.perf_counter()
    past = [ChatMessage(role=m["role"], content=text_of(m)) for m in history]
    # A new LLM per message, because temperature is set when it is created.
    # It costs about a millisecond - nothing is downloaded or connected here.
    llm = rag_chat.get_llm(temperature=float(temperature))
    bot = rag_chat.build_bot(INDEX, llm, top_k=int(top_k), history=past, condense=CONDENSE)

    stream = bot.stream_chat(message)
    panel = sources_panel(stream.source_nodes)

    answer = ""
    for token in stream.response_gen:
        answer += token
        yield answer, panel
    yield answer, sources_panel(stream.source_nodes, time.perf_counter() - started)


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
    n_docs = len([f for f in os.listdir(rag.DATA_DIR) if f.endswith(".pdf")])

    # Chat on the left, citations on the right. gr.Blocks is what makes a layout
    # possible at all - a ChatInterface on its own is one column and nothing else.
    #
    # The sources panel is created with render=False and rendered further down,
    # inside its own column. This matters: a component that is never rendered can
    # still be passed to additional_outputs, and the page then streams into
    # something that does not exist on screen. The browser shows a red "Error"
    # box with no explanation, which is exactly what happened the first time.
    with gr.Blocks(title="Generative AI tutor", fill_height=True) as demo:
        gr.Markdown(
            f"# Generative AI tutor\n"
            f"Ask about transformers, embeddings, LLMs, diffusion, vision or audio "
            f"models. Answers come only from **{n_docs} textbooks and papers**, and "
            f"every one is cited with its page number. If the library does not cover "
            f"it, the honest answer is \"I don't know\"."
        )

        sources = gr.Markdown(
            "### Sources\n_Ask something and the passages behind the answer appear "
            "here, with their page numbers._",
            render=False,
        )

        with gr.Row():
            with gr.Column(scale=3):
                gr.ChatInterface(
                    fn=respond,
                    # buttons= is Gradio 6's name for the per-message actions;
                    # the old show_copy_button argument no longer exists.
                    chatbot=gr.Chatbot(height=460, label="Conversation", resizable=True,
                                       buttons=["copy", "retry"], latex_delimiters=[
                                           {"left": "$$", "right": "$$", "display": True},
                                           {"left": r"\[", "right": r"\]", "display": True},
                                           {"left": r"\(", "right": r"\)", "display": False},
                                           {"left": "$", "right": "$", "display": False}]),
                    examples=EXAMPLES,
                    example_labels=EXAMPLE_LABELS,
                    additional_inputs=[
                        gr.Slider(1, 16, value=rag.TOP_K, step=1,
                                  label="Passages retrieved (top-k)",
                                  info="How much of the library the model may read for "
                                       "one question. Too few and it answers \"I don't "
                                       "know\"; too many and the useful passage drowns "
                                       "in noise."),
                        gr.Slider(0.0, 1.0, value=rag.TEMPERATURE, step=0.1,
                                  label="Temperature",
                                  info="How the next word is picked from the model's "
                                       "probabilities. 0 always takes the likeliest word, "
                                       "so answers repeat exactly; 1 wanders and invents "
                                       "more. Keep it low for a reference tool."),
                    ],
                    additional_inputs_accordion=gr.Accordion("Settings", open=False),
                    additional_outputs=[sources],
                )
            with gr.Column(scale=1, min_width=280):
                sources.render()

    # Gradio picks 7860, or the next free port above it if that one is taken -
    # usually by an earlier run of this app that is still alive. --port pins it.
    port = None
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])

    demo.launch(share="--share" in sys.argv, server_port=port, theme="soft")
