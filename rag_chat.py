"""
RAG with memory - the local version of 4_rag_chatbot.ipynb.

rag.py answered one question at a time and forgot it immediately. Here the same
vector database gets a memory, so follow-up questions work:

    You: Who is the queen?
    Bot: The Queen of Hearts, known for shouting "Off with their heads!"
    You: What does she like to do?          <- "she" only means something
    Bot: Ordering executions.                  because the last turn is remembered

Every turn sends: the system messages + the chunks retrieved for this question
+ the conversation so far. The retrieval is redone on every turn.

By default every question is first rewritten into a standalone one using the
conversation ("What does she like to do?" -> "What does the Queen of Hearts like
to do?") and only then used to search. Measured on this book, that is the
difference between an answer and a shrug:

    plain     chunks scored 0.277 0.248 0.244 0.240  -> "I don't know"
    rewritten chunks scored 0.628 0.608 0.587 0.565  -> the right answer

It costs one extra LLM call per turn.

Run it with:
    python rag_chat.py                 chat in the terminal, type "end" to stop
    python rag_chat.py --plain         no rewriting, the way the course notebook does it
"""

import os
import sys

from dotenv import load_dotenv
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine import ContextChatEngine, CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.groq import Groq

import rag  # the config and the vector database from the previous notebook

# What each of those imported pieces is:
#
#   ContextChatEngine   retriever + memory + LLM in one object. On every turn it
#                       searches the documents with your message AND adds the
#                       conversation so far, then asks the LLM
#   CondensePlusContextChatEngine
#                       the same, but it first rewrites your message into a
#                       standalone question using the history, and searches with
#                       that. "What does she like to do?" becomes "What does the
#                       Queen of Hearts like to do?" before anything is retrieved
#   ChatMemoryBuffer    the conversation itself: a list of messages with a token
#                       budget. When it is full, the oldest messages fall off
#   ChatMessage         one message, with a role (SYSTEM / USER / ASSISTANT).
#                       SYSTEM messages are standing instructions, not chat
#   Groq                the 70B model, called over the internet

# The memory's token budget. Roughly a dozen exchanges; after that the bot
# genuinely forgets how the conversation started.
MEMORY_TOKENS = 3000   # the library's own default, spelled out so it can be changed

# The notebook's three system messages, plus the "say you don't know" line that
# keeps the bot from answering out of its own memory of the book.
SYSTEM_MESSAGES = [
    ChatMessage(role=MessageRole.SYSTEM,
                content="You are a knowledgeable, friendly assistant who explains "
                        "generative AI to people who are learning it. Your knowledge "
                        "comes from the textbooks provided as context."),
    ChatMessage(role=MessageRole.SYSTEM,
                content="Answer using only the context and the previous conversation. "
                        "If the answer is not in the context, say you don't know based "
                        "on these books - never fill the gap from general knowledge."),
    ChatMessage(role=MessageRole.SYSTEM,
                content="Name the book or chapter an idea comes from when the context shows it. "
                        "Explain in plain language, define jargon in passing, and keep "
                        "answers to a few sentences unless asked for more."),
]


def get_llm(temperature=rag.TEMPERATURE):
    """The Groq model. Note api_key, NOT token - the notebook's token= is ignored.

    Creating one costs about a millisecond (it only holds settings and an HTTP
    client), so the web app makes a new one per message to honour its slider.
    """
    return Groq(
        model=rag.LLM_MODEL,
        api_key=os.environ["GROQ_API_KEY"],
        temperature=temperature,
    )


def build_bot(index, llm, top_k=rag.TOP_K, condense=True, history=None):
    """A fresh chat engine. Cheap to build - nothing is loaded or downloaded here.

    condense=True (the default) rewrites the question using the conversation
    before searching, so "What does she like to do?" is searched for as "What
    does the Queen of Hearts like to do?".

    condense=False is what the course notebook does: it searches for the words
    exactly as typed. Any question containing "she", "it" or "that one" then
    retrieves badly, because those words say nothing about the topic.
    """
    # The search half: embeds a question and returns the top_k nearest chunks.
    retriever = index.as_retriever(similarity_top_k=top_k)
    # The conversation. history=None starts empty; app.py passes in the messages
    # belonging to one browser, so each visitor gets their own conversation.
    memory = ChatMemoryBuffer.from_defaults(token_limit=MEMORY_TOKENS, chat_history=history)

    if condense:
        return CondensePlusContextChatEngine.from_defaults(
            retriever=retriever, llm=llm, memory=memory,
            system_prompt="\n".join(m.content for m in SYSTEM_MESSAGES),
        )
    # prefix_messages are put at the very front of every request, before the
    # retrieved chunks and before the conversation. That is what a "system
    # prompt" is: instructions the user never sees and cannot overwrite.
    return ContextChatEngine(
        retriever=retriever, llm=llm, memory=memory, prefix_messages=SYSTEM_MESSAGES
    )


def show_sources(answer):
    """Which chunks this answer was built from - printed after every reply.

    If a reply looks wrong, look here first. Wrong chunks means the search
    failed, and no amount of prompt wording will fix that; raise TOP_K instead.
    """
    for hit in answer.source_nodes:
        print(f"      score {hit.score:.3f}  [{hit.node.metadata.get('file_name', '?')}]")


if __name__ == "__main__":
    # utf-8-sig, because PowerShell writes a BOM that would otherwise turn the
    # first key name into '<BOM>GROQ_API_KEY' and it would silently not be found.
    load_dotenv(os.path.join(rag.HERE, ".env"), encoding="utf-8-sig")
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("no GROQ_API_KEY - put it in generative_ai/.env (free key: console.groq.com)")

    # Question rewriting is on unless --plain asks for the notebook behaviour.
    condense = "--plain" not in sys.argv
    index = rag.load_or_build_index(rag.build_embeddings())
    bot = build_bot(index, get_llm(), condense=condense)

    mode = "questions rewritten before searching" if condense else "plain, no rewriting"
    print(f"\nready ({mode}). type 'end' to stop.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("end", "exit", "quit"):
            print("Bye.")
            break
        if not question:
            continue

        # .chat() searches the documents, adds the stored conversation, calls the
        # LLM, and saves both your message and the reply back into the memory.
        answer = bot.chat(question)
        print(f"Bot: {answer.response.strip()}")
        show_sources(answer)
        print()


# ----------------------------------------------------------------------------
# Notes
# ----------------------------------------------------------------------------
#
# Things to try for the challenge:
#   * ask a follow-up with a pronoun ("what does she do?") normally, then the
#     same conversation with --plain, and watch the chunk scores collapse
#   * lower MEMORY_TOKENS to about 200, chat for five turns, then ask about the
#     first one - the bot has forgotten it, because the oldest messages are
#     dropped when the buffer is full
#   * ask something that is NOT in the documents. With the "say you don't know"
#     line above it should refuse; delete that line and it will happily invent
#     an answer instead
