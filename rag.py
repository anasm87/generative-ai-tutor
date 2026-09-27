"""
RAG - retrieval augmented generation over a folder of documents.

Six steps, reading from a folder on this machine and keeping the vector database
on disk, so the documents are only embedded once.

    1. read every file in data/                     SimpleDirectoryReader
    2. split it into chunks                         SentenceSplitter
    3. turn each chunk into a vector                HuggingFaceEmbedding (local, free)
    4. store the vectors                            VectorStoreIndex  -> storage/
    5. find the chunks nearest to the question      as_retriever
    6. let the LLM answer from those chunks         Groq (over the API)

Step 5 needs no API key, so retrieval can always be checked on its own. Only
step 6 talks to Groq.

Setup:
    put GROQ_API_KEY=... in generative_ai/.env       (never commit that file)
    drop your own .txt / .pdf / .md files into data/ and delete storage/

Run it with:
    python rag.py                          a demo question
    python rag.py "your question here"
    python rag.py --rebuild                throw the database away and rebuild it
"""

import os
import shutil
import sys

from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# What each of those imported pieces actually is:
#
#   SimpleDirectoryReader  reads every file in a folder and returns one Document
#                          per file, carrying the file name with it
#   SentenceSplitter       cuts a Document into chunks, without slicing a sentence
#                          in half at the boundary
#   HuggingFaceEmbedding   turns a piece of text into a list of 384 numbers.
#                          This one runs here on the CPU, not over the internet
#   VectorStoreIndex       the "vector database": it keeps those numbers and can
#                          find the chunks whose numbers are nearest a question
#   StorageContext         points at the folder a VectorStoreIndex was saved in,
#                          so it can be read back instead of rebuilt
#   PromptTemplate         the text the LLM finally receives, with the retrieved
#                          chunks and the question pasted into the gaps

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")                 # your documents go in here
# storage/ is the saved vector database: a few json files holding every chunk and
# its 384 numbers. It exists so the documents are embedded once instead of on
# every run. Delete it (or pass --rebuild) whenever the documents or the chunking
# settings change, otherwise you keep querying the old vectors.
STORAGE_DIR = os.path.join(HERE, "storage")
EMBED_CACHE = os.path.join(HERE, "embedding_model")   # the downloaded embedding model

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384 dimensions, runs on the CPU
# Runs on Groq's servers. Note that "llama-3.3-70b-versatile", which most
# tutorials still name, has been retired and now answers 404 model_not_found.
# To see what your own key can use:
#     curl -H "Authorization: Bearer $env:GROQ_API_KEY" https://api.groq.com/openai/v1/models
LLM_MODEL = "openai/gpt-oss-120b"     # 120B, 131k context. Alternatives: openai/gpt-oss-20b, qwen/qwen3.8-27b
# The usual default is 800-1024 tokens. Measured on a book, that is too coarse: asking
# "Who likes to chop off heads?" retrieved four chunks, none of which even named
# the Queen, and the bot correctly answered "I don't know". At 300 the same
# question is answered. Long chunks average too many topics into one vector, so
# the signal for any single fact gets diluted.
#     800 -> 68 chunks,  0/4 named the Queen, answer: "I don't know"
#     300 -> 200 chunks, 1/4 named the Queen, answer: "The Queen"
#     150 -> 479 chunks, 2/4 named the Queen, answer: "The Queen"
CHUNK_SIZE, CHUNK_OVERLAP = 300, 60
# A small top-k, 2 to 4, is the common default. With 13 documents and ~8,000
# chunks that is far too few:
# "How does CLIP connect images and text?" retrieved only chunks about CLIP's
# results at top_k=4 and the bot correctly refused; at top_k=8 it also pulled in
# the method section and answered properly. More documents need a wider net.
TOP_K = 8
TEMPERATURE = 0.1                      # low, because we want facts and not creativity

DEFAULT_QUESTION = "What is self-attention and what problem does it solve?"

# Note the last line. Without it the model happily answers from its own memory,
# and then there is no way to tell real retrieval from invention.
#
# {context_str} and {query_str} are not free choices - LlamaIndex looks for
# exactly these two names and fills them in with the retrieved chunks and the
# question. Rename them and nothing gets filled in.
PROMPT = PromptTemplate(
    """Here is the context:
{context_str}

Answer the question using only the context above. Keep the answer short.
If the context does not contain the answer, say "I don't know based on these documents."
Question: {query_str}
Answer:"""
)


def build_embeddings():
    """The embedding model. Downloaded once, then read from embedding_model/."""
    return HuggingFaceEmbedding(EMBED_MODEL, cache_folder=EMBED_CACHE)


def load_or_build_index(embeddings):
    """Load the saved database, or build it from data/ the first time."""
    if os.path.isdir(STORAGE_DIR):
        print(f"   loading the saved database from storage/")
        # The same embedding model has to be passed back in - vectors made by a
        # different model are not comparable.
        return load_index_from_storage(
            StorageContext.from_defaults(persist_dir=STORAGE_DIR), embed_model=embeddings
        )

    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.get_nodes_from_documents(documents)
    print(f"   {len(documents)} file(s) -> {len(chunks)} chunks of ~{CHUNK_SIZE} tokens")
    print(f"   embedding them with {EMBED_MODEL} (first run also downloads the model)")

    # The one line that does the whole pipeline: it runs the splitter over the
    # documents, sends every chunk through the embedding model, and keeps the
    # resulting vectors in memory so they can be searched. This is the slow part
    # (a few seconds for a book, minutes for a big PDF) - hence saving it below.
    index = VectorStoreIndex.from_documents(
        documents, transformations=[splitter], embed_model=embeddings
    )
    index.storage_context.persist(persist_dir=STORAGE_DIR)   # write it to storage/
    print(f"   saved to storage/ - next run skips all of this")
    return index


def show_retrieved(index, question):
    """Step 5 on its own: which chunks does the search actually find?

    as_retriever() gives the search half of the system with no LLM attached: it
    embeds the question and returns the TOP_K nearest chunks. Each hit is a
    NodeWithScore:
        hit.score          how close it was, roughly 0 (unrelated) to 1 (same text)
        hit.node.text      the chunk itself, exactly as the LLM will see it
        hit.node.metadata  where it came from, e.g. the file name
    """
    for i, hit in enumerate(index.as_retriever(similarity_top_k=TOP_K).retrieve(question), 1):
        source = hit.node.metadata.get("file_name", "?")
        snippet = " ".join(hit.node.text.split())[:160]   # squash newlines, keep it short
        print(f"   {i}. score {hit.score:.3f}  [{source}]  {snippet}...")


def answer_with_llm(index, question):
    """Step 6: hand the retrieved chunks to the LLM.

    A query engine is retriever + prompt + LLM in one object. Calling .query()
    does all of it: search, fill the template, send it to Groq, return the reply.
    The reply carries both the text (.response) and the chunks it was built from
    (.source_nodes), which is what makes an answer checkable.
    """
    from llama_index.llms.groq import Groq   # imported here so the file still runs without a key

    llm = Groq(model=LLM_MODEL, temperature=TEMPERATURE)   # the key is read from GROQ_API_KEY
    engine = index.as_query_engine(
        llm=llm, text_qa_template=PROMPT, similarity_top_k=TOP_K
    )
    answer = engine.query(question)
    print(f"\n   {answer.response.strip()}\n")
    print("   built from:")
    for hit in answer.source_nodes:
        print(f"      score {hit.score:.3f}  [{hit.node.metadata.get('file_name', '?')}]")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--rebuild"]
    if "--rebuild" in sys.argv and os.path.isdir(STORAGE_DIR):
        shutil.rmtree(STORAGE_DIR)
        print("removed storage/ - the documents will be embedded again")

    question = " ".join(args) if args else DEFAULT_QUESTION
    # utf-8-sig, because PowerShell writes a BOM that would otherwise turn the
    # first key name into '<BOM>GROQ_API_KEY' and it would silently not be found.
    load_dotenv(os.path.join(HERE, ".env"), encoding="utf-8-sig")

    print("\n== the database ==")
    index = load_or_build_index(build_embeddings())

    print(f"\n== retrieval (no API key needed) ==\n   question: {question}")
    show_retrieved(index, question)

    print("\n== the answer ==")
    if os.environ.get("GROQ_API_KEY"):
        answer_with_llm(index, question)
    else:
        print("   no GROQ_API_KEY found, so the LLM step is skipped.")
        print("   put GROQ_API_KEY=your_key in generative_ai/.env and run again.")
        print("   free key: https://console.groq.com")


# ----------------------------------------------------------------------------
# Notes
# ----------------------------------------------------------------------------
#
# Retrieval is the part that breaks, not the LLM. If an answer looks wrong, read
# the "retrieval" section above first: if the right passage isn't in those four
# chunks, the model never had a chance. Raising TOP_K helps more than changing
# the prompt.
#
# Worth experimenting with:
#   CHUNK_SIZE      300 vs 800 vs 1500 - small chunks retrieve precisely but lose
#                   the surrounding context, large ones are the other way round
#   TOP_K           2 vs 4 vs 8 - more chunks, more tokens, better recall
#   EMBED_MODEL     MiniLM is English-only. For German documents try a
#                   multilingual model, e.g. intfloat/multilingual-e5-small
#   TEMPERATURE     0.1 vs 0.8 - factual vs chatty
#
# Changing CHUNK_SIZE or EMBED_MODEL means the stored vectors are stale, so run
# with --rebuild afterwards.
