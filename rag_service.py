import os

import faiss
import numpy as np

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq


# ==========================================
# ENVIRONMENT
# ==========================================

load_dotenv()


# ==========================================
# CONFIGURATION
# ==========================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DOC_PATH = os.path.join(
    "docs",
    "utility_sop.txt"
)

TOP_K = 3


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ==========================================
# LOAD DOCUMENT
# ==========================================

def load_document():

    with open(
        DOC_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()


# ==========================================
# CREATE DOCUMENT CHUNKS
# ==========================================

def create_chunks(
    text,
    chunk_size=800
):

    words = text.split()

    chunks = []

    for i in range(
        0,
        len(words),
        chunk_size
    ):

        chunk = " ".join(
            words[
                i:i + chunk_size
            ]
        )

        if chunk.strip():

            chunks.append(chunk)

    return chunks


# ==========================================
# BUILD FAISS INDEX
# ==========================================

def build_index(chunks):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype(
        "float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index


# ==========================================
# INITIALIZE RAG KNOWLEDGE BASE
# ==========================================

DOCUMENT = load_document()

CHUNKS = create_chunks(
    DOCUMENT
)

INDEX = build_index(
    CHUNKS
)


# ==========================================
# RETRIEVE RELEVANT CONTEXT
# ==========================================

def retrieve_context(
    question,
    top_k=TOP_K
):

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    distances, indices = INDEX.search(
        query_embedding,
        min(
            top_k,
            len(CHUNKS)
        )
    )

    retrieved_chunks = []

    for idx, distance in zip(
        indices[0],
        distances[0]
    ):

        if idx < len(CHUNKS):

            retrieved_chunks.append(
                {
                    "text": CHUNKS[idx],
                    "distance": float(distance)
                }
            )

    return retrieved_chunks


# ==========================================
# GENERATE GROQ ANSWER
# ==========================================

def answer_question(question):

    retrieved_chunks = retrieve_context(
        question
    )

    context = "\n\n".join(
        item["text"]
        for item in retrieved_chunks
    )

    client = Groq(
        api_key=os.getenv(
            "GROQ_API_KEY"
        )
    )

    prompt = f"""
You are the Utility Sentinel AI
Knowledge Assistant.

Answer the user's question using
ONLY the provided Utility SOP context.

Do not invent procedures,
measurements, sensor readings,
events, or operational facts.

If the answer is not available in
the provided SOP context, say:

"The available utility SOP does not
provide enough information to answer
this question."

UTILITY SOP CONTEXT:

{context}


USER QUESTION:

{question}


Provide a concise and practical
operational answer.
"""

    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2
    )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    return {
        "answer": answer,
        "sources": retrieved_chunks
    }