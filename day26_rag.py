import pymupdf
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama


# =========================
# 0. Load Embedding Model
# =========================

model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# 1. Read the PDF
# =========================

def documents(doc, question):

    doc = pymupdf.open(
        stream=doc.read(),
        filetype="pdf"
    )

    chunk_size = 100
    overlap = 20
    step = chunk_size - overlap

    chunks = []

    for page_number, page in enumerate(doc):

        text = page.get_text()
        text_list = text.split()

        for j in range(0, len(text_list), step):

            chunk = {
                "text": " ".join(text_list[j:j + chunk_size]),
                "page": page_number + 1
            }

            chunks.append(chunk)


    # =========================
    # 2. Embed the chunks
    # =========================

    for chunk in chunks:

        text_data = chunk["text"]

        embedding = model.encode(text_data)

        chunk["embedding"] = embedding


    # =========================
    # 3. Get and embed query
    # =========================

    query_embedding = model.encode(question)

    query_embedding = np.asarray(query_embedding)


    # =========================
    # 4. Calculate similarities
    # =========================

    cosines = []

    for chunk in chunks:

        chunk_embedding = np.asarray(chunk["embedding"])

        cosine = (
            np.sum(query_embedding * chunk_embedding)
            /
            (
                np.sqrt(np.sum(query_embedding ** 2))
                *
                np.sqrt(np.sum(chunk_embedding ** 2))
            )
        )

        cosines.append(cosine)


    # =========================
    # 5. Retrieve top 3 chunks
    # =========================

    indexes = np.argsort(cosines)

    top_indexes = indexes[-3:][::-1]

    retrieved_chunks = []

    for i in top_indexes:

        retrieved_chunks.append(chunks[i])


    # =========================
    # 6. Build context
    # =========================

    context = []

    for relevant_info in retrieved_chunks:

        context.append(
            f"Source: Page {relevant_info['page']}\n"
            f"Content: {relevant_info['text']}"
        )

    context_sentence = "\n\n".join(context)


    # =========================
    # 7. Create prompt
    # =========================

    prompt = f"""
    Context:

    {context_sentence}

    Question:

    {question}

    Instruction:
    - Answer the question directly and naturally.
    - Use only information supported by the context.
    - Do not mention the context, retrieved chunks, or the RAG system.
    - Do not say "based on the provided context."
    - If the context does not contain enough information to answer the question,
    clearly say that the document does not provide enough information.
    """


    # =========================
    # 8. Ask Ollama
    # =========================

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]


    # =========================
    # 9. Return answer
    # =========================

    return answer