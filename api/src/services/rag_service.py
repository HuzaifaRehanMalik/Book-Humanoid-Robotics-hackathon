import os
from openai import OpenAI
from src.database.qdrant import client, COLLECTION_NAME

# Hugging Face will provide this automatically from Settings → Variables
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text: str):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def search_similar_chunks(query: str, limit: int = 3):
    query_vector = get_embedding(query)

    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit
    )

    return [hit.payload["text"] for hit in search_result]


def generate_answer(question: str):
    context_chunks = search_similar_chunks(question)
    context = "\n".join(context_chunks)

    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "Answer the question using only the provided context."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            }
        ]
    )

    return response.choices[0].message.content
