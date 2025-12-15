import os
import pandas as pd

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Choose ONE that you actually pulled in Ollama:
EMBED_MODEL = "mxbai-embed-large"   # or "nomic-embed-text"

def get_retriever(csv_path: str, k: int = 5):
    df = pd.read_csv(csv_path)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    db_location = "./chroma_langchain_db"
    os.makedirs(db_location, exist_ok=True)

    # Unique collection per CSV so papers don't mix
    collection_name = "paper_" + os.path.splitext(os.path.basename(csv_path))[0]

    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=db_location,
        embedding_function=embeddings,
    )

    # Add docs only if collection is empty
    if vector_store._collection.count() == 0:
        documents, ids = [], []

        for _, row in df.iterrows():
            documents.append(
                Document(
                    page_content=str(row["text"]),
                    metadata={
                        "source_file": str(row["source_file"]),
                        "page": int(row["page"]),
                        "chunk_id": str(row["chunk_id"]),
                    },
                )
            )
            # stable id per chunk
            ids.append(str(row["chunk_id"]))

        vector_store.add_documents(documents=documents, ids=ids)

    return vector_store.as_retriever(search_kwargs={"k": k})
