from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

def get_retriever(csv_path: str, k: int = 5):
    df = pd.read_csv(csv_path)

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")  # or "nomic-embed-text"

    db_location = "./chroma_langchain_db"
    os.makedirs(db_location, exist_ok=True)

    # IMPORTANT: unique collection per CSV so papers don't mix
    collection_name = "paper_" + os.path.splitext(os.path.basename(csv_path))[0]

    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=db_location,
        embedding_function=embeddings
    )

    # add docs only if this collection is empty
    existing = vector_store._collection.count()
    if existing == 0:
        documents, ids = [], []
        for i, row in df.iterrows():
            documents.append(
                Document(
                    page_content=str(row["text"]),
                    metadata={
                        "source_file": str(row["source_file"]),
                        "page": int(row["page"]),
                        "chunk_id": str(row["chunk_id"]),
                        "csv_file": os.path.basename(csv_path),
                    },
                )
            )
            ids.append(str(i))

        vector_store.add_documents(documents=documents, ids=ids)

    return vector_store.as_retriever(search_kwargs={"k": k})
