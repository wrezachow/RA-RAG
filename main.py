from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from pdf_to_csv import pdf_to_csv
from vector import get_retriever

# --- Choose PDF (single paper mode) ---
pdf_path = input("PDF path (e.g. papers/paper.pdf): ").strip()
out_csv  = input("Output CSV (Enter for default): ").strip() or None

# Convert PDF -> CSV (or reuse if up-to-date)
out_csv = pdf_to_csv(pdf_path, out_csv)

# Build retriever for this CSV
retriever = get_retriever(out_csv, k=5)

# LLM (chat model)
model = OllamaLLM(model="llama3")

template = """
You are a helpful assistant for summarizing and answering questions using research papers.
Use ONLY the provided research excerpts.
If the answer is not in the excerpts, say: "Not found in the provided research excerpts."
Cite claims like (page X, chunk Y).

Research excerpts:
{research}

Question: {question}
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[page {d.metadata['page']} | {d.metadata['chunk_id']}]\n{d.page_content}"
        for d in docs
    )

while True:
    print("\n-------------------------------")
    question = input("Ask your question (q to quit): ").strip()
    if question.lower() == "q":
        break

    docs = retriever.invoke(question)
    research = format_docs(docs)

    result = chain.invoke({"research": research, "question": question})
    print(result)
