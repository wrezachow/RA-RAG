from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from pdf_to_csv import pdf_to_csv
from vector import get_retriever   # <-- function, not a global retriever

# --------- STEP 1: choose input / output ---------
pdf_path = input("PDF path: ").strip()
out_csv  = input("Output CSV name: ").strip()

# --------- STEP 2: convert PDF -> CSV ---------
pdf_to_csv(pdf_path, out_csv)

# --------- STEP 3: build retriever for THIS CSV ---------
retriever = get_retriever(out_csv, k=5)

# --------- STEP 4: LLM setup ---------
model = OllamaLLM(model="llama3")

template = """
You are a helpful assistant for summarizing and answering questions using research papers.
Use ONLY the provided research excerpts.
Cite claims like (page X, chunk Y).

Research:
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

# --------- STEP 5: chat loop ---------
while True:
    print("\n-------------------------------")
    question = input("Ask your question (q to quit): ").strip()
    if question.lower() == "q":
        break

    docs = retriever.invoke(question)
    research = format_docs(docs)

    result = chain.invoke({
        "research": research,
        "question": question
    })

    print(result)
