# # rag_pipeline.py
# import os
# from pathlib import Path
# import fitz  # PyMuPDF

# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_ollama import OllamaLLM
# from langchain.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.load import dumps, loads
# from sentence_transformers import SentenceTransformer
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from operator import itemgetter

# ### 0. Convertir PDF a TXT si no existe
# pdf_path = "Université — Collectives - UniTA Project Datacloud.pdf"
# txt_path = "UNITA.txt"

# if not Path(txt_path).exists():
#     print("📄 Convirtiendo PDF a texto plano...")
#     doc = fitz.open(pdf_path)
#     with open(txt_path, "w", encoding="utf-8") as f:
#         for page in doc:
#             f.write(page.get_text("text") + "\n\n")
#     print(f"✅ Guardado como {txt_path}")

# ### 1. Cargar el texto plano
# loader = TextLoader(txt_path, encoding="utf-8")
# documents = loader.load()

# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = splitter.split_documents(documents)

# ### 2. Embeddings con SentenceTransformers
# embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ### 3. Crear o cargar el vectorstore
# persist_path = "./chroma_store"
# if not Path(persist_path).exists():
#     vectorstore = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory=persist_path)
#     vectorstore.persist()
# else:
#     vectorstore = Chroma(persist_directory=persist_path, embedding_function=embedding)

# retriever = vectorstore.as_retriever()

# ### 4. LLM para preguntas alternativas
# llm_query_gen = OllamaLLM(model="gemma:7b", base_url="http://chat-eva.univ-pau.fr:11434")

# prompt = ChatPromptTemplate.from_template("""Generate 4 rephrasings of the user question to help retrieve more relevant documents from a knowledge base. Separate each with a newline.

# Question: {question}""")

# generate_queries = (
#     prompt
#     | llm_query_gen
#     | StrOutputParser()
#     | (lambda x: x.split("\n"))
# )

# ### 5. RRF - Reciprocal Rank Fusion
# def reciprocal_rank_fusion(results: list[list], k=60):
#     fused_scores = {}
#     for docs in results:
#         for rank, doc in enumerate(docs):
#             doc_str = dumps(doc)
#             fused_scores[doc_str] = fused_scores.get(doc_str, 0) + 1 / (rank + k)
#     reranked = [
#         (loads(doc), score)
#         for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
#     ]
#     return [doc for doc, _ in reranked]

# retrieval_chain = generate_queries | retriever.map() | reciprocal_rank_fusion

# ### 6. Cadena RAG final
# prompt_final = ChatPromptTemplate.from_template("""Answer the following question based on the context:

# {context}

# Question: {question}""")

# qa_llm = OllamaLLM(model="gemma:7b", base_url="http://chat-eva.univ-pau.fr:11434")

# final_rag_chain = (
#     {"context": retrieval_chain, "question": itemgetter("question")}
#     | prompt_final
#     | qa_llm
#     | StrOutputParser()
# )

# rag_pipeline.py
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load vectorstore
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_store", embedding_function=embedding)
retriever = vectorstore.as_retriever()

# Query rephraser LLM
llm_query_gen = OllamaLLM(model="gemma:7b", base_url="http://chat-eva.univ-pau.fr:11434")
prompt = ChatPromptTemplate.from_template("""Generate 4 rephrasings of the user question to help retrieve more relevant documents from a knowledge base. Separate each with a newline.
Question: {question}""")
generate_queries = prompt | llm_query_gen | StrOutputParser() | (lambda x: x.split("\n"))

# Reciprocal Rank Fusion
def reciprocal_rank_fusion(results: list[list], k=60):
    from langchain.load import dumps, loads
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            fused_scores[doc_str] = fused_scores.get(doc_str, 0) + 1 / (rank + k)
    return [loads(doc) for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]

retrieval_chain = generate_queries | retriever.map() | reciprocal_rank_fusion

# Final RAG chain
qa_llm = OllamaLLM(model="gemma:7b", base_url="http://chat-eva.univ-pau.fr:11434")
prompt_final = ChatPromptTemplate.from_template("""Answer the following question based on the context:\n\n{context}\n\nQuestion: {question}""")
final_rag_chain = (
    {"context": retrieval_chain, "question": itemgetter("question")}
    | prompt_final
    | qa_llm
    | StrOutputParser()
)
