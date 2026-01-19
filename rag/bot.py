import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI  # <--- CHANGED: Switched to OpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Configuration
DB_PATH = "vector_db"

def get_rag_components():
    # 1. Setup Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    
    # 2. Base Retriever
    base_retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.3
        }
    )

    # 3. Reranker
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=model, top_n=3)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    # 4. LLM 
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]

    if not api_key:
        # Fallback to prevent crash if key is missing, but will error if used
        print("WARNING: OpenAI API Key not found!")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key
    )

    # 5. Chains

    # --- MEMORY REWRITER ---
    contextualize_q_system_prompt = """
    You are a query rewriting assistant.
    Your ONLY goal is to rephrase the user's question into a standalone search query.

    RULES:
    1. **NO ANSWERS:** Do NOT answer the question. Do NOT define terms.
       - BAD Output: "Visa is a credit card company."
       - GOOD Output: "What is Visa Inc?"
    2. **NO PRONOUNS:** You must replace EVERY instance of 'it', 'its', 'they', or 'their'
       with the actual Company Name from history.
    3. **NEW TOPICS:** If the user explicitly mentions a NEW company name, use THAT name
       and ignore the history.
    4. **NO HALLUCINATION:** Do NOT add facts or details not present in the input.
    5. Return ONLY the rewritten question string.
    """.strip()

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_chain = contextualize_q_prompt | llm | StrOutputParser()

    # --- ANSWERER ---
    qa_system_prompt = """
    You are a helpful financial assistant.
    Use the retrieved context below to answer the user's question.

    IMPORTANT:
    If the context contains data for multiple companies, explicitly state which company
    the data belongs to.

    --- Context ---
    {context}
    """.strip()

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        ("human", "{input}"),
    ])
    
    return history_aware_chain, compression_retriever, qa_prompt, llm