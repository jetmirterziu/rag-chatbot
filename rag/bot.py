import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Configuration
DB_PATH = "vector_db"
OLLAMA_MODEL = "mistral"

def get_rag_components():
    # 1. Setup Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    
    # 2. Base Retriever
    base_retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.5 
        }
    )

    # 3. Reranker
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=model, top_n=3)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    # 4. LLM
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)

    # 5. Chains

    # --- MEMORY REWRITER ---
    contextualize_q_system_prompt = """
    You are a query rewriting assistant.
    Your goal is to clarify the user's question for a search engine.

    RULES:
    1. **NO PRONOUNS:** You must replace EVERY instance of 'it', 'its', 'they', or 'their'
    with the actual Company Name from history. The output MUST NOT contain the word 'it'.
    2. **NEW TOPICS:** If the user explicitly mentions a NEW company name, use THAT name
    and ignore the history.
    3. **GIBBERISH:** If the input looks like random letters, return it AS IS.
    4. **NO HALLUCINATION:** Do NOT invent, infer, guess, or add any company names,
    facts, or details that are not explicitly present in the input or chat history.
    5. Return ONLY the rewritten question.
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