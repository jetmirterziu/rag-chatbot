__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import streamlit as st
import sys
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Ensure we can import from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag.bot import get_rag_components

st.set_page_config(page_title="Finance RAG Chatbot", page_icon="💰")
st.title("💰 Finance RAG Agent")
st.caption("Powered by OpenAI (GPT-4o) & Cross-Encoder Reranking")

# --- 🚦 THE TRAFFIC COP (GUARDRAIL) 🚦 ---
def route_query(user_input, llm):
    router_system_prompt = (
        "You are a firewall for a Financial AI. Classify the user input into exactly one category:\n"
        "1. **GREETING**: If the user says 'hi', 'hello', 'thanks', or 'how are you'.\n"
        "2. **OFF_TOPIC**: If the user asks about cooking, sports, movies, or illegal acts.\n"
        "3. **FINANCE**: If the user asks about business, money, companies, OR asks a follow-up question like 'is it higher?' or 'what about 2023?'.\n\n"
        "OUTPUT INSTRUCTION: Reply ONLY with the category name (GREETING, OFF_TOPIC, or FINANCE)."
    )
    
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", router_system_prompt),
        ("human", "{input}"),
    ])
    
    chain = router_prompt | llm | StrOutputParser()
    result = chain.invoke({"input": user_input}).strip().upper()
    
    if "GREETING" in result: return "GREETING"
    if "OFF_TOPIC" in result: return "OFF_TOPIC"
    return "FINANCE"

# --- UI LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---  WELCOME SCREEN ---
if len(st.session_state.messages) == 0:
    st.markdown("""
    ### 👋 Welcome!
    I am your AI Financial Analyst. I have studied the **2023 Annual Reports** for major financial institutions, including:
    
    * 🏢 **BlackRock Inc.**
    * 💳 **Visa Inc.**
    * 🏦 **JPMorgan Chase & Co.**
    * ...and others.
    
    **I can help you with:**
    * 📊 **Financial Highlights:** Ask for revenue, net income, or EPS.
    * ⚠️ **Risk Analysis:** Understand the strategic risks companies face.
    * 🔮 **Future Outlook:** See what the CEOs are predicting.
    """)
    st.info("💡 **Try asking:** 'What was BlackRock's revenue in 2023?' or 'What are the main risks for Visa?'")

# 1. DISPLAY HISTORY
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Check if this message has stored sources
        if "sources" in message:
            with st.expander("See Source Documents"):
                for source in message["sources"]:
                    st.markdown(f"**📄 {source['source']}** (Page {source['page']})")
                    st.caption(source['content'])

if prompt := st.chat_input("Ask about the financial reports..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # 1. Get Tools
                history_aware_chain, retriever, qa_prompt, llm = get_rag_components()

                # 2. 🚦 RUN THE ROUTER 🚦
                category = route_query(prompt, llm)
                
                # We need to initialize this list to save it later
                saved_sources = [] 

                # --- PATH A: GREETING ---
                if category == "GREETING":
                    answer = "Hello! 👋 I am ready to analyze your financial documents. Ask me about any company in your database."

                # --- PATH B: OFF_TOPIC ---
                elif category == "OFF_TOPIC":
                    answer = "🚫 **Guardrail Alert:** I can only discuss financial reports. I cannot help with that topic."

                # --- PATH C: FINANCE (Run RAG) ---
                else:
                    # A. Memory Rewrite
                    chat_history = []
                    for msg in st.session_state.messages[-4:]: 
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            chat_history.append(AIMessage(content=msg["content"]))

                    rephrased_question = prompt
                    if len(chat_history) > 0:
                        rephrased_question = history_aware_chain.invoke({
                            "chat_history": chat_history,
                            "input": prompt
                        }).strip().replace('"', '')

                    if rephrased_question != prompt:
                         st.caption(f"🧠 *Rewritten search:* {rephrased_question}")

                    # B. Retrieve
                    sources = retriever.invoke(rephrased_question)
                    
                    if not sources:
                        answer = "I cannot find any relevant information in the documents regarding your question."
                    else:
                        # C. Build Context & Save Sources for History
                        context_parts = []
                        for doc in sources:
                            source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
                            page_num = doc.metadata.get("page", "1")
                            
                            # Add to context text
                            context_parts.append(f"--- SOURCE: {source_name} ---\n{doc.page_content}")
                            
                            # Save to list for UI
                            saved_sources.append({
                                "source": source_name,
                                "page": page_num,
                                "content": doc.page_content[:] + "..."
                            })
                        
                        context_text = "\n\n".join(context_parts)
                        
                        # D. Generate Answer
                        final_chain = qa_prompt | llm | StrOutputParser()
                        answer = final_chain.invoke({"context": context_text, "input": rephrased_question})

                # 3. Final Display
                st.markdown(answer)

                # Show sources for the current message immediately
                if saved_sources:
                    with st.expander("See Source Documents"):
                        for source in saved_sources:
                            st.markdown(f"**📄 {source['source']}** (Page {source['page']})")
                            st.caption(source['content'])

                # 4. SAVE TO HISTORY (Now including sources!)
                # We save a dictionary with 'role', 'content', AND 'sources'
                message_data = {"role": "assistant", "content": answer}
                if saved_sources:
                    message_data["sources"] = saved_sources
                
                st.session_state.messages.append(message_data)
            
            except Exception as e:
                st.error(f"❌ Error: {e}")