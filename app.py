import streamlit as st
import os
import zipfile
from google import genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# 1. Website ka Sunder Look aur Title
st.set_page_config(page_title="Nyaya AI", page_icon="⚖️", layout="centered")

st.markdown("## ⚖️ Nyaya AI: Legal Assistant Chatbot")
st.write("Apna kanooni sawal kisi bhi bhasha me poochein")
st.write("---")

# 2. ZIP Folder ko automatic extract karna (Roz upload ka jhanjhat khatam!)
@st.cache_resource
def setup_database():
    if os.path.exists("legal_tijori.zip") and not os.path.exists("legal_tijori"):
        with zipfile.ZipFile("legal_tijori.zip", 'r') as zip_ref:
            zip_ref.extractall("legal_tijori")
            
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory="legal_tijori", embedding_function=embedding_function)

try:
    db = setup_database()
    db_ready = True
except Exception as e:
    db_ready = False
    st.error("⚠️ Database load hone me dikkat hai. Check karein ki legal_tijori.zip upload hui hai.")

# 3. Gemini AI Connect Karna
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
    ai_ready = True
except Exception:
    ai_ready = False
    st.warning("🔒 Please configure GEMINI_API_KEY in Space Settings Secrets.")

# 4. Asli Chat Interface (User sirf query dalega)
user_question = st.chat_input("Yahan apna sawal likhein (e.g., What is Article 21?)...")

if user_question:
    if not db_ready or not ai_ready:
        st.error("Backend configuration incomplete. Please check files and keys.")
    else:
        # User ka sawal screen par dikhana
        with st.chat_message("user"):
            st.markdown(f"**{user_question}**")
            
        # Database se kanoon nikalna
        retrieved_docs = db.similarity_search(user_question, k=1)
        
        if retrieved_docs:
            best_match = retrieved_docs[0]
            source_book = os.path.basename(best_match.metadata.get('source', 'Unknown Book'))
            page_no = best_match.metadata.get('page', 0) + 1
            
            # AI se multi-language wala formatted jawab banwana
            with st.chat_message("assistant"):
                st.markdown(f"📚 **Source:** `{source_book}` | 📄 **Page:** `{page_no}`")
                
                prompt = f"""
                You are a highly professional yet friendly Indian AI Legal Assistant. 
                
                CRITICAL INSTRUCTION: Detect the language of the user's question (it could be English, Hinglish, Hindi, Bengali, Tamil, Marathi, Gujarati, French, etc.). You MUST write the entire response in that SAME exact language and script used by the user. If the user asks in Hinglish, reply in Hinglish. If the user asks in Hindi script, reply in Hindi script.
                
                Strictly follow this layout using the detected language:
                ## 📜 Legal Query: {user_question}
                ---
                ### 💡 Simple Terms Me Iska Matlab / Meaning
                * **Main Concept**: Explain beautifully in 2-3 lines.
                * **Key Details**: Give clear details, punishment, or facts in simple bullet points.
                
                ### 🌟 Rules / Points
                * 📌 Point 1 with matching emoji.
                * 📌 Point 2 with matching emoji.
                
                Context text from local database: {best_match.page_content}
                """
                try:
                    response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"AI Server Error: {str(e)}")
                    
                st.caption("⚠️ Disclaimer: Yeh ek AI tool hai. Formal kanooni salah ke liye lawyer se sampark karein.")
        else:
            with st.chat_message("assistant"):
                st.markdown("❌ Maaf kijiye, mujhe isse juda kanoon aapki files me nahi mila.")
