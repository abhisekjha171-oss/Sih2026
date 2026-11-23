import streamlit as st
import os
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="IP-SAKTI Sahayak | Ayurveda IPR Assistant",
    page_icon="🌿",
    layout="wide"
)

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; }
    [data-testid="stSidebar"] { background-color: #1E3A2F; color: white; }
    [data-testid="stSidebar"] * { color: white !important; }
    .stChatMessage {
        background-color: white; border-radius: 10px; padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 15px;
    }
    h1, h2, h3 { color: #1E3A2F; font-family: 'Helvetica Neue', sans-serif; }
    .disclaimer {
        background-color: #FFF3CD; color: #856404; padding: 10px;
        border-radius: 5px; font-size: 0.9em; text-align: center;
        margin-top: 20px; border: 1px solid #FFEEBA;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZATION (Cached for speed) ---
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    collection = chromadb.PersistentClient(path="db").get_collection("ipsakti")
    return embedder, collection

embedder, collection = load_models()

# Load API Key safely
def load_api_key():
    # 1. Try Streamlit secrets (for Cloud)
    try:
        return st.secrets["GEMINI_API_KEY"]
    except:
        pass
    # 2. Try environment variable
    key = os.getenv("GEMINI_API_KEY", "")
    if key: return key
    # 3. Try local file
    secrets_path = os.path.join(".streamlit", "secrets.toml")
    if os.path.isfile(secrets_path):
        with open(secrets_path) as f:
            for line in f:
                if line.strip().startswith("GEMINI_API_KEY"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""

api_key = load_api_key()
gemini = genai.Client(api_key=api_key)

# --- HELPER FUNCTIONS ---
def fetch_live_web_context(query):
    try:
        results = DDGS().text(f"{query} India government", max_results=2)
        if not results:
            return "No live web results found."
        
        top_url = results[0]['href']
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(top_url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)[:3000]
        return f"Live Web Data from {top_url}:\n{text}"
    except Exception as e:
        return f"Web search failed: {str(e)}"

# --- SIDEBAR ---
with st.sidebar:
    st.title("IP-SAKTI Sahayak")
    st.markdown("**Hybrid AI Assistant for Ayurveda IPR & Regulatory Compliance**")
    st.markdown("---")
    
    jurisdiction = st.radio("Select Jurisdiction:", ["India", "International"])
    
    st.markdown("---")
    st.subheader("💡 Example Questions")
    if st.button("Can I patent a traditional formulation?"):
        st.session_state.example_q = "Can I patent a traditional Ayurvedic formulation using turmeric?"
    if st.button("Latest Biological Diversity Act?"):
        st.session_state.example_q = "What is the latest amendment to the Biological Diversity Act?"
    if st.button("How to protect my brand name?"):
        st.session_state.example_q = "How can I protect my Ayurvedic brand name?"

# --- MAIN PAGE ---
st.title("🌿 IP-SAKTI Sahayak")
st.markdown("Your source-cited AI assistant for protecting Ayurvedic innovation.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Handle example button clicks
if "example_q" in st.session_state:
    prompt = st.session_state.example_q
    del st.session_state.example_q # Clear it
else:
    prompt = None

# Chat Input
user_input = st.chat_input("Ask a question about Ayurveda IPR...")
if user_input:
    prompt = user_input

# If there is a prompt (either from button or input box)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing legal database and official portals..."):
            # 1. Router
            live_keywords = ["latest", "current", "update", "amendment", "news", "2024", "2025", "status"]
            
            if any(word in prompt.lower() for word in live_keywords):
                context = fetch_live_web_context(prompt)
                engine = "Live Web Search"
            else:
                q_vec = embedder.encode([prompt]).tolist()
                results = collection.query(query_embeddings=q_vec, n_results=2)
                context = "\n".join([doc for doc in results["documents"][0]])
                engine = "Local Database (RAG)"

            # 2. Formatted Prompt
            final_prompt = f"""
            You are IP-SAKTI Sahayak. Provide information, NOT legal advice.
            Use ONLY the context below.
            
            Format STRICTLY:
            **1. Summary:** (2 sentences)
            **2. Legal/Regulatory Details:** (Bullet points)
            **3. Next Steps:** (Numbered list)
            **4. Source Citation:** (Mention the source)
            
            Context: {context}
            Question: {prompt}
            """
            
            # 3. LLM Call with Fallback Logic
            try:
                r = gemini.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=final_prompt,
                    config={"automatic_function_calling": {"disable": True}}
                )
                answer = r.text
            except Exception as e:
                # Simple fallback message if Gemini fails
                answer = f"⚠️ **API Service Busy.** Google's servers are currently experiencing high demand. Please try again in a few seconds. \n\n*(Error details: {str(e)})*"

            st.markdown(answer)
            st.caption(f"Engine: {engine} | Jurisdiction: {jurisdiction}")
            
            st.session_state.messages.append({"role": "assistant", "content": answer})

# Display chat history (moved to bottom to show chronological order)
for message in st.session_state.messages[:-2]: # Skip the last two we just rendered
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- FOOTER DISCLAIMER ---
st.markdown(
    '<div class="disclaimer">⚖️ <b>Disclaimer:</b> IP-SAKTI Sahayak provides information, not legal advice. Always consult a qualified IP professional.</div>', 
    unsafe_allow_html=True
)
