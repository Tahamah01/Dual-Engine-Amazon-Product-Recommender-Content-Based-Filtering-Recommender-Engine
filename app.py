import streamlit as st
import requests

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Amazon AI Recommender Engine",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Dual Engine Amazon Product Recommender")
st.caption("Comparing Traditional Keyword Matching (TF-IDF) vs. Deep Semantic Embeddings (Nomic Transformer)")

# Check API status
try:
    health = requests.get(f"{API_BASE_URL}/health", timeout=3).json()
    if not health.get("model_loaded"):
        st.error("Backend model is still loading, please refresh in a moment...")
        st.stop()
except Exception:
    st.error("Cannot connect to FastAPI backend. Ensure `uvicorn main:app --reload` is running.")
    st.stop()

# Sidebar options
st.sidebar.header("Recommendation Settings")
mode = st.sidebar.radio("Input Mode", ["Browse Catalog Product", "Free-Text Search"])
top_k = st.sidebar.slider("Number of Recommendations", min_value=3, max_value=10, value=5)

if mode == "Browse Catalog Product":
    st.subheader("Item-to-Item Recommendation")
    
    # Fetch catalog items from API
    @st.cache_data
    def fetch_catalog():
        res = requests.get(f"{API_BASE_URL}/catalog?limit=1000")
        return res.json() if res.status_code == 200 else []
    
    catalog = fetch_catalog()
    if catalog:
        options = {f"{item['title']} (ASIN: {item['parent_asin']})": item['parent_asin'] for item in catalog}
        selected_label = st.selectbox("Select a product from the catalog:", list(options.keys()))
        selected_asin = options[selected_label]
        
        if st.button("Get Similar Products"):
            with st.spinner("Calculating similarity scores..."):
                response = requests.get(f"{API_BASE_URL}/recommend/asin/{selected_asin}?top_k={top_k}")
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Showing recommendations for: **{data['target_title']}**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📑 TF-IDF (Keyword Baseline)")
                        for item in data['tfidf_recs']:
                            st.info(f"**{item['title']}**\n\n Price: {item['price']} | Rating: ⭐ {item['rating']} | ASIN: `{item['asin']}`")
                            
                    with col2:
                        st.markdown("### 🧠 Transformer (Deep Semantic)")
                        for item in data['transformer_recs']:
                            st.success(f"**{item['title']}**\n\n Price: {item['price']} | Rating: ⭐ {item['rating']} | ASIN: `{item['asin']}`")
                else:
                    st.error("Failed to fetch recommendations.")

else:
    st.subheader("Free-Text Search Recommendation")
    user_query = st.text_input("Type a description of what you are looking for:", placeholder="e.g., clear slim phone case for galaxy with shock absorption")
    
    if st.button("Search Recommendations") and user_query:
        with st.spinner("Embedding query & retrieving best matches..."):
            response = requests.post(
                f"{API_BASE_URL}/recommend/query",
                json={"query": user_query, "top_k": top_k}
            )
            if response.status_code == 200:
                data = response.json()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🔤 TF-IDF Matches")
                    for item in data['tfidf_recs']:
                        st.info(f"**{item['title']}**\n\n Price: {item['price']} | Rating: ⭐ {item['rating']} | ASIN: `{item['asin']}`")
                        
                with col2:
                    st.markdown("### 🧠 Transformer Matches")
                    for item in data['transformer_recs']:
                        st.success(f"**{item['title']}**\n\n Price: {item['price']} | Rating: ⭐ {item['rating']} | ASIN: `{item['asin']}`")
            else:
                st.error("Failed to fetch recommendations for query.")