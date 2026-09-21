# Dual-Engine Amazon Product Recommendation System

An end-to-end, production-grade Content-Based Filtering (CBF) recommendation engine built on the Amazon Product Reviews dataset (Cell Phones & Accessories). 

This project implements and compares two distinct recommendation strategies:
1. **Sparse Baseline (TF-IDF):** Fast, exact keyword matching using term frequency-inverse document frequency and cosine similarity.
2. **Dense Semantic Embeddings (Nomic Transformer):** High-dimensional vector search using `nomic-ai/nomic-embed-text-v1.5` to capture deep contextual and semantic relationships across long product descriptions (up to 1,024 tokens).

The system is architected around an **Offline Embedding Pipeline** and an **Online Inference Microservice** built with FastAPI and Streamlit.

---

## 🏗 Architecture & System Design

```text
                               OFFLINE PIPELINE (Kaggle / Colab)
┌───────────────────────┐    ┌─────────────────────────────────┐    ┌───────────────────────────────┐
│ Amazon Metadata CSV   │ ─► │ Cleaning & Feature Engineering  │ ─► │ Metadata String Concatenation │
└───────────────────────┘    └─────────────────────────────────┘    └───────────────┬───────────────┘
                                                                                    │
                                             ┌──────────────────────────────────────┴──────────────────────────────────────┐
                                             ▼                                                                             ▼
                                 ┌─────────────────────────┐                                                   ┌─────────────────────────┐
                                 │ TF-IDF Vectorizer Fit   │                                                   │ Nomic Model Embeddings  │
                                 └───────────┬─────────────┘                                                   └───────────┬─────────────┘
                                             │                                                                             │
                                             ▼                                                                             ▼
                                 ┌─────────────────────────┐                                                   ┌─────────────────────────┐
                                 │  tfidf_matrix.joblib    │                                                   │  transformer_vectors.pt │
                                 └───────────┬─────────────┘                                                   └───────────┬─────────────┘
                                             │                                                                             │
─────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┼──────────────────────────────
                                             │                 ONLINE INFERENCE (Local Microservices)                      │
                                             └──────────────────────────────────────┬──────────────────────────────────────┘
                                                                                    │
                                                                                    ▼
                                                                     ┌────────────────────────────┐
                                                                     │ ProductionRecommender Engine│
                                                                     └──────────────┬─────────────┘
                                                                                    │
                                                                                    ▼
                                                                     ┌────────────────────────────┐
                                                                     │   FastAPI Backend (8000)   │
                                                                     └──────────────┬─────────────┘
                                                                                    │  REST API
                                                                                    ▼
                                                                     ┌────────────────────────────┐
                                                                     │   Streamlit Web UI (8501)  │
                                                                     └────────────────────────────┘
```

---

## ✨ Key Features

* **Dual-Engine Comparison:** Side-by-side benchmark of traditional lexical matching vs. modern deep semantic embeddings.
* **Item-to-Item Recommendations:** Input an Amazon ASIN to instantly retrieve top K similar products.
* **Free-Text Semantic Search:** Vectorizes arbitrary user queries on the fly (*e.g., "waterproof heavy-duty case for galaxy"*) using Nomic query prompts.
* **Sub-Millisecond Inference:** Pre-computed PyTorch vectors and TF-IDF sparse matrices load directly into RAM on startup, avoiding runtime re-embedding.
* **Decoupled Architecture:** Asynchronous FastAPI REST server with `lifespan` state management, completely separated from the Streamlit frontend.

---

## 📂 Project Structure

```text
amazon_recommender/
│
├── data/
│   ├── df_final.csv             # Pre-processed Amazon product catalog
│   ├── tfidf_matrix.joblib      # Tuple: (TF-IDF Sparse Matrix, Fitted Vectorizer)
│   └── transformer_vectors.pt   # Pre-computed PyTorch dense embeddings tensor
│
├── 01_CBF_data_cleaning.ipynb   # Data cleaning notebook
├── .gitignore                   # Git exclusion rules
├── .python-version              # Python version target
├── app.py                       # Streamlit web application frontend
├── main.py                      # FastAPI REST API backend
├── recommender.py               # Production inference engine class
└── requirements.txt             # Environment dependencies
```

---

## 🛠 Tech Stack

* **Language:** Python 3.10+
* **Machine Learning:** PyTorch, `sentence-transformers`, `scikit-learn`
* **Embeddings Model:** `nomic-ai/nomic-embed-text-v1.5`
* **Backend API:** FastAPI, Uvicorn, Pydantic
* **Frontend:** Streamlit

---

## 🚀 Setup & Installation Instructions

### Prerequisites
* Python 3.10 or higher
* Git

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/amazon-recommendation-engine.git
cd amazon-recommendation-engine
```

### 2. Set Up Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🖥 Running the Application

To run the complete application, start both microservices in separate terminal windows:

### Terminal 1: Launch FastAPI Backend
```bash
uvicorn main:app --reload --port 8000
```
* The API will initialize on `http://127.0.0.1:8000`.
* Interactive OpenAPI (Swagger) docs are available at `http://127.0.0.1:8000/docs`.

### Terminal 2: Launch Streamlit Frontend
```bash
streamlit run app.py
```
* The web interface will open automatically in your browser at `http://localhost:8501`.

---

## 📖 API Documentation

### Base URL
`http://127.0.0.1:8000`

### Endpoints

| Method | Endpoint | Description | Query / Body Parameters |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Health check & model status | None |
| `GET` | `/catalog` | Fetch product list for UI dropdowns | `limit` (int, default: 500) |
| `GET` | `/recommend/asin/{asin}` | Item-to-item recommendations | `top_k` (int, 1-20, default: 5) |
| `POST` | `/recommend/query` | Free-text search recommendations | `{"query": "string", "top_k": 5}` |

---

### Sample Request & Response

#### Request (`POST /recommend/query`)
```json
{
  "query": "shockproof clear protective case for galaxy s21",
  "top_k": 2
}
```

#### Response (`200 OK`)
```json
{
  "target_query": "shockproof clear protective case for galaxy s21",
  "TFIDF_recs": [
    {
      "asin": "B08X1Y8Z99",
      "title": "Clear Shockproof Case Compatible with Samsung Galaxy S21",
      "rating": "4.5",
      "price": "$12.99"
    },
    {
      "asin": "B07Z8P4M11",
      "title": "Galaxy S21 Heavy Duty Protection Phone Cover",
      "rating": "4.2",
      "price": "$15.49"
    }
  ],
  "Transformer_recs": [
    {
      "asin": "B08X1Y8Z99",
      "title": "Clear Shockproof Case Compatible with Samsung Galaxy S21",
      "rating": "4.5",
      "price": "$12.99"
    },
    {
      "asin": "B091K2L3M4",
      "title": "Ultra-Slim Transparent Anti-Drop Armor Case for Galaxy S21",
      "rating": "4.7",
      "price": "$11.99"
    }
  ]
}
```

---

## 🧹 Data Cleaning & Engineering Summary

1. **Category Isolation:** Focused on the `Cell Phones & Accessories` category (~10,000 products).
2. **Missing Data Strategy:** Imputed missing text fields with empty strings; missing prices assigned `-1.0` to avoid skewed distributions during filtering.
3. **Structured Text Fusion:** Merged `store`, `categories`, `title`, `features`, and `description` into a normalized, unified `metadata` document per product.
4. **Task-Specific Prompting:** Integrated Nomic's required `document` and `query` task prefixes to maintain vector alignment during asymmetric retrieval.
