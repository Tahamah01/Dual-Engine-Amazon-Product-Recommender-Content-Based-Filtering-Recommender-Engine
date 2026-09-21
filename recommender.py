from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import torch
import joblib
import os



class ProductRecommenderEngine:
    def __init__(self, data_dir:str = "data"):
        self.data_dir = data_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        df_path = os.path.join(data_dir, "cbf_data.csv")
        self.df = pd.read_csv(df_path).reset_index(drop=True)
        self.df["metadata"] = self.df["metadata"].fillna("")

        tfidf_path = os.path.join(data_dir, "tfidf_matrix.joblib")
        vectors_path = os.path.join(data_dir, "transformer_vectors.pt")

        print("Loading saved TF-IDF matrix...")
        self.tfidf_matrix, self.tfidf_vectorizer = joblib.load(tfidf_path)

        print("Loading pre-computed transformer vectors...")
        self.transformer_embeddings = torch.load(vectors_path, map_location=self.device)

        print("Loading Nomic Transformer model for real-time query encoding...")
        self.transformer = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True).to(self.device)
        self.transformer.max_seq_length = 1024

        print("======================= Ready ========================")

    def recommend_by_asin(self, asin:str, top_k:int = 5):
        """Item-to-Item Recommendation using an existing ASIN"""
        if asin not in self.df["parent_asin"].values:
            return {"error": f"ASIN: {asin} not found in catalog"}

        idx = self.df.index[self.df['parent_asin'] == asin].to_list()[0]

        # TF-IDF Retrieval
        sim_scores_tfidf = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        top_indices_tfidf = sim_scores_tfidf.argsort()[-(top_k+1):-1][::-1]

        # Transformer Retrieval
        sim_scores_trans = util.cos_sim(self.transformer_embeddings[idx], self.transformer_embeddings)[0]
        top_indices_trans = torch.topk(sim_scores_trans, k=top_k+1).indices[1:].cpu().numpy()

        return {
            "query_type" : "asin",
            "target_asin" : asin,
            "target_title" : self.df.iloc[idx]["title"],
            "tfidf_recs" : self._format_results(top_indices_tfidf),
            "transformer_recs" : self._format_results(top_indices_trans)
        }

    def recommend_by_query(self, query:str, top_k: int = 5) -> dict:
        """Free-Text Query Recommendation by encoding search prompt"""
        query = query.lower().strip()

        # TF-IDF query search
        query_tfidf = self.tfidf_vectorizer.transform([query])
        query_sim_tfidf = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        top_scores_tfidf = query_sim_tfidf.argsort()[-top_k:][::-1]

        # Transformer query search
        query_vector = self.transformer.encode(
            [query], convert_to_tensor=True, prompt_name="query"
        )
        sim_scores_trans = util.cos_sim(query_vector, self.transformer_embeddings)[0]
        top_scores_trans = torch.topk(sim_scores_trans, k=top_k).indices.cpu().numpy()

        return {
            "query_type" : "text_search",
            "target_query" : query,
            "tfidf_recs" : self._format_results(top_scores_tfidf),
            "transformer_recs" : self._format_results(top_scores_trans)
        }


    def _format_results(self, indices) -> list:
        results = []
        for i in indices:
            price = self.df.iloc[i]['price']
            rating = self.df.iloc[i]['average_rating']
            results.append({
                "asin" : self.df.iloc[i]['parent_asin'],
                "title" : self.df.iloc[i]["title"],
                "rating" : f"{rating:.1f}" if rating > 0 else "N/A",
                "price" : "Unknown" if price == -1 or pd.isna(price) else f"${price:.2f}"
            })
        return results

    def get_catalog_sample(self, limit: int = 500) -> list:
        """Returns ASIN-Title pairs for the Streamlit dropdown selector"""
        sample = self.df[['parent_asin', 'title']].head(limit)
        return sample.to_dict(orient="records")