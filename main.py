from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from pydantic import BaseModel
from recommender import ProductRecommenderEngine



@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender_engine
    recommender_engine = ProductRecommenderEngine(data_dir="data")
    yield

    recommender_engine = None

app = FastAPI(
    title="Hybrid Content-Based Recommendation API",
    description="Dual-Engine E-Commerce Recommender using TF-IDF and Nomic Sentence-Transformers",
    version="1.0.0",
    lifespan=lifespan
)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5



@app.get("/health")
def health_check():
    return {"status": "online", "model_loaded": recommender_engine is not None}

@app.get("/catalog")
async def get_catalog(limit: int = 500):
    return recommender_engine.get_catalog_sample(limit=limit)


@app.get("/recommend/asin/{asin}")
async def recommend_by_asin(asin: str, top_k: int = Query(default=5, ge=1, le=20)):

    result = recommender_engine.recommend_by_asin(asin=asin, top_k=top_k)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/recommend/query")
async def recommend_by_query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail='Search query cannot be empty.')
    return recommender_engine.recommend_by_query(query=request.query, top_k=request.top_k)



