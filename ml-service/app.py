
from fastapi import FastAPI, Query
from model import find_similar_items
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/recommend")
def recommend(title: str = Query(...)):
    return {"results": find_similar_items(title)}
