from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy

from app.bigram_model import BigramModel

app = FastAPI(title="SPS GenAI API")

# ---------- Module 3 class activity: bigram text generation ----------
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]
bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


# ---------- Assignment 1: spaCy word embeddings ----------
nlp = spacy.load("en_core_web_lg")  # 300-dim word vectors, loaded once at startup


class EmbeddingRequest(BaseModel):
    word: str


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    doc = nlp(request.word)
    if not doc.has_vector or doc.vector_norm == 0:
        raise HTTPException(status_code=404, detail=f"No embedding found for '{request.word}'")
    vector = doc.vector
    return {
        "word": request.word,
        "dimension": len(vector),
        "embedding": vector.tolist(),  # numpy array -> JSON-serializable list
    }


class SimilarityRequest(BaseModel):
    word1: str
    word2: str


@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    similarity = nlp(request.word1).similarity(nlp(request.word2))
    return {"word1": request.word1, "word2": request.word2, "similarity": float(similarity)}
