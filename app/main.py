import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
import spacy
import torch
from PIL import Image
from torchvision import transforms

from app.bigram_model import BigramModel
from helper_lib.model import get_model
from helper_lib.utils import load_model

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



# ---------- Assignment 2: CNN image classifier (CIFAR-10) ----------
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
cnn_model = load_model(get_model("CNN"), "models/cnn.pth")
cnn_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    x = cnn_transform(image).unsqueeze(0)  # 1 x 3 x 64 x 64
    with torch.no_grad():
        probs = torch.softmax(cnn_model(x), dim=1)[0]
    confidence, index = torch.max(probs, dim=0)
    return {
        "filename": file.filename,
        "predicted_class": CIFAR10_CLASSES[index.item()],
        "confidence": round(confidence.item(), 4),
    }

# ---------- Assignment 3: GAN hand-written digit generator (MNIST) ----------
gan_model = load_model(get_model("GAN"), "models/gan.pth")


class DigitRequest(BaseModel):
    num_samples: int


@app.post("/generate_digit")
def generate_digit(request: DigitRequest):
    with torch.no_grad():
        z = torch.randn(request.num_samples, gan_model.latent_dim)  # random noise
        images = gan_model.generator(z)                              # num_samples x 1 x 28 x 28
        images = (images + 1) / 2                                     # [-1, 1] -> [0, 1]
    return {
        "num_samples": request.num_samples,
        "image_size": [28, 28],
        "images": images.squeeze(1).tolist(),  # tensor -> JSON list, like /embedding
    }