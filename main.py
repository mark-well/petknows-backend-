from fastapi import FastAPI, UploadFile
from PIL import Image
import io
from services.inference import get_embedding_async
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import torch
import numpy as np
import ast
from supabase_client import get_all_pets

load_dotenv()
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "").split(",")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],    
)

@app.get("/")
async def root():
    return {"message": "Server is running!"}

@app.post("/get_embedding")
async def get_embedding_endpoint(file: UploadFile) -> object:
    image_contents = await file.read()
    image = Image.open(io.BytesIO(image_contents))
    embedding = await get_embedding_async(image)
    return {
        "embedding": embedding.squeeze().cpu().tolist()
    }

@app.post("/identify")
async def identify_pet(file: UploadFile):
    THRESHOLD = 0.70
    image_contents = await file.read()
    image = Image.open(io.BytesIO(image_contents))
    current_pet_embedding = await get_embedding_async(image)

    pets = get_all_pets().data
    results = []

    for pet in pets:
        pet_embedding = torch.from_numpy(np.array(ast.literal_eval(pet["embedding"]), dtype=np.float32))
        similarity = torch.nn.functional.cosine_similarity(current_pet_embedding, pet_embedding.unsqueeze(0)).item()

        results.append({
            "pet": pet,
            "similarity": similarity
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    results = [result for result in results if result["similarity"] >= THRESHOLD]
    top_matches = results[:3]

    if not top_matches:
        return {
            "found": False,
            "message": "Pet not found."
        }

    return {
            "found": True,
            "message": "Pet found.",
            "top_matches": top_matches
        }