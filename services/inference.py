import torch
from siamese_network.siamese import SiamseNetwork
from PIL import Image
from datasets.data_transformer import test_transform
import asyncio

device = torch.device("cpu")
# model = SiamseNetwork().to(device)
# model.load_state_dict(torch.load("models/siamese_resnet18_v2.pth", map_location=device))
# model.eval()

model = None
def get_model():
    global model

    if model is None:
        model = SiamseNetwork().to(device)
        model.load_state_dict(
            torch.load(
                "models/siamese_resnet18_v2.pth",
                map_location=device
            )
        )
        model.eval()

    return model

def load_image(image: Image.Image):
    image = image.convert("RGB")
    image = test_transform(image)
    image = image.unsqueeze(0)

    return image

def get_embedding(image):
    model = get_model()
    img = load_image(image)
    with torch.no_grad():
        embedding = model.get_embedding(img.to(device))
        return embedding

async def get_embedding_async(image):
    return await asyncio.to_thread(get_embedding, image)