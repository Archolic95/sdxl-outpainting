from fastapi import Depends, FastAPI, Request
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from datetime import datetime
from pydantic import BaseModel
import uvicorn

app = FastAPI()
pipe = AutoPipelineForInpainting.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")

# class OutpaintingRequest(BaseModel):
#     prompt: []
#     images: []
#     positions: []
#     scales: []
#     resolution: ()
class OutpaintingRequest(BaseModel):
    prompt: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/outpainting")
async def generate():
    print('request received')
    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
    
    image = load_image(img_url).resize((1024, 1024))
    mask_image = load_image(mask_url).resize((1024, 1024))
    
    prompt = "a tiger sitting on a park bench"
    #prompt = requsetBody.prompt
    generator = torch.Generator(device="cuda").manual_seed(0)

    image = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=8.0,
        num_inference_steps=20,  # steps between 15 and 30 work well for us
        strength=0.99,  # make sure to use `strength` below 1.0
        generator=generator,
    ).images[0]

    time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    image_file_name = f'outputs/sdxl_image_{time_stamp}.png'

    cv2.imwrite(image_file_name, np.array(image))

    with open(image_file_name, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        req_file = encoded_image.decode('utf-8')

    return {"image": req_file}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30000)