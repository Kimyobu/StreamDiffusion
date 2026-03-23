import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

from utils.wrapper import StreamDiffusionWrapper

import torch

from config import Args
from pydantic import BaseModel, Field
from PIL import Image
import math

# สามารถเปลี่ยนโมเดลตรงนี้ได้ เช่น "KBlueLeaf/Kohaku-V2.1" หรือ "runwayml/stable-diffusion-v1-5"
base_model = "stabilityai/sd-turbo"
taesd_model = "madebyollin/taesd"

# แก้ Prompt ให้ตรงกับเสื้อผ้า/หน้าผม/เพศที่ต้องการเปลี่ยน เช่น "1girl, casual outfit, short hair"
default_prompt = "1girl, beautiful face, masterpiece, highres, casual outfit, short hair, realistic photo"
default_negative_prompt = "black and white, blurry, low resolution, pixelated,  pixel art, low quality, low fidelity"

page_content = """<h1 class="text-3xl font-bold">StreamDiffusion</h1>
<h3 class="text-xl font-bold">Image-to-Image SD-Turbo</h3>
<p class="text-sm">
    This demo showcases
    <a
    href="https://github.com/cumulo-autumn/StreamDiffusion"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">StreamDiffusion
</a>
Image to Image pipeline using
    <a
    href="https://huggingface.co/stabilityai/sd-turbo"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">SD-Turbo</a
    > with a MJPEG stream server.
</p>
"""


class Pipeline:
    class Info(BaseModel):
        name: str = "StreamDiffusion img2img"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        prompt: str = Field(
            default_prompt,
            title="Prompt",
            field="textarea",
            id="prompt",
        )
        # negative_prompt: str = Field(
        #     default_negative_prompt,
        #     title="Negative Prompt",
        #     field="textarea",
        #     id="negative_prompt",
        # )
        width: int = Field(
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        params = self.InputParams()
        self.stream = StreamDiffusionWrapper(
            model_id_or_path=base_model,
            use_tiny_vae=args.taesd,
            device=device,
            dtype=torch_dtype,
            # Denoise Strength: หากต้องการเปลี่ยนรูปภาพใหม่เกือบทั้งหมด (เช่น ทรงผม, หน้า, ชุด, เพศ)
            # ให้เริ่มที่ index ต่ำๆ เช่น [15, 25, 35, 45] จะทำให้เพิ่ม noise เข้าไปเยอะ รูปจะเปลี่ยนไปเป็นคนละคน
            # ถ้า [35, 45] (ค่าเริ่มต้น) จะหมายถึงแก้รูปแค่นิดเดียว
            t_index_list=[15, 25, 35, 45],
            frame_buffer_size=1,
            width=params.width,
            height=params.height,
            # หากเปลี่ยนโมเดลเป็นรุ่นที่ไม่ใช่ turbo (เช่น SD1.5 ทั่วไป) ต้องปรับ use_lcm_lora=True 
            use_lcm_lora=False,
            output_type="pil",
            warmup=10,
            vae_id=None,
            acceleration=args.acceleration,
            mode="img2img",
            use_denoising_batch=True,
            cfg_type="none",
            use_safety_checker=args.safety_checker,
            # enable_similar_image_filter=True,
            # similar_image_filter_threshold=0.98,
            engine_dir=args.engine_dir,
        )

        self.last_prompt = default_prompt
        self.stream.prepare(
            prompt=default_prompt,
            negative_prompt=default_negative_prompt,
            num_inference_steps=50,
            guidance_scale=1.2,
        )

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        image_tensor = self.stream.preprocess_image(params.image)
        output_image = self.stream(image=image_tensor, prompt=params.prompt)

        return output_image
