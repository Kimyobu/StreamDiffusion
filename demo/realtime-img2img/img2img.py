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
default_negative_prompt = "black and white, blurry, low resolution, pixelated, pixel art, low quality, low fidelity"

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
        negative_prompt: str = Field(
            default_negative_prompt,
            title="Negative Prompt",
            field="textarea",
            id="negative_prompt",
        )
        denoise_strength: str = Field(
            "15, 25, 35, 45",
            title="Denoise Strength (t_index_list, e.g., '15, 25, 35, 45' for 4 steps)",
            field="textarea",
            id="denoise_strength",
        )
        width: int = Field(
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        self.args = args
        self.device = device
        self.torch_dtype = torch_dtype
        self.params = self.InputParams()
        
        self.init_pipeline(
            model_id=base_model,
            lora_dict=None,
            t_index_list=[15, 25, 35, 45]
        )

    def init_pipeline(self, model_id: str, lora_dict: dict = None, t_index_list: list = [15, 25, 35, 45]):
        use_lcm_lora = "turbo" not in model_id.lower()
        
        self.stream = StreamDiffusionWrapper(
            model_id_or_path=model_id,
            lora_dict=lora_dict,
            use_tiny_vae=self.args.taesd,
            device=self.device,
            dtype=self.torch_dtype,
            t_index_list=t_index_list,
            frame_buffer_size=1,
            width=self.params.width,
            height=self.params.height,
            use_lcm_lora=use_lcm_lora,
            output_type="pil",
            warmup=10,
            vae_id=None,
            acceleration=self.args.acceleration,
            mode="img2img",
            use_denoising_batch=True,
            cfg_type="none",
            use_safety_checker=None,
            engine_dir=self.args.engine_dir,
        )

        self.last_prompt = default_prompt
        self.last_negative_prompt = default_negative_prompt
        self.last_denoise_strength = "15, 25, 35, 45"
        
        self.stream.prepare(
            prompt=self.last_prompt,
            negative_prompt=self.last_negative_prompt,
            num_inference_steps=50,
            guidance_scale=1.2,
        )

    def reload_pipeline(self, model_id: str, lora_dict: dict = None, t_index_list: list = [15, 25, 35, 45]):
        import gc
        if hasattr(self, 'stream'):
            del self.stream
        gc.collect()
        torch.cuda.empty_cache()
        
        self.last_base_model = model_id
        self.last_lora_dict = lora_dict
        self.init_pipeline(model_id, lora_dict, t_index_list)

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        # Check if denoise_strength changed.
        if params.denoise_strength != getattr(self, "last_denoise_strength", "15, 25, 35, 45"):
            try:
                new_t_index_list = [int(x.strip()) for x in params.denoise_strength.split(",")]
                self.reload_pipeline(
                    getattr(self, "last_base_model", base_model),
                    getattr(self, "last_lora_dict", None),
                    new_t_index_list
                )
                self.last_denoise_strength = params.denoise_strength
            except Exception as e:
                print(f"Invalid denoise_strength format: {e}")
                self.last_denoise_strength = params.denoise_strength # Prevent infinite reload attempts
                
        # Check if prompt or negative_prompt changed.
        if params.prompt != self.last_prompt or params.negative_prompt != self.last_negative_prompt:
            self.stream.prepare(
                prompt=params.prompt,
                negative_prompt=params.negative_prompt,
                num_inference_steps=50,
                guidance_scale=1.2,
            )
            self.last_prompt = params.prompt
            self.last_negative_prompt = params.negative_prompt

        image_tensor = self.stream.preprocess_image(params.image)
        output_image = self.stream(image=image_tensor, prompt=params.prompt)

        return output_image
