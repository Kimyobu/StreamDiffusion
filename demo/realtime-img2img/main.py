from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Request

import markdown2

import logging
import uuid
import time
from types import SimpleNamespace
import asyncio
import os
import mimetypes
import torch
import glob
import urllib.request
from pydantic import BaseModel
from typing import Optional, Dict

from config import config, Args
from util import pil_to_frame, bytes_to_pil
from connection_manager import ConnectionManager, ServerFullException
from img2img import Pipeline

# fix mime error on windows
mimetypes.add_type("application/javascript", ".js")

THROTTLE = 1.0 / 120
# logging.basicConfig(level=logging.DEBUG)


class App:
    def __init__(self, config: Args, pipeline):
        self.args = config
        self.pipeline = pipeline
        self.app = FastAPI()
        self.conn_manager = ConnectionManager()
        self.init_app()

    def init_app(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.websocket("/api/ws/{user_id}")
        async def websocket_endpoint(user_id: uuid.UUID, websocket: WebSocket):
            try:
                await self.conn_manager.connect(
                    user_id, websocket, self.args.max_queue_size
                )
                await handle_websocket_data(user_id)
            except ServerFullException as e:
                logging.error(f"Server Full: {e}")
            finally:
                await self.conn_manager.disconnect(user_id)
                logging.info(f"User disconnected: {user_id}")

        async def handle_websocket_data(user_id: uuid.UUID):
            if not self.conn_manager.check_user(user_id):
                return HTTPException(status_code=404, detail="User not found")
            last_time = time.time()
            try:
                while True:
                    if (
                        self.args.timeout > 0
                        and time.time() - last_time > self.args.timeout
                    ):
                        await self.conn_manager.send_json(
                            user_id,
                            {
                                "status": "timeout",
                                "message": "Your session has ended",
                            },
                        )
                        await self.conn_manager.disconnect(user_id)
                        return
                    data = await self.conn_manager.receive_json(user_id)
                    if data["status"] == "next_frame":
                        info = pipeline.Info()
                        params = await self.conn_manager.receive_json(user_id)
                        params = pipeline.InputParams(**params)
                        params = SimpleNamespace(**params.dict())
                        if info.input_mode == "image":
                            image_data = await self.conn_manager.receive_bytes(user_id)
                            if len(image_data) == 0:
                                await self.conn_manager.send_json(
                                    user_id, {"status": "send_frame"}
                                )
                                continue
                            params.image = bytes_to_pil(image_data)
                        await self.conn_manager.update_data(user_id, params)

            except Exception as e:
                logging.error(f"Websocket Error: {e}, {user_id} ")
                await self.conn_manager.disconnect(user_id)

        @self.app.get("/api/queue")
        async def get_queue_size():
            queue_size = self.conn_manager.get_user_count()
            return JSONResponse({"queue_size": queue_size})

        @self.app.get("/api/stream/{user_id}")
        async def stream(user_id: uuid.UUID, request: Request):
            try:

                async def generate():
                    while True:
                        last_time = time.time()
                        await self.conn_manager.send_json(
                            user_id, {"status": "send_frame"}
                        )
                        params = await self.conn_manager.get_latest_data(user_id)
                        if params is None:
                            continue
                        image = pipeline.predict(params)
                        if image is None:
                            continue
                        frame = pil_to_frame(image)
                        yield frame
                        if self.args.debug:
                            print(f"Time taken: {time.time() - last_time}")

                return StreamingResponse(
                    generate(),
                    media_type="multipart/x-mixed-replace;boundary=frame",
                    headers={"Cache-Control": "no-cache"},
                )
            except Exception as e:
                logging.error(f"Streaming Error: {e}, {user_id} ")
                return HTTPException(status_code=404, detail="User not found")

        # route to setup frontend
        @self.app.get("/api/settings")
        async def settings():
            info_schema = pipeline.Info.schema()
            info = pipeline.Info()
            if info.page_content:
                page_content = markdown2.markdown(info.page_content)

            input_params = pipeline.InputParams.schema()
            return JSONResponse(
                {
                    "info": info_schema,
                    "input_params": input_params,
                    "max_queue_size": self.args.max_queue_size,
                    "page_content": page_content if info.page_content else "",
                }
            )

        # Model Manager API Schemas
        class LoadModelRequest(BaseModel):
            base_model: str
            lora_dict: Optional[Dict[str, float]] = None
            t_index_list: Optional[list] = None

        class DownloadModelRequest(BaseModel):
            url: str
            model_name: str
            is_lora: bool = False

        # Create directories for models if they don't exist
        os.makedirs("models", exist_ok=True)
        os.makedirs("loras", exist_ok=True)

        @self.app.get("/api/models")
        async def get_models():
            base_models = glob.glob("models/*.safetensors") + glob.glob("models/*.ckpt")
            base_models = [os.path.basename(m) for m in base_models]
            # Add default models
            base_models.extend(["stabilityai/sd-turbo", "runwayml/stable-diffusion-v1-5", "KBlueLeaf/Kohaku-V2.1"])
            
            loras = glob.glob("loras/*.safetensors") + glob.glob("loras/*.ckpt")
            loras = [os.path.basename(l) for l in loras]
            
            return JSONResponse({
                "base_models": list(set(base_models)),
                "loras": loras
            })

        @self.app.post("/api/models/load")
        async def load_model(req: LoadModelRequest):
            try:
                base_model_path = req.base_model
                if base_model_path.endswith(".safetensors") or base_model_path.endswith(".ckpt"):
                    base_model_path = os.path.join("models", base_model_path)
                
                parsed_lora_dict = None
                if req.lora_dict and len(req.lora_dict) > 0:
                    parsed_lora_dict = {}
                    for k, v in req.lora_dict.items():
                        lora_path = os.path.join("loras", k) if (k.endswith(".safetensors") or k.endswith(".ckpt")) else k
                        parsed_lora_dict[lora_path] = v

                if req.t_index_list and len(req.t_index_list) > 0:
                    current_t_index_list = req.t_index_list
                else:
                    # Use current denoise strength
                    current_t_index_list = getattr(self.pipeline, "last_denoise_strength", [15, 25, 35, 45])
                    if isinstance(current_t_index_list, str):
                        current_t_index_list = [int(x.strip()) for x in current_t_index_list.split(",")]

                # Hard Reload using A1111/ComfyUI technique
                self.pipeline.reload_pipeline(
                    model_id=base_model_path,
                    lora_dict=parsed_lora_dict,
                    t_index_list=current_t_index_list
                )
                return JSONResponse({"status": "success", "message": "Model and LoRAs loaded successfully"})
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/models/download")
        async def download_model(req: DownloadModelRequest):
            try:
                dir_path = "loras" if req.is_lora else "models"
                file_ext = ".safetensors" if ".safetensors" in req.url else ".ckpt"
                filename = req.model_name
                if not filename.endswith(".safetensors") and not filename.endswith(".ckpt"):
                    filename += file_ext
                save_path = os.path.join(dir_path, filename)
                
                # Simple synchronous download for now (will block thread, but it's a local demo tool)
                urllib.request.urlretrieve(req.url, save_path)
                
                return JSONResponse({"status": "success", "message": f"Downloaded {filename} to {dir_path}"})
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))

        if not os.path.exists("public"):
            os.makedirs("public")

        self.app.mount(
            "/", StaticFiles(directory="./frontend/public", html=True), name="public"
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16
pipeline = Pipeline(config, device, torch_dtype)
app = App(config, pipeline).app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        ssl_certfile=config.ssl_certfile,
        ssl_keyfile=config.ssl_keyfile,
    )
