"""Internal TurboSlop nodes: preserve RGBA and use Qwen 2.1's 64 channels.

Loaded only inside the isolated Comfy process. No node editor is exposed.
"""
import hashlib
import os

import numpy as np
import torch
from aiohttp import web
from PIL import Image, ImageOps

import comfy.model_management
import folder_paths
import execution
from server import PromptServer


@PromptServer.instance.routes.get("/turboslop/health")
async def health(request):
    return web.json_response({"token": os.environ.get("TURBOSLOP_COMFY_TOKEN", ""),
                              "schema": 1})


@PromptServer.instance.routes.post("/turboslop/validate")
async def validate(request):
    value = await request.json()
    result = await execution.validate_prompt("turboslop-validation", value["prompt"], None)
    return web.json_response({"valid": result[0], "error": result[1], "node_errors": result[3]})


class LoadRGBA:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": (folder_paths.get_filename_list("input_images")
                 if "input_images" in folder_paths.folder_names_and_paths else
                 sorted(p for p in os.listdir(folder_paths.get_input_directory())
                        if p.lower().endswith(".png")), {"image_upload": True})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"
    CATEGORY = "TurboSlop/internal"

    def load(self, image):
        path = folder_paths.get_annotated_filepath(image)
        with Image.open(path) as source:
            pixels = np.asarray(ImageOps.exif_transpose(source).convert("RGBA"), dtype=np.float32) / 255
        return (torch.from_numpy(pixels.copy())[None].to(
                dtype=comfy.model_management.intermediate_dtype(),
                device=comfy.model_management.intermediate_device()),)

    @classmethod
    def IS_CHANGED(cls, image):
        with open(folder_paths.get_annotated_filepath(image), "rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        return True if folder_paths.exists_annotated_filepath(image) else "Missing local image"


class EmptyQwen21:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"width": ("INT", {"default": 1024, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 32, "max": 8192, "step": 32})}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "create"
    CATEGORY = "TurboSlop/internal"

    def create(self, width, height):
        return ({"samples": torch.zeros((1, 64, height // 16, width // 16),
                 device=comfy.model_management.intermediate_device())},)


NODE_CLASS_MAPPINGS = {"TurboSlopLoadRGBA": LoadRGBA, "TurboSlopEmptyQwen21": EmptyQwen21}
