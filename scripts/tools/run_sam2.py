"""SAM 2.1 small: local image segmentation using the Transformers implementation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load(model_dir, device):
    import json
    from transformers import Sam2Config, Sam2Model, Sam2Processor
    raw = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    # The official repo labels its config sam2_video even though this converted
    # checkpoint contains all 357 image-model tensors. Select the image config
    # explicitly, preserving every nested architecture parameter.
    config = Sam2Config(**{key: raw[key] for key in
        ("vision_config", "prompt_encoder_config", "mask_decoder_config", "initializer_range")})
    model, info = Sam2Model.from_pretrained(model_dir, config=config,
        local_files_only=True, output_loading_info=True)
    if info["missing_keys"] or info["unexpected_keys"] or info.get("mismatched_keys"):
        raise RuntimeError("The installed SAM 2 image checkpoint does not match its architecture. Run Install / repair.")
    model = model.to(device).eval()
    processor = Sam2Processor.from_pretrained(model_dir, local_files_only=True)
    return model, processor


def predict(model, processor, image, points, device, embeddings=None):
    import torch
    # [image][object][click][xy]: each probe is a separate object.
    inputs = processor(images=image, input_points=[[[p] for p in points]],
                       input_labels=[[[1] for _ in points]], return_tensors="pt").to(device)
    with torch.inference_mode():
        if embeddings is not None:
            inputs.pop("pixel_values")
            inputs["image_embeddings"] = embeddings
        outputs = model(**inputs, multimask_output=True)
    masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
    scores = outputs.iou_scores.detach().cpu()[0]
    return masks, scores


def auto_masks(model_dir, image, points_per_side, batch, log, iou_max=.75):
    from _device import pick_device, label
    from atelier.engine import masks as M
    import torch
    device = pick_device(torch)
    log(f"[layers] loading SAM 2.1 small on {label(device)}")
    model, processor = load(model_dir, device)
    with torch.inference_mode():
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        embeddings = model.get_image_embeddings(pixel_values)
        del pixel_values
    w, h = image.size
    points = [[int(w*(x+1)/(points_per_side+1)), int(h*(y+1)/(points_per_side+1))]
              for y in range(points_per_side) for x in range(points_per_side)]
    kept, seen = [], []
    for start in range(0, len(points), batch):
        masks, scores = predict(model, processor, image, points[start:start+batch], device, embeddings)
        for candidates, confidence in zip(masks, scores):
            best = int(confidence.argmax())
            if float(confidence[best]) < .80:
                continue
            mask = candidates[best].numpy().astype(bool)
            if not mask.any():
                continue
            grid = M.coverage_grid(mask)
            if any(M.grid_iou(grid, other) > iou_max for other in seen):
                continue
            kept.append(mask)
            seen.append(grid)
        log(f"[layers] SAM 2: {min(start+batch, len(points))}/{len(points)} probes; {len(kept)} regions")
    return kept
