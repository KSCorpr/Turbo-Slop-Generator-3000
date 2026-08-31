#!/usr/bin/env python3
"""Runner « image → prompt » : un VLM regarde l'image et écrit un PROMPT.

La différence avec un sous-titrage classique est le cœur du sujet. Un modèle de
vision produit spontanément « une photo d'un chat sur un canapé » : c'est une
LÉGENDE — une phrase sur l'image. Un prompt, lui, est une CONSIGNE : il nomme le
médium, la lumière, l'objectif, la palette, le cadrage, parce que ce sont ces
mots-là qui pilotent un modèle de diffusion. Coller une légende dans le champ
Prompt donne une image vaguement ressemblante et plate.

Le system prompt ci-dessous fait donc trois choses : il interdit les formules de
légende, il impose de nommer le médium détecté (et de n'employer QUE son
vocabulaire — mêler « oil painting » et « 85mm lens » brouille le rendu), et il
force l'anglais quelle que soit la langue de l'interface, parce que c'est la
langue d'entraînement des modèles.

Lancé en sous-process, comme l'améliorateur : le modèle est chargé puis
déchargé, donc aucune VRAM n'est retenue pendant la génération sd.cpp.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _device import label, pick_device, pick_dtype  # noqa: E402

# --------------------------------------------------------------------------- #
#  Règles communes aux trois modes
# --------------------------------------------------------------------------- #
_COMMON = (
    "You write prompts for text-to-image diffusion models. You are shown ONE "
    "image and you answer with the prompt itself — nothing else.\n"
    "\n"
    "HARD RULES:\n"
    "• Answer in ENGLISH, whatever language the request is in. Diffusion models "
    "are trained on English.\n"
    "• NEVER start with caption formulas: no 'This image shows', 'A photo of', "
    "'The picture depicts', 'In this image'. Start directly with the content.\n"
    "• No preamble, no explanation, no quotes, no bullet list, no markdown. One "
    "single flowing block of comma-separated phrases.\n"
    "• Describe ONLY what is visible. Never invent a brand, a place, a name or "
    "a date you cannot actually see.\n"
    "• Name the MEDIUM explicitly (photograph, oil painting, anime cel, 3D "
    "render, vector illustration, pencil sketch…) and then use ONLY that "
    "medium's vocabulary. A photograph gets lens, aperture, depth of field and "
    "real texture; a painting gets brushwork, canvas and pigment. Mixing the "
    "two vocabularies is the single most common way to ruin a prompt.\n"
)

_MODE_FULL = _COMMON + (
    "\n"
    "TASK — write a prompt that would REPRODUCE this image on a model that has "
    "never seen it. Cover, in this order and only when relevant:\n"
    "1. the main subject, precisely (species, age, build, pose, expression, "
    "clothing, materials);\n"
    "2. what surrounds it: setting, background, secondary objects;\n"
    "3. the LIGHT — direction, hardness, colour temperature, time of day;\n"
    "4. the COMPOSITION and framing: shot type, angle, depth of field;\n"
    "5. the COLOURS, named precisely (crimson, muted sage, warm ochre — not "
    "'nice colours');\n"
    "6. the MEDIUM and its style markers.\n"
    "Aim for 60 to 110 words. Dense, concrete, no filler adjectives."
)

_MODE_STYLE = _COMMON + (
    "\n"
    "TASK — extract ONLY THE STYLE, so it can be applied to a completely "
    "different subject.\n"
    "• Say NOTHING about what the picture is of. No subject, no object, no "
    "character, no place. If the image shows a red car in Rome, the words "
    "'car' and 'Rome' must NOT appear.\n"
    "• Describe only: the medium and technique, the light, the colour palette, "
    "the contrast and grain, the level of detail, the framing habits, the era "
    "or movement it evokes.\n"
    "• Write it so that it can be pasted in front of any subject.\n"
    "Aim for 25 to 50 words."
)

_MODE_PLAIN = _COMMON.replace(
    "You write prompts for text-to-image diffusion models. You are shown ONE "
    "image and you answer with the prompt itself — nothing else.",
    "You describe images in plain, factual English.") + (
    "\n"
    "TASK — say what is in the image, plainly, for someone who cannot see it. "
    "Two or three sentences. No prompt vocabulary, no camera settings, no "
    "style jargon — just what is there."
)

MODES = {"full": _MODE_FULL, "style": _MODE_STYLE, "plain": _MODE_PLAIN}

# Le modèle sait lire de très grandes images, mais chaque pixel devient des
# jetons visuels : au-delà de ~1 Mpx on paie beaucoup de temps pour un détail
# qui ne changera pas le prompt. On borne donc le côté long.
MAX_SIDE = 1024

# Amorces de légende que le modèle ressort malgré la consigne. On les coupe
# après coup plutôt que d'espérer : la consigne réduit la fréquence, elle ne la
# met pas à zéro, et une seule occurrence suffit à polluer le champ Prompt.
_LEAD_INS = (
    "this image shows", "this image depicts", "this image features",
    "the image shows", "the image depicts", "the image features",
    "the picture shows", "the picture depicts", "this picture shows",
    "here is a prompt", "here's a prompt", "prompt:", "sure,", "certainly,",
    "in this image,", "the photo shows", "this photograph shows",
)


def _clean(text: str) -> str:
    """Retire les amorces de légende et les guillemets d'encadrement."""
    out = (text or "").strip()
    # Certains modèles encadrent leur réponse de guillemets ou de ```.
    if out.startswith("```"):
        out = out.split("\n", 1)[-1]
        out = out.rsplit("```", 1)[0]
    out = out.strip().strip('"').strip("'").strip()
    changed = True
    while changed:
        changed = False
        low = out.lower()
        for lead in _LEAD_INS:
            if low.startswith(lead):
                out = out[len(lead):].lstrip(" :,-—").lstrip()
                changed = True
                break
    # Une majuscule initiale perdue en coupant l'amorce se rattrape.
    return (out[:1].upper() + out[1:]) if out else out


def _load_image(path: str):
    from PIL import Image
    img = Image.open(path)
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / float(max(w, h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                         Image.LANCZOS)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", default="full", choices=sorted(MODES))
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = pick_device(torch)
    dtype = pick_dtype(torch, device)
    print(f"[image→prompt] chargement du modèle sur {label(device)}…",
          flush=True)
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=dtype).to(device).eval()

    image = _load_image(args.image)
    print(f"[image→prompt] image lue : {image.size[0]}×{image.size[1]}",
          flush=True)

    system = MODES[args.mode]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Write it now. Output the text only."},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = inputs.to(device)

    n = max(1, min(4, int(args.variants or 1)))
    budget = {"full": 320, "style": 160, "plain": 200}[args.mode]
    max_new = int(args.max_new_tokens) or budget
    print(f"[image→prompt] rédaction ({args.mode}, {n} proposition(s))…",
          flush=True)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new,
                             do_sample=True, temperature=0.7, top_p=0.9,
                             num_return_sequences=n)
    start = inputs["input_ids"].shape[1]
    results, seen = [], set()
    for row in out:
        cand = _clean(processor.tokenizer.decode(row[start:],
                                                 skip_special_tokens=True))
        key = cand.lower()
        if cand and key not in seen:
            seen.add(key)
            results.append(cand)
    if not results:
        sys.exit("[image→prompt] le modèle n'a produit aucun texte.")

    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    print(f"[image→prompt] {len(results)} proposition(s) écrite(s) : {dest}",
          flush=True)


if __name__ == "__main__":
    main()
