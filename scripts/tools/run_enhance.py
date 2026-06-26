#!/usr/bin/env python3
"""Runner d'amélioration de prompt (petit LLM instruct via transformers).

Prend un prompt brut et renvoie UNIQUEMENT un prompt enrichi en anglais, prêt à
injecter dans le champ Prompt. Lancé en sous-process pour ne pas verrouiller les
DLL torch ni occuper la VRAM pendant la génération sd.cpp.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

# Système : méthodologie « Prompt Engineer » adaptée pour ne SORTIR QUE le prompt
# enrichi (anglais, dense, propre pour les modèles rapides Turbo/Lightning).
SYSTEM = (
    "You are an expert Prompt Engineer specializing in generative AI art models "
    "(Flux, Krea, Stable Diffusion, Midjourney). Take the user's raw idea and "
    "rewrite it into a single, highly detailed, professional image-generation "
    "prompt. Expand it using these pillars when relevant: (1) core subject & "
    "action with precise details, textures, materials and colors; (2) environment "
    "& setting; (3) lighting & mood with the exact type of light; (4) composition "
    "& camera (angle, framing, lens, depth of field); (5) style & rendering "
    "terms. Be descriptive, not abstract: avoid empty words like 'beautiful' or "
    "'amazing' and instead describe why it looks good. ALWAYS write the final "
    "prompt in English. Keep it dense yet clean, perfect for fast Turbo/Lightning "
    "models. Output ONLY the final enhanced prompt as a single plain-text "
    "paragraph: no preamble, no explanations, no markdown, no quotes, no labels."
)


def _clean(text: str) -> str:
    """Retire un éventuel formatage parasite (guillemets, puces, libellés)."""
    text = (text or "").strip()
    for tag in ("Enhanced Prompt:", "Prompt:", "enhanced prompt:", "prompt:"):
        if text.lower().startswith(tag.lower()):
            text = text[len(tag):].strip()
    text = text.strip().strip("`").strip()
    if len(text) >= 2 and text[0] in "\"'«" and text[-1] in "\"'»":
        text = text[1:-1].strip()
    return " ".join(text.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    args = ap.parse_args()

    import torch
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        sys.exit("transformers manquant. Réinstallez l'outil (« ✨ Améliorer »).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[enhance] chargement du modèle sur {device}…", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=dtype).to(device).eval()

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": args.prompt.strip()},
    ]
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(device)
    print("[enhance] génération…", flush=True)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=int(args.max_new_tokens),
                             do_sample=True, temperature=0.7, top_p=0.9,
                             pad_token_id=tok.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    result = _clean(tok.decode(gen, skip_special_tokens=True))

    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(result, encoding="utf-8")
    print(f"[enhance] prompt enrichi ({len(result)} car.) : {dest}", flush=True)


if __name__ == "__main__":
    main()
