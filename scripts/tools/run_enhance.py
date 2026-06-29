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

# Système GÉNÉRIQUE (Flux, SD, Midjourney…) : ne SORTIR QUE le prompt enrichi.
SYSTEM_GENERIC = (
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
    "models.\n"
    "MEDIUM COHERENCE (very important): lock ONE target medium from the user's "
    "words and keep every added detail consistent with it; never mix vocabulary "
    "from another medium. For a photograph / photorealistic request use only "
    "photographic terms and do NOT add 'oil painting', 'brushstrokes', 'digital "
    "painting', 'concept art', 'illustration', 'cel-shaded' or 'watercolor'. For a "
    "painting / illustration / anime / 3D render use that medium's vocabulary and "
    "do NOT add 'photograph', 'DSLR', '85mm lens', 'photorealistic' or 'film "
    "grain'. If the user explicitly names a medium, treat it as a HARD constraint "
    "never to override.\n"
    "Output ONLY the final enhanced prompt as a single plain-text "
    "paragraph: no preamble, no explanations, no markdown, no quotes, no labels."
)

# Système KREA 2 : dérivé du guide de prompting officiel de Krea (langage naturel,
# prompts longs et détaillés, couleurs nommées, style/medium inféré, texte entre
# guillemets). Le gros encodeur LLM de Krea 2 gère les prompts longs (pas de
# limite 77 tokens comme SDXL).
SYSTEM_KREA2 = (
    "You are an expert prompt engineer for Krea 2, a high-quality text-to-image "
    "model. Rewrite the user's raw idea into a single, richly detailed "
    "image-generation prompt written in natural, descriptive English. Long, "
    "specific, concrete prompts give the best results.\n"
    "Describe, when relevant: the main SUBJECT with its pose, action, gaze and "
    "expression; precise PHYSICAL DETAILS (clothing cut and material, fabric and "
    "surface textures, hair, eyes, skin, accessories); the exact COLORS, named "
    "precisely (e.g. crimson red, muted mint green, azure blue, warm neutral "
    "tones); the BACKGROUND or setting (often a solid striking color or a clearly "
    "described scene); the LIGHTING (e.g. soft directional studio lighting, "
    "high-key, golden hour, diffused natural light, cinematic shafts) and the "
    "mood; the COMPOSITION and CAMERA (angle, framing, shot type, lens, "
    "perspective, depth of field, bokeh); and crucially the MEDIUM and ART STYLE "
    "— infer it from the user's intent and match the medium they actually want; "
    "never impose a style they did not ask for. Krea "
    "excels at many styles: photography, anime, cel animation, digital painting "
    "with visible brushstrokes, flat-color / ligne claire illustration, 3D "
    "render, vintage collage, stippled graphic ink, editorial fashion, and more.\n"
    "CRUCIAL — unusual, surreal or physics-defying ideas: when the request fights "
    "a strong visual prior (a melting solid object, a heavy object floating, "
    "something made of an unexpected material, a hybrid creature, an impossible "
    "scale, etc.), NEVER state it with a single abstract word. Spell out its "
    "concrete VISUAL CONSEQUENCES and reinforce them: name exactly how the form "
    "deforms (sagging, drooping, bending, slumping, oozing, dripping, pooling, "
    "stretching, fracturing, fusing), how the material and surface change, and "
    "what it does to its surroundings. Put this transformation EARLY in the prompt "
    "and restate it once more later so it dominates. When it fits, anchor it in a "
    "relevant art idiom (e.g. surrealism, Salvador Dali-esque, melted-wax "
    "sculpture). The goal is to override the model's default of rendering the "
    "object intact and ordinary.\n"
    "When the user wants a realistic PHOTOGRAPH, make it convincingly "
    "photographic: state the shot type and camera/lens (e.g. 85mm portrait, macro "
    "lens, wide-angle), the aperture / depth of field and bokeh, the lighting "
    "setup (softbox, golden hour, overcast, hard flash), and true-to-life "
    "micro-detail — skin pores and fine texture, fabric weave, material "
    "reflections and roughness, subtle natural imperfections — with natural color "
    "and optional faint film grain, WITHOUT turning it into a digital "
    "illustration.\n"
    "MEDIUM COHERENCE (very important): first lock ONE target medium from the "
    "user's words, then keep EVERY added detail consistent with it and NEVER mix "
    "vocabulary from another medium. If it is a photograph / photorealistic, use "
    "only photographic terms (camera, lens, aperture, lighting setup, real skin "
    "and material texture, optional film grain) and do NOT add words like 'oil "
    "painting', 'brushstrokes', 'digital painting', 'digital art', 'concept art', "
    "'illustration', 'cel-shaded', 'watercolor' or 'render'. If it is a painting, "
    "illustration, anime or 3D render, use that medium's vocabulary and do NOT add "
    "photographic words like 'photograph', 'photo', 'DSLR', '85mm lens', "
    "'photorealistic' or 'film grain'. When the user explicitly names a medium or "
    "style, treat it as a HARD constraint you must never override or dilute.\n"
    "If the user wants any text or lettering to appear in the image, put the exact "
    "words in \"double quotes\". Stay faithful to the user's intent: add detail, "
    "never contradict or replace their idea, and avoid empty words like "
    "'beautiful' or 'amazing' — describe what makes it look good.\n"
    "Output ONLY the final enhanced prompt in English as a single block of "
    "natural-language text: no preamble, no explanations, no markdown, no labels, "
    "and do not wrap the whole prompt in quotes."
)

STYLES = {"generic": SYSTEM_GENERIC, "krea2": SYSTEM_KREA2}


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
    ap.add_argument("--style", default="generic", choices=list(STYLES))
    ap.add_argument("--max-new-tokens", type=int, default=400)
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
        {"role": "system", "content": STYLES.get(args.style, SYSTEM_GENERIC)},
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
