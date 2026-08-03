#!/usr/bin/env python3
"""Runner d'amélioration de prompt (petit LLM instruct via transformers).

Prend un prompt brut et renvoie UNIQUEMENT un prompt enrichi en anglais, prêt à
injecter dans le champ Prompt. Lancé en sous-process pour ne pas verrouiller les
DLL torch ni occuper la VRAM pendant la génération sd.cpp.
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

# Cœur commun : détection d'intention à partir des mots-clés, puis expansion
# cohérente avec le médium détecté. Partagé par les deux system prompts.
_CORE = (
    "Work in two internal steps and reveal ONLY the final prompt.\n"
    "\n"
    "STEP 1 — DETECT THE INTENT from the user's words (do not output this):\n"
    "• MEDIUM / STYLE — pick exactly ONE from the keywords. Cues: "
    "'photo/photograph/photorealistic/cinematic still/portrait photo' → "
    "photography; 'anime/manga/cel' → anime; 'oil/acrylic/watercolor/gouache/"
    "painting/painterly/brushstrokes' → traditional painting; 'digital painting/"
    "concept art/illustration/artstation' → digital art; '3D/render/octane/"
    "blender/CGI/clay' → 3D render; also pixel art, vector / flat / ligne claire, "
    "line art / ink / sketch, comic, sculpture, product shot, etc. If the user "
    "NAMES a medium it is a HARD constraint. If none is given, choose the best fit "
    "— default to a clean photograph only for real-world people or scenes; "
    "otherwise follow the implied style.\n"
    "• SUBJECT & TYPE (person/portrait, animal, landscape, object/product, "
    "architecture, food, abstract…) with pose, action and expression.\n"
    "• MOOD / GENRE cues (dark, cozy, epic, dreamy, vintage, futuristic, minimal…).\n"
    "• SPECIAL INTENTS: impossible / surreal / physics-defying ideas; any literal "
    "TEXT to render; a requested aspect ratio, shot type or camera.\n"
    "\n"
    "STEP 2 — EXPAND into one rich, natural-English prompt covering, when "
    "relevant: the subject with precise details (materials, textures, hair, eyes, "
    "skin, clothing); the environment / background; the LIGHTING and mood; the "
    "COMPOSITION and camera / framing; the exact COLORS, named precisely (e.g. "
    "crimson red, muted mint green, azure blue); and explicit MEDIUM / style "
    "descriptors.\n"
    "\n"
    "MEDIUM COHERENCE (critical): every added word must match the ONE detected "
    "medium — never mix vocabularies.\n"
    "• Photograph / photorealistic → ONLY photographic terms (camera body, lens "
    "e.g. 85mm / macro / wide-angle, aperture, depth of field, lighting setup, "
    "real skin pores and material texture, optional faint film grain). NEVER add "
    "'oil painting', 'brushstrokes', 'digital painting', 'concept art', "
    "'illustration', 'cel-shaded', 'watercolor' or 'render'.\n"
    "• Painting / illustration / anime / 3D → that medium's vocabulary "
    "(brushstrokes, linework, cel shading, flat colors, render engine…). NEVER add "
    "'photograph', 'photo', 'DSLR', '85mm lens', 'photorealistic' or 'film grain'.\n"
    "\n"
    "EXTRA RULES:\n"
    "• Surreal / impossible ideas: never leave them as one abstract word — spell "
    "out the concrete visual consequences (how the form deforms: melting, sagging, "
    "dripping, oozing, fracturing, fusing…), how materials change, the effect on "
    "surroundings; put it EARLY and restate it once; anchor in a fitting idiom "
    "(e.g. Salvador Dali-esque) when relevant.\n"
    "• Any words the user wants written in the image go in \"double quotes\".\n"
    "• Be concrete, not vague — avoid empty adjectives like 'beautiful' or "
    "'amazing'; show why it looks good. Stay faithful: add detail, never replace "
    "the user's idea, never contradict yourself.\n"
    "\n"
    "OUTPUT: only the final enhanced prompt in English, as a single block of "
    "natural-language text — no preamble, no explanations, no markdown, no labels, "
    "and do not wrap the whole prompt in quotes."
)

# Système KREA 2 : system prompt OFFICIEL de Krea (expansion.txt), adapté pour ne
# SORTIR QUE le paragraphe final (raisonnement gardé interne) afin de l'injecter.
SYSTEM_KREA2 = (
    "You are an expert prompt engineer for text-to-image models. Your task is to "
    "expand the user's prompt into a highly effective image-generation prompt.\n"
    "Reason INTERNALLY (never reveal it) about the request: what is the subject "
    "and mood; what visual styles, mediums and lighting fit (weigh two or three "
    "alternatives and pick the one that best serves the caption); what "
    "composition, framing and grounded details will help the model. Then output "
    "ONLY the final single paragraph.\n"
    "Follow these rules strictly:\n"
    "1. Faithfulness First: preserve all original subjects, actions, colors and "
    "spatial relationships. Do not add new objects, props, characters or animals "
    "unless the user clearly implies them.\n"
    "2. Practical T2I Structure: write a prompt a text-to-image model can parse "
    "cleanly. Group subjects with their own attributes and actions; use grounded "
    "phrasing for poses, interactions and spatial layout.\n"
    "3. Style Planning Stays Internal: use your internal reasoning to choose "
    "style, medium, framing and lighting; do not emit planning tags or wrappers.\n"
    "4. Text Rendering: if the user requests visible text, quotes, labels or "
    "typography, specify the exact text and wrap requested words in quotes.\n"
    "5. Avoid Over-Specification: do not invent highly specific clothing, colors, "
    "materials or scene details unless the input supports them.\n"
    "6. Structure: write one cohesive paragraph. No bullets, JSON or markdown.\n"
    "7. Respect Existing Detail: if the user's prompt is already detailed, lightly "
    "polish and finalize rather than heavily expanding — preserve their phrasing "
    "and direction.\n"
    "8. Preserve User Medium: when the user explicitly requests a medium "
    "('photo of', 'photograph of', 'illustration of', 'painting of', 'sketch of', "
    "'3D render of'), honor it; do not pivot to a different medium to avoid "
    "difficulty — match the user's stated intent.\n"
    "Output ONLY that final paragraph: no preamble, no analysis, no headings, no "
    "markdown, and do not wrap the whole paragraph in quotes."
)

# Système GÉNÉRIQUE (Flux, SDXL, Midjourney…).
SYSTEM_GENERIC = (
    "You are an expert prompt engineer for modern text-to-image models "
    "(Flux, SDXL, Midjourney).\n\n"
    + _CORE
)

STYLES = {"generic": SYSTEM_GENERIC, "krea2": SYSTEM_KREA2}


# Intensité de l'amélioration (ajoutée au system prompt) + budget de tokens.
LEVELS = {
    "light": ("LEVEL: LIGHT — only a light touch-up: keep the user's wording and "
              "length almost intact, just add a few precise quality and detail "
              "descriptors. Do NOT over-expand.", 140),
    "medium": ("LEVEL: MEDIUM — a balanced enhancement: enrich with the key "
               "pillars while staying faithful and reasonably concise.", 320),
    "strong": ("LEVEL: STRONG — a full, rich expansion: develop every relevant "
               "pillar in vivid detail for a long, dense, professional prompt.",
               520),
}


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
    ap.add_argument("--level", default="medium", choices=list(LEVELS))
    ap.add_argument("--max-new-tokens", type=int, default=0)
    ap.add_argument("--variants", type=int, default=1,
                    help="nombre de propositions produites en un chargement")
    ap.add_argument("--style-constraint", default="",
                    help="préfixe de style déjà appliqué à la génération : le "
                         "prompt produit doit rester compatible avec lui")
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

    level_text, level_tokens = LEVELS.get(args.level, LEVELS["medium"])
    max_new = int(args.max_new_tokens) or level_tokens
    system = STYLES.get(args.style, SYSTEM_GENERIC) + "\n\n" + level_text
    n = max(1, min(8, int(args.variants)))
    if n > 1:
        system += ("\n\nYou will be sampled several times for the same idea. "
                   "Commit to ONE clear artistic direction per answer rather "
                   "than hedging, so the proposals differ from each other.")
    # Échantillonnage : assez chaud pour que plusieurs propositions diffèrent,
    # assez sage pour rester fidèle à l'idée de départ.
    temperature, top_p = 0.85, 0.92

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": args.prompt.strip()},
    ]
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(device)
    print(f"[enhance] génération ({args.style}, niveau {args.level}, "
          f"{n} proposition(s))…", flush=True)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new,
                             do_sample=True, temperature=temperature,
                             top_p=top_p, num_return_sequences=n,
                             pad_token_id=tok.eos_token_id)
    start = inputs["input_ids"].shape[1]
    results, seen = [], set()
    for row in out:
        cand = _clean(tok.decode(row[start:], skip_special_tokens=True))
        key = cand.lower()
        if cand and key not in seen:      # doublons possibles
            seen.add(key)
            results.append(cand)
    if not results:
        sys.exit("[enhance] le modèle n'a produit aucun texte exploitable.")

    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Toujours du JSON : un seul format à lire côté application.
    dest.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    print(f"[enhance] {len(results)} proposition(s) écrite(s) : {dest}",
          flush=True)


if __name__ == "__main__":
    main()
