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

# Choix du back-end de calcul (CUDA / Metal-MPS / CPU), partagé par les runners.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _device import label, pick_device, pick_dtype  # noqa: E402

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

# Système XANAX : l'entrée n'est PAS une description d'image, c'est une phrase
# de la vie courante, écrite à la première personne, souvent en français et
# souvent banale (« j'ai mangé chez Flunch avec Mamie »). Le travail n'est donc
# pas d'enrichir un prompt mais de RÉPONDRE À UNE QUESTION : qu'est-ce qu'on
# verrait sur la photo que quelqu'un aurait prise à ce moment-là ?
#
# Les deux autres system prompts font ici exactement le contraire de ce qu'on
# veut : ils réclament un éclairage travaillé, un objectif nommé, une
# composition, des couleurs choisies — soit la photo réussie d'un photographe,
# alors que tout l'onglet vise la photo ratée d'un oncle. D'où un prompt à part
# entière plutôt qu'une consigne ajoutée aux autres.
#  Consigne « Xanax », telle que fournie par l'auteur du projet. Elle est
#  reproduite mot pour mot ci-dessous : c'est la SPÉCIFICATION, et elle prime.
#
#  Ce qui la suit (« PIPELINE ») n'en fait pas partie et n'est pas une
#  réécriture. Ce sont les quatre choses que la machinerie exige et que la
#  spécification ne couvre pas — chacune correspond à une panne déjà constatée,
#  pas à une préférence. Elles sont séparées et annotées pour qu'on voie tout
#  de suite ce qui vient de qui.
SYSTEM_XANAX = (
    "You are an expert in prompt engineering and image generation.\n"
    "When I write a sentence or a theme, follow these instructions exactly.\n"
    "\n"
    "## Instructions\n"
    "\n"
    "1. Translate and adapt my input into an English generation prompt.\n"
    "2. Generate an image regardless of how short or long my input is.\n"
    "3. NEVER generate a collage, diptych, triptych, split image, or any "
    "composition containing more than one photo within a single image frame.\n"
    "\n"
    "## Visual style (fixed and non-negotiable)\n"
    "\n"
    "Every image must strictly follow this style. It is invariable.\n"
    "\n"
    "- **Aesthetic**: naturalistic, mundane, unadorned photography\n"
    "- **Subject**: everyday life\n"
    "- **Shooting style**: amateur, low-end consumer camera\n"
    "- **Location**: provincial France\n"
    "- **Era**: 1995-2005\n"
    "- **People**: ordinary - attractive, unattractive, or unremarkable\n"
    "- **Atmosphere**: raw daily life, no staging, no composition\n"
    "- **Weather**: always overcast\n"
    "- **Processing**: no grain, no filter, no post-processing\n"
    "- **Aspect ratio**: 4:3\n"
    "\n"
    "## PIPELINE (how this instance is wired)\n"
    "\n"
    "You are the TEXT stage of a two-stage tool. You never draw anything "
    "yourself: instruction 2 means you must always PRODUCE THE PROMPT, however "
    "short the input. A separate image model reads what you write.\n"
    "\n"
    "OUTPUT: the English prompt and nothing else. No preamble, no label, no "
    "markdown, no quotes around it, no explanation. What you type is sent "
    "verbatim to the image model.\n"
    "\n"
    "NEVER add photographic craft. No camera body, no lens or focal length, no "
    "aperture, no depth of field, no lighting setup, no golden hour, no bokeh, "
    "no 'perfectly composed', no 'award-winning', no 'cinematic'. Those words "
    "contradict 'amateur, low-end consumer camera' above, and a prompt model "
    "reaches for them by reflex.\n"
    "\n"
    "TRANSLATE THINGS, NOT NAMES. The image model has never heard of a French "
    "chain or product, so 'adapt' in instruction 1 means describing what the "
    "thing physically looks like: 'j'ai mange chez Flunch' becomes a "
    "self-service cafeteria with plastic trays, fluorescent ceiling lights and "
    "laminate tables; 'des clopes' becomes a pack of cigarettes on a "
    "tobacconist's counter under its red sign. A brand name left as-is draws "
    "nothing.\n"
    "\n"
    "DO NOT RESTATE THE STYLE. The visual style above is appended to your "
    "prompt automatically, word for word. Repeating it doubles its weight and "
    "crowds out the subject. Describe only what is visible at that moment: "
    "who, where, what they are doing.\n"
    "\n"
    "Keep it SHORT - two or three sentences. A long prompt makes the image "
    "model compose and beautify, which is the opposite of the style above."
)


STYLES = {"generic": SYSTEM_GENERIC, "krea2": SYSTEM_KREA2,
          "xanax": SYSTEM_XANAX}


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
                    help="how many suggestions to produce in one model load")
    ap.add_argument("--style-constraint", default="",
                    help="style prefix already applied to the generation: the "
                         "produced prompt must stay compatible with it")
    args = ap.parse_args()

    import torch
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        sys.exit("transformers is missing. Reinstall the tool (“✨ Enhance”).")

    device = pick_device(torch)
    dtype = pick_dtype(torch, device)
    print(f"[enhance] loading the model on {label(device)}…", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=dtype).to(device).eval()

    level_text, level_tokens = LEVELS.get(args.level, LEVELS["medium"])
    max_new = int(args.max_new_tokens) or level_tokens
    system = STYLES.get(args.style, SYSTEM_GENERIC) + "\n\n" + level_text
    if args.style == "xanax":
        # Les niveaux disent « enrichis », « développe chaque pilier » : c'est
        # l'inverse de la consigne Xanax. On ne les ajoute pas, et on coupe
        # court en tokens — un budget large est une invitation à broder.
        system = SYSTEM_XANAX
        max_new = int(args.max_new_tokens) or 160
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
    print(f"[enhance] generating ({args.style}, niveau {args.level}, "
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
        sys.exit("[enhance] the model produced no usable text.")

    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Toujours du JSON : un seul format à lire côté application.
    dest.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    print(f"[enhance] {len(results)} suggestion(s) written: {dest}",
          flush=True)


if __name__ == "__main__":
    main()
