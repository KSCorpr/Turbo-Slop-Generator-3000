"""Internationalisation légère (FR par défaut, EN optionnel).

Principe : les chaînes SOURCE sont en français. En mode anglais, `t()` les
traduit via le dictionnaire `_EN` (clé = texte français). Un shim sur les
constructeurs Gradio traduit automatiquement les libellés/markdown statiques,
ce qui évite de toucher des centaines d'appels. Les chaînes dynamiques
(f-strings) et les listes de choix servant de clés sont traduites explicitement.

La langue est choisie dans Réglages et persistée : elle s'applique au
redémarrage (Gradio construit l'UI une seule fois au lancement).
"""
from __future__ import annotations

from . import settings

_LANG = "fr"


def set_lang(lang: str) -> None:
    global _LANG
    _LANG = "en" if (lang or "").lower().startswith("en") else "fr"


def get_lang() -> str:
    return _LANG


def init_from_prefs() -> str:
    set_lang(settings.load_prefs().get("lang", "fr"))
    return _LANG


def t(s):
    """Traduit une chaîne française vers l'anglais (identité en mode FR)."""
    if _LANG != "en" or not isinstance(s, str):
        return s
    return _EN.get(s, s)


def to_source(s):
    """Retrouve la chaîne française d'origine depuis l'anglais (pour les choix
    de menu servant de clé). Identité en mode FR ou si inconnue."""
    if _LANG != "en" or not isinstance(s, str):
        return s
    return _EN_INV.get(s, s)


# --------------------------------------------------------------------------- #
#  Traduction APRÈS construction : on parcourt l'arbre des composants et on
#  traduit les textes d'affichage en place. AUCUN monkeypatch des constructeurs
#  (qui cassait l'introspection Gradio -> page blanche). Sans effet en mode FR.
# --------------------------------------------------------------------------- #
def translate_blocks(demo) -> None:
    """Traduit les libellés/markdown d'un Blocks déjà construit (mode EN).

    Mute `label`/`info`/`placeholder` sur tous les composants, et `value` sur
    les composants de texte (Markdown/HTML/Button). Ne touche ni aux `choices`
    (traduits explicitement là où ils servent de clé) ni aux `value` des champs
    de saisie (données)."""
    if _LANG != "en":
        return
    try:
        import gradio as gr
        text_value = (gr.Markdown, gr.HTML, gr.Button)
        for comp in list(getattr(demo, "blocks", {}).values()):
            for attr in ("label", "info", "placeholder"):
                v = getattr(comp, attr, None)
                if isinstance(v, str):
                    setattr(comp, attr, t(v))
            if isinstance(comp, text_value):
                v = getattr(comp, "value", None)
                if isinstance(v, str):
                    comp.value = t(v)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
#  Dictionnaire FR -> EN
# --------------------------------------------------------------------------- #
_EN: dict[str, str] = {
    # ---- app.py : entête & avertissements ----
    "Génération d'images locale": "Local image generation",
    "> ⚠️ **Binaire `sd-cli` introuvable.** Lancez "
    "`install.bat` / `install.sh`, ou "
    "`python scripts/get_sdcpp.py`.":
        "> ⚠️ **`sd-cli` binary not found.** Run "
        "`install.bat` / `install.sh`, or "
        "`python scripts/get_sdcpp.py`.",
    "> ⚠️ **Aucun GPU NVIDIA détecté** (mode CPU très lent). "
    "Vérifiez vos pilotes / `nvidia-smi`.":
        "> ⚠️ **No NVIDIA GPU detected** (CPU mode is very slow). "
        "Check your drivers / `nvidia-smi`.",

    # ---- titres d'onglets ----
    "📚 Catalogue de modèles": "📚 Model Catalog",
    "🧰 Toolkit": "🧰 Toolkit",
    "🎬 Vidéo (LTX-2.3)": "🎬 Video (LTX-2.3)",
    "⚙️ Réglages": "⚙️ Settings",

    # ---- generate_tab : statut / mode (dynamiques) ----
    "édition d'image": "image editing",
    "image-to-image": "image-to-image",
    "● modèle prêt": "● model ready",
    "○ à télécharger (onglet Catalogue de modèles)":
        "○ to download (Model Catalog tab)",
    "### {title} — text-to-image & {mode}  ·  {status}":
        "### {title} — text-to-image & {mode}  ·  {status}",
    "✨ Générer ({title})": "✨ Generate ({title})",

    # ---- generate_tab : prompt système / styles ----
    "🎭 Prompt système / style (préfixe, optionnel)":
        "🎭 System prompt / style (prefix, optional)",
    "Appliqué en tête de chaque génération":
        "Prepended to every generation",
    "ex. : style aquarelle, palette pastel, éclairage doux":
        "e.g. watercolor style, pastel palette, soft lighting",
    "Styles enregistrés": "Saved styles",
    "Nom du style à enregistrer": "Name of the style to save",
    "ex. : Aquarelle pastel": "e.g. Pastel watercolor",
    "💾 Enregistrer": "💾 Save",
    "🗑️ Supprimer": "🗑️ Delete",
    "↻ Rafraîchir": "↻ Refresh",

    # ---- generate_tab : prompt ----
    "Prompt": "Prompt",
    "Décrivez l'image…": "Describe the image…",
    "✨ Améliorer le prompt (IA)": "✨ Enhance prompt (AI)",
    "Intensité": "Strength",
    "Léger": "Light",
    "Moyen": "Medium",
    "Fort": "Strong",
    "Prompt négatif": "Negative prompt",

    # ---- generate_tab : améliorateur ----
    "✨ Améliorateur de prompt — installer (1 clic)":
        "✨ Prompt enhancer — install (1 click)",
    "Petit LLM (**Qwen2.5-3B-Instruct**, PyTorch ~6 Go) qui "
    "réécrit votre idée en un prompt **anglais** détaillé "
    "(sujet, lumière, cadrage, style). Chargé puis déchargé à "
    "chaque appel : **aucun conflit de VRAM** avec la "
    "génération. Aucune commande à taper.":
        "Small LLM (**Qwen2.5-3B-Instruct**, PyTorch ~6 GB) that "
        "rewrites your idea into a detailed **English** prompt "
        "(subject, lighting, composition, style). Loaded then unloaded "
        "per call: **no VRAM conflict** with generation. No command to type.",
    "Journal d'installation": "Install log",
    "⬇️ Installer l'améliorateur de prompt": "⬇️ Install the prompt enhancer",

    # ---- generate_tab : référence / img2img ----
    "🖼️ Images de référence (édition d'image)":
        "🖼️ Reference images (image editing)",
    "🖼️ Image de référence / départ (image-to-image)":
        "🖼️ Reference / starting image (image-to-image)",
    "**Éditer une image** : chargez-la et décrivez **la "
    "modification** dans le prompt (ex. *« change la "
    "couleur de la voiture en rouge »*, *« ajoute de la "
    "neige »*). Édition pilotée par le prompt (pas de "
    "curseur de force). Le format de sortie s'adapte à "
    "votre image. Vous pouvez ajouter **2 images de "
    "référence** supplémentaires pour combiner des éléments "
    "(ex. *« mets le personnage de l'image 1 dans le décor "
    "de l'image 2 »*).":
        "**Edit an image**: load it and describe **the change** in the "
        "prompt (e.g. *“change the car color to red”*, *“add snow”*). "
        "Editing is prompt-driven (no strength slider). The output aspect "
        "follows your image. You can add **2 extra reference images** to "
        "combine elements (e.g. *“put the character from image 1 into the "
        "scene of image 2”*).",
    "**Image de référence** (image-to-image) : chargez une "
    "photo, décrivez le rendu voulu, et réglez la **force "
    "de transformation** — **bas (0.2–0.4)** = garde la "
    "structure de la référence ; **haut (0.7–1.0)** = "
    "réinventé. Le format de sortie s'adapte à votre image.":
        "**Reference image** (image-to-image): load a photo, describe the "
        "desired result, and set the **transformation strength** — **low "
        "(0.2–0.4)** = keeps the reference's structure; **high (0.7–1.0)** = "
        "reinvented. The output aspect follows your image.",
    "Image à éditer": "Image to edit",
    "Image de départ": "Starting image",
    "Référence 2 (option)": "Reference 2 (optional)",
    "Référence 3 (option)": "Reference 3 (optional)",
    "Force de transformation": "Transformation strength",
    "🧩 Outpaint — étendre la toile (1.0 = off ; ⚠️ expérimental)":
        "🧩 Outpaint — extend the canvas (1.0 = off; ⚠️ experimental)",
    "Agrandit la toile et laisse le modèle remplir les bords. Décrivez "
    "l'extension dans le prompt.":
        "Enlarges the canvas and lets the model fill the borders. Describe the "
        "extension in the prompt.",

    # ---- generate_tab : LoRA ----
    "🧩 LoRA": "🧩 LoRA",
    "LoRA 1": "LoRA 1",
    "LoRA 2": "LoRA 2",
    "Poids": "Weight",
    "↻ Rafraîchir la liste": "↻ Refresh list",
    "↻ Rafraîchir les modèles": "↻ Refresh models",
    "✖ Vider les LoRA": "✖ Clear LoRAs",
    "Déposez vos fichiers LoRA dans `{dir}`":
        "Drop your LoRA files into `{dir}`",
    "Importer un LoRA Civitai (URL ou ID de version)":
        "Import a Civitai LoRA (URL or version ID)",
    "⬇️ Importer": "⬇️ Import",
    "Collez une URL ou un ID de version Civitai.":
        "Paste a Civitai URL or version ID.",
    "✓ LoRA importé : **{name}** — sélectionnez-le ci-dessus.":
        "✓ LoRA imported: **{name}** — select it above.",

    # ---- generate_tab : fichiers locaux ----
    "📂 Fichiers locaux (modèle perso)": "📂 Local files (custom model)",
    "Pour utiliser un modèle **téléchargé ailleurs** : déposez "
    "le(s) fichier(s) dans `{dir}` puis "
    "sélectionnez-le ci-dessous. Vide = modèle du catalogue.":
        "To use a model **downloaded elsewhere**: drop the file(s) into "
        "`{dir}` then select it below. Empty = catalog model.",
    "Diffusion (local)": "Diffusion (local)",
    "VAE (local)": "VAE (local)",
    "Encodeur (local)": "Encoder (local)",
    "↻ Rafraîchir les fichiers locaux": "↻ Refresh local files",
    "✖ Vider les champs perso": "✖ Clear custom fields",

    # ---- generate_tab : résolution / sampler ----
    "Format (ratio)": "Aspect ratio",
    "Largeur": "Width",
    "Hauteur": "Height",
    "Étapes": "Steps",
    "CFG": "CFG",
    "Sur sd.cpp, CFG désactivé = 1.0 (normal pour les modèles distillés). 0.0 = "
    "pur inconditionnel : peut IGNORER le prompt (la « cfg 0 » de Krea = sa "
    "convention maison, ≠ sd.cpp). >1 = guidage.":
        "On sd.cpp, CFG disabled = 1.0 (normal for distilled models). 0.0 = pure "
        "unconditional: may IGNORE the prompt (Krea's “cfg 0” is its own "
        "convention, ≠ sd.cpp). >1 = guidance.",
    "Préréglage (sampler/scheduler/pas)": "Preset (sampler/scheduler/steps)",
    "Sampler": "Sampler",
    "Scheduler (sigmas)": "Scheduler (sigmas)",
    "Flow shift": "Flow shift",
    "Laissez 0 (auto) : le modèle choisit la bonne valeur "
    "selon la résolution. Une valeur trop basse (1–2) laisse "
    "du GRAIN/bruit en haute résolution ; ~3–4 renforce la "
    "structure.":
        "Leave at 0 (auto): the model picks the right value for the "
        "resolution. Too low (1–2) leaves GRAIN/noise at high resolution; "
        "~3–4 reinforces structure.",
    "Seed (-1 = aléatoire)": "Seed (-1 = random)",
    "Images": "Images",
    "⏹️ Annuler": "⏹️ Cancel",

    # ---- generate_tab : sorties ----
    "Aperçu (temps réel)": "Live preview",
    "Résultats (légende = seed)": "Results (caption = seed)",
    "Aperçu temps réel → résultats (légende = seed)":
        "Live preview → results (caption = seed)",
    "aperçu en cours…": "preview…",
    "Seed de l'image sélectionnée": "Selected image's seed",
    "♻️ Réutiliser ce seed": "♻️ Reuse this seed",
    "📤 Envoyer la sélection vers le Toolkit":
        "📤 Send the selection to the Toolkit",
    "Générez puis sélectionnez une image.":
        "Generate then select an image.",
    "Journal": "Log",

    # ---- generate_tab : erreurs ----
    "Saisissez un prompt (décrivez l'image, ou la modification à appliquer).":
        "Enter a prompt (describe the image, or the change to apply).",

    # ---- ratios (generate) ----
    "Carré 1:1 — 1024×1024": "Square 1:1 — 1024×1024",
    "Carré 1:1 — 1440×1440 (2K)": "Square 1:1 — 1440×1440 (2K)",
    "Carré 1:1 — 1536×1536 (2K)": "Square 1:1 — 1536×1536 (2K)",
    "Paysage 3:2 — 1248×832": "Landscape 3:2 — 1248×832",
    "Portrait 2:3 — 832×1248": "Portrait 2:3 — 832×1248",
    "Paysage 4:3 — 1184×880": "Landscape 4:3 — 1184×880",
    "Portrait 3:4 — 880×1184": "Portrait 3:4 — 880×1184",
    "Large 16:9 — 1392×752": "Wide 16:9 — 1392×752",
    "Vertical 9:16 — 752×1392": "Vertical 9:16 — 752×1392",
    "Cinéma 21:9 — 1568×672": "Cinema 21:9 — 1568×672",
    "Paysage 3:2 — 1216×832": "Landscape 3:2 — 1216×832",
    "Portrait 2:3 — 832×1216": "Portrait 2:3 — 832×1216",
    "Paysage 4:3 — 1152×896": "Landscape 4:3 — 1152×896",
    "Portrait 3:4 — 896×1152": "Portrait 3:4 — 896×1152",
    "Large 16:9 — 1344×768": "Wide 16:9 — 1344×768",
    "Vertical 9:16 — 768×1344": "Vertical 9:16 — 768×1344",
    "Personnalisé (sliders)": "Custom (sliders)",

    # ---- presets (generate) ----
    "⚡ Rapide (recommandé)": "⚡ Fast (recommended)",
    "Équilibré (res ms)": "Balanced (res ms)",
    "Qualité (dpm++ 2M)": "Quality (dpm++ 2M)",
    "⚡ Officiel (8 pas)": "⚡ Official (8 steps)",
    "Très rapide (4 pas)": "Very fast (4 steps)",
    "Qualité (12 pas)": "Quality (12 steps)",

    # ---- library_tab ----
    "### Modèles de base\n"
    "Téléchargement à la demande. La quantification est choisie "
    "automatiquement selon votre VRAM/RAM (modifiable dans Réglages).":
        "### Base models\n"
        "On-demand download. Quantization is chosen automatically from your "
        "VRAM/RAM (changeable in Settings).",
    "Journal des téléchargements": "Download log",
    "> ℹ️ La quantification affichée (Réglages) est une **cible**. Si le "
    "dépôt ne la propose pas, on télécharge le quant disponible le plus "
    "proche **en dessous** (pour tenir dans la VRAM) — c'est indiqué dans "
    "le journal et signalé après le téléchargement.":
        "> ℹ️ The quantization shown (Settings) is a **target**. If the repo "
        "doesn't offer it, the closest available quant **below** it is "
        "downloaded (to fit your VRAM) — shown in the log and flagged after "
        "the download.",
    "Quantification ajustée : le dépôt ne propose pas le quant cible, repli "
    "sur le plus proche disponible (voir le journal).":
        "Quantization adjusted: the repo doesn't offer the target quant, fell "
        "back to the closest available (see the log).",
    "⬇️ Télécharger": "⬇️ Download",
    "↻ Rafraîchir l'état": "↻ Refresh status",
    "● installé": "● installed",
    "○ non installé": "○ not installed",
    "---\n*Les outils (profondeur, détourage, SAM, améliorateur de "
    "prompt) et la vidéo (LTX-2.3) sont dans leurs onglets dédiés.*":
        "---\n*The tools (depth, cutout, SAM, prompt enhancer) and video "
        "(LTX-2.3) are in their own tabs.*",
    "🗑️ « {name} » supprimé : {n} fichier(s) effacé(s).":
        "🗑️ “{name}” deleted: {n} file(s) removed.",
    "Rien à supprimer pour « {name} » (non installé ou fichiers partagés).":
        "Nothing to delete for “{name}” (not installed or shared files).",

    # ---- toolkit_tab ----
    "### Outils utilitaires\n"
    "Carte de **profondeur**, **suppression d'arrière-plan** (PNG "
    "transparent), **détourage d'objet au clic** (Segment Anything) et "
    "**agrandissement ESRGAN** (simple, 100% GPU).":
        "### Utility tools\n"
        "**Depth** map, **background removal** (transparent PNG), "
        "**click-to-cutout** (Segment Anything) and **ESRGAN upscale** "
        "(simple, 100% GPU).",
    "⚙️ Installer {title} (en 1 clic)": "⚙️ Install {title} (1 click)",
    "⬇️ Installer {title}": "⬇️ Install {title}",
    "🌐 Profondeur": "🌐 Depth",
    "*Depth Anything V2* — carte de profondeur (clair = proche, "
    "sombre = loin). Téléchargez le résultat pour le réutiliser.":
        "*Depth Anything V2* — depth map (light = near, dark = far). "
        "Download the result to reuse it.",
    "Depth Anything V2": "Depth Anything V2",
    "Repose sur PyTorch + transformers (~100 Mo de modèle). "
    "Aucune commande à taper.":
        "Uses PyTorch + transformers (~100 MB model). No command to type.",
    "Image source": "Source image",
    "🌐 Générer la profondeur": "🌐 Generate depth",
    "Carte de profondeur": "Depth map",
    "✂️ Sans arrière-plan": "✂️ Background removal",
    "*RMBG-1.4* — détoure le sujet et renvoie un **PNG "
    "transparent**.  \n"
    "⚠️ Modèle sous licence **non commerciale** (BRIA RMBG-1.4).":
        "*RMBG-1.4* — cuts out the subject and returns a **transparent "
        "PNG**.  \n⚠️ **Non-commercial** license (BRIA RMBG-1.4).",
    "Repose sur PyTorch + transformers (~176 Mo de modèle). "
    "Aucune commande à taper.":
        "Uses PyTorch + transformers (~176 MB model). No command to type.",
    "✂️ Détourer": "✂️ Cut out",
    "Sujet détouré (PNG transparent)": "Cutout subject (transparent PNG)",
    "🪄 Détourer un objet (SAM)": "🪄 Cut out an object (SAM)",
    "*Segment Anything* — **cliquez sur un objet** : SAM affiche "
    "aussitôt la **zone sélectionnée en surbrillance**. Ajustez en "
    "recliquant, puis « Extraire » pour le **PNG transparent**.":
        "*Segment Anything* — **click an object**: SAM instantly shows the "
        "**selected area highlighted**. Re-click to adjust, then “Extract” for "
        "the **transparent PNG**.",
    "Zone sélectionnée (aperçu)": "Selected area (preview)",
    "Segment Anything n'est pas installé "
    "(bouton « Installer » ci-dessus).":
        "Segment Anything is not installed (the “Install” button above).",
    "Zone sélectionnée en ({x}, {y}). Cliquez "
    "« Extraire » ou recliquez ailleurs.":
        "Area selected at ({x}, {y}). Click “Extract” or re-click elsewhere.",
    "Segment Anything": "Segment Anything",
    "PyTorch + transformers (~375 Mo, facebook/sam-vit-base). "
    "Aucune commande à taper.":
        "PyTorch + transformers (~375 MB, facebook/sam-vit-base). "
        "No command to type.",
    "Image — cliquez sur l'objet": "Image — click the object",
    "Cliquez un point sur l'image.": "Click a point on the image.",
    "🪄 Extraire l'objet": "🪄 Extract object",
    "Objet extrait (PNG transparent)": "Extracted object (transparent PNG)",
    "Point : ({x}, {y}). Cliquez « Extraire l'objet ».":
        "Point: ({x}, {y}). Click “Extract object”.",
    "Fournissez une image.": "Provide an image.",
    "Cliquez d'abord sur un objet dans l'image.":
        "Click an object in the image first.",

    # ---- toolkit : ESRGAN ----
    "🔼 Agrandir (ESRGAN)": "🔼 Upscale (ESRGAN)",
    "Agrandissement **simple** par réseau ESRGAN GGUF, natif "
    "**sd.cpp** : déterministe, **100% GPU**, aucun PyTorch ni "
    "prompt. Le facteur (×2 ou ×4) dépend du modèle choisi ; "
    "« Répéter » ré-applique le modèle (×2 deux fois = ×4).":
        "**Simple** enlargement with a GGUF ESRGAN network, native "
        "**sd.cpp**: deterministic, **100% GPU**, no PyTorch and no prompt. "
        "The factor (×2 or ×4) depends on the chosen model; “Repeat” "
        "re-applies the model (×2 twice = ×4).",
    "⬇️ Télécharger les upscalers (en 1 clic)":
        "⬇️ Download the upscalers (1 click)",
    "Récupère **tous** les modèles ESRGAN GGUF (~1 Go au "
    "total) depuis `wbruna/upscalers-sdcpp-gguf`. "
    "Réutilisables ensuite hors-ligne.":
        "Fetches **all** the GGUF ESRGAN models (~1 GB total) from "
        "`wbruna/upscalers-sdcpp-gguf`. Reusable offline afterwards.",
    "Journal de téléchargement": "Download log",
    "⬇️ Télécharger les upscalers": "⬇️ Download the upscalers",
    "Image à agrandir": "Image to upscale",
    "Modèle d'upscale (×2 / ×4 selon le nom)":
        "Upscale model (×2 / ×4 by name)",
    "Répétition": "Repeat",
    "🔼 Agrandir": "🔼 Upscale",
    "Résultat (pleine résolution dans outputs/)":
        "Result (full resolution in outputs/)",
    "Choisissez un modèle d'upscale (téléchargez-les d'abord).":
        "Choose an upscale model (download them first).",

    # ---- toolkit : SDXL créatif ----
    "✨ Upscale créatif (SDXL)": "✨ Creative upscale (SDXL)",
    "Upscale **créatif** « Ultimate SD Upscale » : pré-agrandit "
    "puis **raffine tuile par tuile** en SDXL img2img à faible "
    "débruitage (modèle **résident** → tuiles rapides, fondu par "
    "recouvrement). Invente du détail fin façon Magnific. "
    "**100% GPU** (PyTorch).":
        "**Creative** “Ultimate SD Upscale”: pre-enlarges then **refines "
        "tile by tile** with SDXL img2img at low denoise (model **resident** "
        "→ fast tiles, overlap feather blending). Invents fine detail, "
        "Magnific-style. **100% GPU** (PyTorch).",
    "Upscale créatif SDXL": "Creative SDXL upscale",
    "PyTorch + diffusers (~9,5 Go : SDXL base + VAE fp16-fix + "
    "ControlNet Tile). Modèle résident sur le GPU. Aucune "
    "commande à taper.":
        "PyTorch + diffusers (~9.5 GB: SDXL base + VAE fp16-fix + "
        "ControlNet Tile). Model resident on the GPU. No command to type.",
    "🔒 ControlNet Tile (verrouille la structure — "
    "permet de monter la créativité sans dériver)":
        "🔒 ControlNet Tile (locks structure — lets you raise creativity "
        "without drifting)",
    "Fidélité ControlNet (↑ = plus fidèle)":
        "ControlNet fidelity (↑ = more faithful)",
    "> ℹ️ ControlNet **pas encore téléchargé** : "
    "relancez « Installer l'upscale créatif SDXL » "
    "ci-dessus (ajoute ~2,5 Go) puis **redémarrez** "
    "pour activer le verrouillage de structure.":
        "> ℹ️ ControlNet **not downloaded yet**: re-run “Install the creative "
        "SDXL upscale” above (adds ~2.5 GB) then **restart** to enable the "
        "structure lock.",
    "ControlNet pas installé : upscale sans ControlNet. Relancez "
    "l'installateur pour l'activer.":
        "ControlNet not installed: upscaling without it. Re-run the installer "
        "to enable it.",
    "Prompt (optionnel — guide le détail, COURT : ~77 tokens max SDXL ; "
    "inutile de recopier le prompt de génération)":
        "Prompt (optional — guides the detail, KEEP IT SHORT: ~77 tokens max "
        "for SDXL; no need to copy the generation prompt)",
    "Facteur d'agrandissement": "Upscale factor",
    "Jusqu'à ~8K (plafonné à 8192 px). ×6–×8 = "
    "beaucoup de tuiles : très long + ~1–2 Go de RAM.":
        "Up to ~8K (capped at 8192 px). ×6–×8 = many tiles: very slow "
        "+ ~1–2 GB RAM.",
    "Modèle SDXL (déposez vos .safetensors dans "
    "tools_repo/upscale/checkpoints/)":
        "SDXL model (drop your .safetensors into "
        "tools_repo/upscale/checkpoints/)",
    "VAE fp16-fix (externe, recommandé)":
        "VAE fp16-fix (external, recommended)",
    "VAE intégrée au modèle": "Model's built-in VAE",
    "Pré-agrandissement (base avant SDXL)":
        "Pre-upscale (base before SDXL)",
    "Lanczos (par défaut)": "Lanczos (default)",
    "Préréglage de prompt (remplit le prompt + ajuste la créativité)":
        "Prompt preset (fills the prompt + sets creativity)",
    "🔍 Net & fidèle (aucun ajout)": "🔍 Sharp & faithful (no additions)",
    "✨ Ajouter du détail": "✨ Add detail",
    "🧴 Peau réaliste (portrait)": "🧴 Realistic skin (portrait)",
    "🌿 Nature / paysage": "🌿 Nature / landscape",
    "🏙️ Architecture / produit": "🏙️ Architecture / product",
    "🎨 Illustration / peinture": "🎨 Illustration / painting",
    "🚀 Détail maximum (créatif)": "🚀 Maximum detail (creative)",
    "🪶 Doux & propre (anti-grain)": "🪶 Soft & clean (anti-grain)",
    "Créativité (débruitage — ↑ = détail inventé)":
        "Creativity (denoise — ↑ = invented detail)",
    "Pas / tuile": "Steps / tile",
    "Taille de tuile": "Tile size",
    "✨ Upscaler": "✨ Upscale",
    "Aperçu temps réel (pleine résolution dans outputs/)":
        "Live preview (full resolution in outputs/)",
    "Installez d'abord l'upscale créatif SDXL (accordéon ci-dessus).":
        "Install the creative SDXL upscale first (accordion above).",

    # ---- video_tab ----
    "### Génération vidéo — LTX-2.3 (natif sd.cpp)\n"
    "Texte→vidéo, image→vidéo, ou image **début→fin**.  \n"
    "> ⚠️ Modèle **22B** + encodeur Gemma-3-12B : **TRÈS lourd**. Idéal "
    "≥16 Go. Sur 11–12 Go : quant basse (Réglages → `Q3_K`/`Q2_K`) + "
    "offload, et compte **plusieurs minutes** par clip. Commence petit "
    "(640×360, 25 images).":
        "### Video generation — LTX-2.3 (native sd.cpp)\n"
        "Text→video, image→video, or **first→last** frame.  \n"
        "> ⚠️ **22B** model + Gemma-3-12B encoder: **VERY heavy**. Ideal "
        "≥16 GB. On 11–12 GB: low quant (Settings → `Q3_K`/`Q2_K`) + offload, "
        "and expect **several minutes** per clip. Start small "
        "(640×360, 25 frames).",
    "⚙️ Installer LTX-2.3 (en 1 clic)": "⚙️ Install LTX-2.3 (1 click)",
    "Télécharge la diffusion 22B GGUF + l'encodeur Gemma-3-12B + les "
    "VAE vidéo/audio + les connecteurs (**plusieurs Go**, c'est long).":
        "Downloads the 22B GGUF diffusion + Gemma-3-12B encoder + the "
        "video/audio VAEs + connectors (**several GB**, it’s long).",
    "⬇️ Installer LTX-2.3": "⬇️ Install LTX-2.3",
    "Mode": "Mode",
    "Décrivez la scène / le mouvement…": "Describe the scene / motion…",
    "Image (début)": "Image (start)",
    "Image de fin": "End image",
    "Format": "Format",
    "Images (≈ durée × fps)": "Frames (≈ duration × fps)",
    "🎬 Générer la vidéo": "🎬 Generate video",
    "Vidéo": "Video",
    "Saisissez un prompt.": "Enter a prompt.",
    "Fournissez l'image de départ.": "Provide the starting image.",
    "Fournissez l'image de fin.": "Provide the end image.",
    "Paysage 16:9 — 1280×720": "Landscape 16:9 — 1280×720",
    "Paysage 16:9 — 960×544": "Landscape 16:9 — 960×544",
    "Léger 16:9 — 640×360": "Light 16:9 — 640×360",
    "Carré — 768×768": "Square — 768×768",
    "Portrait 9:16 — 720×1280": "Portrait 9:16 — 720×1280",

    # ---- settings_tab ----
    "### Matériel & optimisations": "### Hardware & optimization",
    "Optimisation automatique (selon GPU + RAM)":
        "Automatic optimization (by GPU + RAM)",
    "GPU à utiliser": "GPU to use",
    "#### ⚡ Optimiser pour ma génération de carte (1 clic)\n"
    "Applique un préréglage adapté (quantification + offload + tiling) "
    "calé sur la VRAM réelle de la carte sélectionnée. Désactive "
    "l'optimisation automatique.":
        "#### ⚡ Optimize for my card generation (1 click)\n"
        "Applies a suitable preset (quantization + offload + tiling) keyed to "
        "the selected card’s actual VRAM. Disables automatic optimization.",
    "Quant. diffusion (vide = auto)": "Diffusion quant (empty = auto)",
    "Quant. encodeur (vide = auto)": "Encoder quant (empty = auto)",
    "**Réglages manuels** (utilisés si l'auto est décochée)":
        "**Manual settings** (used when auto is unchecked)",
    "Flash attention": "Flash attention",
    "Offload CPU": "CPU offload",
    "VAE tiling": "VAE tiling",
    "CLIP sur CPU": "CLIP on CPU",
    "VAE sur CPU": "VAE on CPU",
    "Endpoint Hugging Face (miroir éventuel)":
        "Hugging Face endpoint (optional mirror)",
    "#### 🗃️ Accélération par cache (expérimental)\n"
    "Réutilise les calculs quasi identiques entre les pas de diffusion "
    "(doc sd.cpp `caching.md`). Gain surtout au-delà de ~10 pas — sur les "
    "modèles distillés (4–8 pas) le gain est faible et des artefacts sont "
    "possibles. Nécessite un moteur récent (`update-engine.bat`).":
        "#### 🗃️ Cache acceleration (experimental)\n"
        "Reuses near-identical computations across diffusion steps (sd.cpp "
        "`caching.md`). Pays off mostly above ~10 steps — on distilled models "
        "(4–8 steps) the gain is small and artifacts are possible. Requires a "
        "recent engine (`update-engine.bat`).",
    "Désactivé (recommandé)": "Disabled (recommended)",
    "Mode de cache (Flux/Krea = DiT)": "Cache mode (Flux/Krea = DiT)",
    "Option de cache (vide = défauts)": "Cache option (empty = defaults)",
    "Jeton Civitai (optionnel — LoRA protégés)":
        "Civitai token (optional — gated LoRAs)",
    "Langue de l'interface": "Interface language",
    "🎨 Thème (redémarrage requis)": "🎨 Theme (restart required)",
    "Clair": "Light",
    "Sombre": "Dark",
    "✅ Thème enregistré. **Redémarrez l'application** pour l'appliquer.":
        "✅ Theme saved. **Restart the app** to apply it.",
    "#### 🧮 Multi-GPU — carte secondaire dédiée au TEXTE\n"
    "Faites tourner le **texte** (amélioration de prompt + encodage) "
    "sur une 2e carte (ex. 1080 Ti). La **génération d'images** et "
    "l'**upscale SDXL** restent **toujours** sur le GPU de génération "
    "— jamais sur la carte secondaire.":
        "#### 🧮 Multi-GPU — secondary card dedicated to TEXT\n"
        "Run **text** (prompt enhancement + encoding) on a 2nd card (e.g. "
        "1080 Ti). **Image generation** and the **SDXL upscale** always stay "
        "on the generation GPU — never on the secondary card.",
    "GPU pour l'améliorateur de prompt (texte)":
        "GPU for the prompt enhancer (text)",
    "Auto (même que génération)": "Auto (same as generation)",
    "GPU pour l'encodeur de texte (⚠️ expérimental)":
        "GPU for the text encoder (⚠️ experimental)",
    "Désactivé (normal)": "Disabled (normal)",
    "🌐 Langue / Language (redémarrage requis)":
        "🌐 Langue / Language (restart required)",
    "✅ Réglages enregistrés.": "✅ Settings saved.",
    "✅ Optimisé pour **{label}** : diffusion `{quant}`, "
    "encodeur `{enc}` (optimisation auto désactivée).":
        "✅ Optimized for **{label}**: diffusion `{quant}`, encoder `{enc}` "
        "(auto-optimization disabled).",
    "✅ Langue enregistrée. **Redémarrez l'application** "
    "(`run.bat` / `run.sh`) pour appliquer « {lang} ».":
        "✅ Language saved. **Restart the app** (`run.bat` / `run.sh`) to "
        "apply “{lang}”.",
    "Français": "French",
    "English": "English",

    # ---- settings : profil (hardware notes) ----
    "**Profil automatique :**": "**Automatic profile:**",
    "- Diffusion : `{quant}` · Encodeur : `{enc}`":
        "- Diffusion: `{quant}` · Encoder: `{enc}`",
    "- Optimisations : `{flags}`": "- Optimizations: `{flags}`",
    "aucune": "none",
    "RAM système : **{ram} Go**": "System RAM: **{ram} GB**",
    "**GPU détectés :**": "**Detected GPUs:**",
    "⚠️ Aucun GPU NVIDIA détecté · RAM {ram} Go":
        "⚠️ No NVIDIA GPU detected · RAM {ram} GB",
    "tensor cores": "tensor cores",
    "sans tensor cores": "no tensor cores",

    # ---- erreurs moteur fréquentes ----
    "Saisissez d'abord un prompt à améliorer.":
        "Enter a prompt to enhance first.",
    "Générez d'abord une image.": "Generate an image first.",

    # ---- hardware : résumé & notes de profil ----
    "{n} GPU détectés — calcul épinglé sur #{idx} ({name}). "
    "Modifiable dans Réglages.":
        "{n} GPUs detected — compute pinned to #{idx} ({name}). "
        "Changeable in Settings.",
    "Aucun GPU NVIDIA détecté : mode CPU (très lent). "
    "Vérifiez les pilotes / nvidia-smi.":
        "No NVIDIA GPU detected: CPU mode (very slow). "
        "Check drivers / nvidia-smi.",
    "Carte Pascal (GTX 10xx) : flash-attention désactivé "
    "(peu efficace), génération plus lente.":
        "Pascal card (GTX 10xx): flash-attention disabled "
        "(ineffective), slower generation.",
    "VRAM {vram} Go ({arch}) -> diffusion en {quant}.":
        "VRAM {vram} GB ({arch}) -> diffusion in {quant}.",
    "RAM {ram} Go -> encodeur de texte en {enc} "
    "(déchargé en RAM, sans coût VRAM).":
        "RAM {ram} GB -> text encoder in {enc} "
        "(offloaded to RAM, no VRAM cost).",
    "VRAM serrée : préférez des résolutions ≤ 768 px et une "
    "quantification plus basse (la génération sera plus lente).":
        "Tight VRAM: prefer resolutions ≤ 768 px and a lower "
        "quantization (generation will be slower).",
    "VRAM {vram} Go → diffusion {quant}, encodeur {enc}.":
        "VRAM {vram} GB → diffusion {quant}, encoder {enc}.",

    # ---- hardware : notes par génération ----
    "Turing : flash-attention OK, pas d'accélération fp8 "
    "(sd.cpp calcule en fp16). VRAM souvent serrée → quant "
    "légère pour rester rapide.":
        "Turing: flash-attention OK, no fp8 acceleration "
        "(sd.cpp computes in fp16). VRAM often tight → light quant "
        "to stay fast.",
    "Ampere : bf16 natif, bon équilibre. Quant selon la VRAM.":
        "Ampere: native bf16, well balanced. Quant by VRAM.",
    "Ada : très rapide, grande marge VRAM → on monte d'un cran "
    "de qualité.":
        "Ada: very fast, large VRAM headroom → bump up one quality step.",
    "Blackwell : architecture récente + grosse VRAM → qualité "
    "élevée.":
        "Blackwell: recent architecture + large VRAM → high quality.",

    # ---- registry : recommandations (cartes du catalogue) ----
    "✅ adapté à votre carte": "✅ suits your card",
    "⚠️ {min} Go conseillés (vous : {vram})":
        "⚠️ {min} GB recommended (you: {vram})",

    # ---- video : modes ----
    "Texte → vidéo": "Text → video",
    "Image → vidéo": "Image → video",
    "Début → fin": "First → last",

    # ---- app.py : bannière réseau local ----
    "{app} est accessible sur le réseau local !":
        "{app} is reachable on your local network!",
    "Partagez cette adresse à vos collègues (Mac/PC, même Wi-Fi),":
        "Share this address with colleagues (Mac/PC, same Wi-Fi),",
    "à ouvrir dans Safari ou Chrome :": "to open in Safari or Chrome:",
    "(un identifiant/mot de passe leur sera demandé)":
        "(they will be asked for a username/password)",
    "Si l'accès échoue : autorisez le port dans le pare-feu Windows.":
        "If access fails: allow the port in the Windows firewall.",
}

_EN_INV: dict[str, str] = {v: k for k, v in _EN.items()}
