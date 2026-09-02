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
    "> ⚠️ **Aucun GPU détecté** (mode CPU très lent). "
    "Sur PC, vérifiez vos pilotes NVIDIA / `nvidia-smi`.":
        "> ⚠️ **No GPU detected** (CPU mode is very slow). "
        "On PC, check your NVIDIA drivers / `nvidia-smi`.",

    # ---- titres d'onglets ----
    "📚 Catalogue de modèles": "📚 Model Catalog",
    "🧰 Toolkit": "🧰 Toolkit",
    "Votre `sd-cli` est trop ancien pour « {name} » (option {opt} inconnue). Mettez le moteur à jour avec **update-engine.bat** — inutile de télécharger les poids avant.":
        "Your `sd-cli` is too old for “{name}” (unknown option {opt}). Update the engine with **update-engine.bat** — no need to download the weights first.",
    "⬇️ **« {name} » n'est pas installé** — il manque : {parts}.  \nTéléchargez-le depuis l'onglet **📚 Catalogue de modèles** (~{gb} Go, c'est long).":
        "⬇️ **“{name}” is not installed** — missing: {parts}.  \nDownload it from the **📚 Model Catalog** tab (~{gb} GB, it takes a while).",
    "encodeur de prompt": "prompt encoder",
    "Fournissez au moins une image de référence.":
        "Provide at least one reference image.",
    "Binaire `sd-cli` introuvable. Lancez `install.bat` (ou `python scripts/get_sdcpp.py`).":
        "`sd-cli` binary not found. Run `install.bat` (or `python scripts/get_sdcpp.py`).",
    "modèle de diffusion": "diffusion model",
    "⚙️ Réglages": "⚙️ Settings",

    # ---- toolkit : décomposition en calques (PSD) ----
    "🧩 Calques": "🧩 Layers",
    "🌱 Restaurer": "🌱 Restore",
    "Image à décomposer": "Image to decompose",
    "Mode": "Mode",
    "Automatique — SAM balaie l'image": "Automatic — SAM sweeps the image",
    "Manuel — je clique les zones": "Manual — I click the areas",
    "Finesse du balayage (points par côté)":
        "Sweep density (points per side)",
    "↑ = plus de zones trouvées, et beaucoup plus long. 12 est un bon départ.":
        "↑ = more areas found, and much slower. 12 is a good starting point.",
    "Surface minimale d'un calque (% de l'image)":
        "Minimum layer area (% of the image)",
    "Monter cette valeur est le meilleur moyen d'éviter la soupe de petits "
    "calques.":
        "Raising this is the best way to avoid a soup of tiny layers.",
    "Nombre maximum de calques": "Maximum number of layers",
    "Fichier PSD": "PSD file",
    "PNG transparents séparés": "Separate transparent PNGs",
    "🧩 Décomposer": "🧩 Decompose",
    "Zones retenues (aperçu)": "Selected areas (preview)",
    "Fichiers produits": "Files produced",
    "Choisissez au moins un format de sortie (PSD ou PNG).":
        "Choose at least one output format (PSD or PNG).",
    "**Cliquez un objet** dans l'image ci-dessus : il devient un calque. "
    "Recliquez ailleurs pour en ajouter d'autres.":
        "**Click an object** in the image above: it becomes a layer. Click "
        "elsewhere to add more.",
    "*Aucune zone choisie.*": "*No area selected.*",
    "↩️ Retirer la dernière": "↩️ Remove the last one",
    "🗑️ Tout effacer": "🗑️ Clear all",
    "Fond (image complète)": "Background (full image)",
    "CLIP (compréhension des zones)": "CLIP (zone understanding)",
    "Même add-on que « Détourer un objet ». Indispensable.":
        "Same add-on as “Cut out an object”. Required.",
    "Les zones sont **nettoyées** avant d'être posées : trous "
    "intérieurs bouchés, morceaux épars séparés en zones "
    "distinctes, miettes écartées, bords adoucis. Et les calques "
    "sont **disjoints** — les afficher tous redonne exactement "
    "l'image d'origine, aucun pixel n'est peint deux fois.":
        "Zones are **cleaned up** before being placed: interior holes "
        "filled, scattered pieces split into distinct zones, crumbs "
        "discarded, edges feathered. And the layers are **disjoint** — "
        "showing them all reproduces the source image exactly, no pixel "
        "is painted twice.",

    # ---- réglages : exécution segmentée ----
    "Budget VRAM du graphe": "Graph VRAM budget",
    "Désactivé (recommandé pour la génération)":
        "Disabled (recommended for generation)",
    "Auto — VRAM libre moins 1 Go": "Auto — free VRAM minus 1 GB",
    "Plafond ferme : 6 Go": "Hard cap: 6 GB",
    "Plafond ferme : 8 Go": "Hard cap: 8 GB",
    "Plafond ferme : 10 Go": "Hard cap: 10 GB",
    "Valeur libre acceptée : « 6 », ou « cuda0=6,cuda1=4 » sur une machine "
    "multi-cartes.":
        "Free-form values accepted: “6”, or “cuda0=6,cuda1=4” on a multi-GPU "
        "machine.",
    "Streaming des couches (sans effet sans budget)":
        "Layer streaming (no effect without a budget)",
    "Précharge les couches à la demande. Encore plus dépendant du PCIe : à "
    "n'essayer que si le budget seul ne suffit pas.":
        "Prefetches layers on demand. Even more PCIe-dependent: only worth "
        "trying if the budget alone is not enough.",

    # ---- groupes d'onglets racine ----
    "🧰 Outils": "🧰 Tools",
    "⚙️ Système": "⚙️ System",
    "mode CPU": "CPU mode",
    "**Binaire `sd-cli` introuvable** — lancez `install.bat` / `install.sh`.":
        "**`sd-cli` binary not found** — run `install.bat` / `install.sh`.",
    "**Aucun GPU détecté** — mode CPU (très lent). Vérifiez vos pilotes "
    "NVIDIA / `nvidia-smi`.":
        "**No GPU detected** — CPU mode (very slow). Check your NVIDIA "
        "drivers / `nvidia-smi`.",

    # ---- onglet Xanax (style figé) ----
    "💊 Xanax": "💊 Xanax",
    "Modèle": "Model",
    "Modèle indisponible.": "Model unavailable.",
    "### Racontez votre journée, on en fait une photo\n"
    "N'écrivez **pas une description d'image** mais une phrase de la "
    "vie courante, comme dans un carnet : *« j'ai mangé chez Flunch "
    "avec Mamie »*, *« journée pas terrible mais j'ai pu aller acheter "
    "des clopes »*. C'est ce décalage qui donne la photo prise en "
    "passant plutôt que la photo posée.\n\n"
    "**Le style est figé et non modifiable** : photo amateur, France "
    "provinciale, 1995-2005, temps couvert, aucun post-traitement, "
    "format 4:3 sur la grille native du modèle. C'est le principe de "
    "cet onglet — pour régler quoi que ce soit, utilisez un onglet de "
    "génération normal.":
        "### Tell us about your day, we make a photo of it\n"
        "Do **not** write an image description — write a plain sentence about "
        "your day, the way you would in a diary: *“had lunch at the cafeteria "
        "with Gran”*, *“rubbish day but at least I got my cigarettes”*. That "
        "gap is what produces a photo taken in passing rather than a posed "
        "one.\n\n"
        "**The style is fixed and cannot be changed**: amateur snapshot, "
        "provincial France, 1995-2005, overcast, no post-processing, 4:3 on "
        "the model's native grid. That is the point of this tab — to tune "
        "anything at all, use a normal generation tab.",
    "Ce que vous avez fait": "What you did",
    "j'ai mangé chez Flunch avec Mamie…":
        "had lunch at the motorway cafeteria with Gran…",
    "Une phrase de votre journée, à la première personne. Pas « un homme attend le bus » mais « j'ai attendu le bus une plombe ».":
        "One sentence about your day, in the first person. Not “a man waits for the bus” but “waited ages for the bus”.",
    "🎲 Une journée au hasard": "🎲 A random day",
    "📷 Générer": "📷 Generate",
    "Racontez d'abord quelque chose — une phrase suffit.":
        "Tell us something first — one sentence is enough.",
    "✨ Transformer ma phrase en photo (améliorateur IA)":
        "✨ Turn my sentence into a photo (AI enhancer)",
    "Cherche ce qu'on VERRAIT sur la photo : le lieu, les gens, l'heure. Traduit au passage, et sait ce qu'est un Flunch — le modèle d'image, non.":
        "Works out what the photo would SHOW: the place, the people, the time of day. Translates along the way, and knows what a French cafeteria chain is — the image model does not.",
    "Améliorateur non installé — installez-le depuis un onglet de génération. Sans lui votre phrase part TELLE QUELLE : écrivez alors en anglais et dites ce qu'on voit, pas ce que vous avez fait.":
        "Enhancer not installed — install it from a generation tab. Without it your sentence is sent AS IS: write in English then, and say what is visible rather than what you did.",
    "Une graine fixe rejoue exactement la même photo.":
        "A fixed seed replays exactly the same photo.",
    "Photo": "Photo",
    "⏳ On cherche à quoi ressemblait ce moment…":
        "⏳ Working out what that moment looked like…",

    # ---- onglet Xanax : banque d'anecdotes (bouton 🎲) ----
    # Traduites, mais gardées dans leur décor : le style impose la France
    # provinciale, une anecdote délocalisée ne collerait pas à l'image.
    "j'ai mangé chez Flunch avec Mamie":
        "had lunch at the motorway cafeteria with Gran",
    "journée pas terrible mais j'ai pu aller acheter des clopes":
        "rubbish day but at least I got my cigarettes",
    "on a fait les courses au Leclerc, y'avait la queue à la caisse":
        "did the shopping at the hypermarket, long queue at the till",
    "anniversaire de Papy, on était tous dans la véranda":
        "Grandad's birthday, we were all crammed in the conservatory",
    "j'ai attendu le bus vingt minutes sous la pluie":
        "waited twenty minutes for the bus in the rain",
    "réveillon chez ma tante, on a mangé de la bûche":
        "New Year's Eve at my aunt's, we had the yule log",
    "j'ai lavé la voiture dans l'allée": "washed the car in the driveway",
    "on est allés à la kermesse de l'école de mon fils":
        "went to the school fête with my son",
    "j'ai poireauté à la CAF toute la matinée":
        "hung around the benefits office all morning",
    "barbecue chez les voisins, il a commencé à pleuvoir":
        "barbecue at the neighbours', then it started raining",
    "j'ai repeint la chambre, c'est pas fini":
        "repainted the bedroom, not finished yet",
    "on a mangé au routier sur la nationale":
        "ate at the truck stop on the main road",
    "j'ai emmené le chien chez le véto": "took the dog to the vet",
    "communion de ma cousine, photo devant l'église":
        "my cousin's communion, photo outside the church",
    "on a bu un café au bar-tabac après le marché":
        "had a coffee at the corner café after the market",
    "j'ai déménagé le canapé de ma sœur": "moved my sister's sofa",
    "vide-grenier dimanche matin, j'ai rien vendu":
        "car boot sale on Sunday morning, sold nothing",
    "on a regardé le match chez Kévin": "watched the match round at Kev's",
    "j'ai passé l'après-midi à la laverie":
        "spent the afternoon at the launderette",
    "on est allés voir la mer, il faisait gris":
        "went to see the sea, it was grey",
    "j'ai monté le meuble Ikea de la cuisine":
        "assembled the flat-pack kitchen unit",
    "pot de départ au boulot dans la salle de pause":
        "leaving drinks at work in the break room",
    "j'ai fait la queue à la poste pour un colis":
        "queued at the post office for a parcel",
    "on a pris l'apéro dans le jardin, rien de spécial":
        "had drinks in the garden, nothing special",
    "j'ai gagné trois euros au PMU": "won three euros at the betting shop",
    "on a mangé une pizza devant la télé":
        "had pizza in front of the telly",
    "j'ai attendu ma fille à la sortie du collège":
        "waited for my daughter outside the school gates",
    "on a fait une pause sur l'aire d'autoroute":
        "stopped at the motorway services",
    "j'ai réparé le vélo dans le garage": "fixed the bike in the garage",
    "mariage de mon collègue, salle des fêtes":
        "a colleague's wedding, in the village hall",
    "j'ai tondu la pelouse avant qu'il pleuve":
        "mowed the lawn before the rain",
    "on a fêté ça au kebab en bas de chez moi":
        "celebrated at the kebab shop downstairs",
    "✅ Photo générée (seed {s})": "✅ Photo generated (seed {s})",
    "⬆️ Mettre à jour le binaire (sans les modèles)":
        "⬆️ Update the binary (models untouched)",
    "⏳ Téléchargement du binaire trellis le plus récent…":
        "⏳ Downloading the latest trellis binary…",
    "⏹️ Arrêt du serveur résident (il verrouille le binaire)…":
        "⏹️ Stopping the resident server (it locks the binary)…",

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
    "🪄 Détourer (SAM)": "🪄 Cut out (SAM)",
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

    # ---- toolkit : HD natif sd.cpp ----
    "🚀 HD": "🚀 HD",
    "**Passe HD native de sd.cpp** : l'image est agrandie puis "
    "**re-débruitée en entier** par votre modèle de génération "
    "(Krea 2, Flux.2). Le tout en **une seule commande**, "
    "100% GPU, sans PyTorch.\n\n"
    "Deux différences de fond avec l'upscale créatif SDXL :\n"
    "- **aucun découpage en tuiles** — le second passage voit "
    "l'image entière, donc il n'y a pas de couture *possible*, "
    "et pas d'incohérence entre deux carrés voisins ;\n"
    "- **c'est votre modèle** qui redessine, pas un SDXL de "
    "2023 : le détail ajouté reste dans le style que le modèle "
    "connaît déjà.\n\n"
    "En échange, **c'est gourmand en VRAM** : refuser les "
    "tuiles a un prix. Le second passage réserve un tampon "
    "proportionnel au nombre de pixels, **qui s'ajoute aux "
    "poids du modèle** déjà sur la carte. Le facteur est donc "
    "budgété selon votre VRAM et la taille de votre modèle, et "
    "réduit tout seul si ça ne tient pas — le journal annonce "
    "la valeur retenue. Sur 11–12 Go avec un modèle en Q5, "
    "comptez ×1,25 à ×1,5 ; une quantification plus légère "
    "achète du facteur.":
        "**Native sd.cpp HD pass**: the image is enlarged, then "
        "**re-denoised as a whole** by your generation model (Krea 2, "
        "Flux.2). All in **a single command**, 100% GPU, no PyTorch.\n\n"
        "Two fundamental differences from the creative SDXL upscale:\n"
        "- **no tiling at all** — the second pass sees the whole image, so "
        "there is no seam *possible*, and no mismatch between neighbouring "
        "squares;\n"
        "- **your model** does the redrawing, not a 2023 SDXL: the added "
        "detail stays in the style the model already knows.\n\n"
        "In exchange, **it is VRAM-hungry**: refusing to tile has a price. The "
        "second pass allocates a buffer proportional to the pixel count, "
        "**on top of the model weights** already on the card. The factor is "
        "therefore budgeted from your VRAM and your model's size, and lowered "
        "on its own if it does not fit — the log states the value it settled "
        "on. On 11–12 GB with a Q5 model, expect ×1.25 to ×1.5; a lighter "
        "quantization buys factor.",
    "> ⚠️ **Aucun modèle de génération installé.** "
    "Téléchargez Krea 2 Turbo ou Flux.2 Klein depuis "
    "l'onglet « Catalogue de modèles », puis revenez ici.":
        "> ⚠️ **No generation model installed.** Download Krea 2 Turbo or "
        "Flux.2 Klein from the “Model catalog” tab, then come back here.",
    "Le côté final est plafonné : au-delà, le facteur est réduit "
    "automatiquement et le journal annonce la valeur retenue.":
        "The final side is capped: beyond it the factor is reduced "
        "automatically and the log states the value it settled on.",
    "Modèle de génération": "Generation model",
    "Image à passer en HD": "Image to take to HD",
    "Facteur d'agrandissement": "Enlargement factor",
    "Détail ajouté (débruitage de la passe HD)":
        "Added detail (HD pass denoise)",
    "Le SEUL réglage qui compte vraiment. 0,2 = "
    "reste très près de l'original ; 0,5+ = le "
    "modèle réinvente franchement la matière.":
        "The ONLY setting that really matters. 0.2 = stays very close to the "
        "original; 0.5+ = the model frankly reinvents the material.",
    "Agrandissement intermédiaire": "Intermediate enlargement",
    "Ce qui agrandit AVANT le second débruitage. "
    "« Latent » travaille dans l'espace du modèle "
    "et laisse le débruitage tout reconstruire ; "
    "un ESRGAN donne une base déjà nette (utile "
    "sur du trait), au risque de figer ses propres "
    "défauts.":
        "What enlarges BEFORE the second denoise. “Latent” works in the "
        "model's own space and lets the denoise rebuild everything; an ESRGAN "
        "gives an already-crisp base (useful on line art), at the risk of "
        "freezing its own flaws.",
    "Latent (défaut — le plus doux)": "Latent (default — the gentlest)",
    "Latent antialiasé": "Latent antialiased",
    "Lanczos (image, neutre)": "Lanczos (image, neutral)",
    "Description (optionnelle)": "Description (optional)",
    "ce que montre l'image, en quelques mots":
        "what the image shows, in a few words",
    "Guide le détail ajouté. Vide fonctionne très "
    "bien : le modèle part de l'image.":
        "Guides the added detail. Empty works very well: the model starts "
        "from the image.",
    "🚀 Passer en HD": "🚀 Take to HD",
    "Passe HD…": "HD pass…",
    "Aucun modèle de génération installé : téléchargez-en un depuis l'onglet "
    "« Catalogue de modèles ».":
        "No generation model installed: download one from the “Model "
        "catalog” tab.",

    # ---- toolkit : SDXL créatif ----
    "✨ Upscale SDXL": "✨ SDXL upscale",
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
    "Préréglage (règle prompt, négatif, créativité, CFG et structure)":
        "Preset (sets prompt, negative, creativity, CFG and structure)",
    "Prompt négatif (vide = défaut orienté photo)":
        "Negative prompt (empty = photo-oriented default)",
    "photorealistic, film grain, noise…": "photorealistic, film grain, noise…",
    "Ce qu'on interdit à SDXL d'ajouter. Sur du dessin, c'est ce qui empêche "
    "le grain et la matière photo de se poser sur les aplats.":
        "What SDXL is forbidden to add. On drawings, this is what keeps grain "
        "and photo texture off the flat color areas.",
    "pré-agrandissement **{m}** (dessin) au lieu de Lanczos":
        "pre-upscale **{m}** (drawing) instead of Lanczos",
    "⚠️ aucun upscaler **dessin** installé — la base restera en Lanczos "
    "(traits plus mous). Téléchargez les upscalers dans l'onglet "
    "« 🔼 Agrandir ».":
        "⚠️ no **drawing** upscaler installed — the base stays on Lanczos "
        "(softer linework). Download the upscalers from the "
        "“🔼 Upscale” tab.",
    "⚠️ ControlNet Tile pas installé : la structure ne sera pas verrouillée.":
        "⚠️ ControlNet Tile not installed: structure will not be locked.",
    "structure verrouillée par ControlNet Tile ({v})":
        "structure locked by ControlNet Tile ({v})",
    "créativité {d} · CFG {c} · {s} pas":
        "creativity {d} · CFG {c} · {s} steps",
    "négatif adapté": "matching negative",
    "🔍 Net & fidèle (aucun ajout)": "🔍 Sharp & faithful (no additions)",
    "✨ Ajouter du détail": "✨ Add detail",
    "🧴 Peau réaliste (portrait)": "🧴 Realistic skin (portrait)",
    "🌿 Nature / paysage": "🌿 Nature / landscape",
    "🏙️ Architecture / produit": "🏙️ Architecture / product",
    "🖍️ Illustration / BD — trait net, sans interpolation":
        "🖍️ Illustration / comics — crisp linework, no interpolation",
    "🎨 Illustration peinte / concept art":
        "🎨 Painted illustration / concept art",
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
    "Mode": "Mode",
    "Décrivez la scène ET le mouvement : « a red fox walking through tall "
    "grass, camera slowly pushing in »…":
        "Describe the scene AND the motion: “a red fox walking through tall "
        "grass, camera slowly pushing in”…",
    "Image de départ": "Starting image",
    "Image de fin": "End image",
    "Format": "Format",
    "Passe de reprise à résolution doublée. Nettement plus net, mais nettement "
    "plus long et plus gourmand : à garder pour la fin.":
        "A refine pass at doubled resolution. Clearly sharper, but clearly "
        "slower and hungrier: save it for last.",
    "1.0 sur la version distillée : elle est entraînée pour ça.":
        "1.0 on the distilled version: that is what it was trained for.",
    "Saisissez un prompt.": "Enter a prompt.",
    "Fournissez l'image de départ.": "Provide the starting image.",
    "Fournissez l'image de fin.": "Provide the end image.",

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
    "⚡ Accélération (avancé)": "⚡ Acceleration (advanced)",
    "Convolution directe — modèle de diffusion":
        "Direct convolution — diffusion model",
    "Convolution directe — VAE": "Direct convolution — VAE",
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
    "Mémoire unifiée : **{ram} Go**": "Unified memory: **{ram} GB**",
    "**GPU détecté :**": "**Detected GPU:**",
    "~{vram} Go adressables par le GPU": "~{vram} GB addressable by the GPU",
    "{name} — mémoire unifiée {ram} Go, dont ~{vram} Go utilisables par le GPU -> diffusion en {quant}.":
        "{name} — {ram} GB unified memory, ~{vram} GB usable by the GPU -> {quant} diffusion.",
    "Mémoire unifiée : la décharge en RAM est désactivée (elle n'économise rien ici) et le calcul passe par Metal.":
        "Unified memory: RAM offload is disabled (it saves nothing here) and compute goes through Metal.",
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

    # ---- samplers & schedulers : fiches d'aide (atelier/sampling.py) ----
    "⭐ recommandé · △ peu adapté · ⚠️ déconseillé pour CE modèle":
        "⭐ recommended · △ poorly suited · ⚠️ discouraged for THIS model",
    "Répartition des pas de débruitage":
        "How the denoising steps are spread out",
    "📖 Pourquoi la moitié du menu est inutile ici":
        "📖 Why half of this menu is useless here",
    "La méthode de base : un pas, une évaluation, aucun ajout de bruit.":
        "The baseline method: one step, one evaluation, no added noise.",
    "Prévisible, reproductible, et la seule qui n'a rien à « rattraper » quand les pas sont comptés. C'est le défaut de tous nos modèles.":
        "Predictable, reproducible, and the only one with nothing to “catch up” when steps are scarce. It is the default on every one of our models.",
    "Aucun raffinement : sur BEAUCOUP de pas, d'autres méthodes la dépassent — mais on n'est pas dans ce régime.":
        "No refinement: over MANY steps other methods beat it — but that is not the regime we are in.",
    "Euler + réinjection de bruit frais à chaque pas.":
        "Euler plus a fresh injection of noise at every step.",
    "Sur des modèles non distillés et beaucoup de pas, apporte de la variété et du micro-détail.":
        "On non-distilled models with many steps, it adds variety and micro-detail.",
    "Le bruit réinjecté doit ensuite être reconvergé, ce qui demande un budget de pas confortable. Trop court : rendu mou ou bruité. Et deux rendus ne sont jamais identiques.":
        "The injected noise then has to be reconverged, which takes a comfortable step budget. Too short: a soft or noisy render. And no two renders are ever alike.",
    "Euler avec une correction : deux évaluations par pas.":
        "Euler with a correction: two evaluations per step.",
    "Trajectoire plus juste par pas.":
        "A more accurate trajectory per step.",
    "**Deux fois plus lent** à nombre de pas égal. Quand les pas sont comptés, ce budget est mieux dépensé en pas supplémentaires d'Euler.":
        "**Twice as slow** for the same step count. When steps are scarce, that budget is better spent on extra Euler steps.",
    "Méthode d'ordre 2, deux évaluations par pas.":
        "A second-order method, two evaluations per step.",
    "Bonne précision par pas sur les modèles classiques.":
        "Good per-step accuracy on classic models.",
    "Même coût double que Heun ; il faut assez de pas pour que le gain s'exprime.":
        "The same doubled cost as Heun; it takes enough steps for the gain to show.",
    "Ordre 2, à un seul pas de mémoire, avec bruit ancestral.":
        "Second order, single-step memory, with ancestral noise.",
    "Réputée sur SD1.5/SDXL en 20-30 pas.":
        "Well regarded on SD1.5/SDXL at 20-30 steps.",
    "Cumule les deux défauts qui comptent ici : coût double ET bruit ancestral non reconvergé.":
        "It stacks the two flaws that matter here: doubled cost AND unreconverged ancestral noise.",
    "Multi-pas : réutilise l'évaluation précédente au lieu d'en refaire une.":
        "Multistep: it reuses the previous evaluation instead of computing a new one.",
    "Le meilleur rapport qualité/temps du lot… à partir d'une quinzaine de pas.":
        "The best quality/time ratio of the lot… from about fifteen steps up.",
    "Son historique n'existe qu'après le 2ᵉ pas : sur un parcours très court, une bonne part se fait sans lui.":
        "Its history only exists after the 2nd step: on a very short run, a good part of it happens without one.",
    "Variante de DPM++ 2M au calcul de pas révisé.":
        "A DPM++ 2M variant with a revised step computation.",
    "Corrige des artefacts de la v1 sur les premiers pas.":
        "Fixes some v1 artefacts on the first steps.",
    "Même limite : le multi-pas a besoin de pas.":
        "Same limit: multistep needs steps.",
    "DPM++ 2M en formulation stochastique (bruit à chaque pas).":
        "DPM++ 2M in stochastic form (noise at every step).",
    "Texture plus riche sur les longs échantillonnages.":
        "Richer texture on long sampling runs.",
    "Stochastique : même exigence de pas que l'ancestral, et rendu non reproductible.":
        "Stochastic: the same step requirement as the ancestral ones, and a non-reproducible render.",
    "Variante à arbre brownien : le bruit devient reproductible.":
        "A Brownian-tree variant: the noise becomes reproducible.",
    "Retrouve la reproductibilité que la version SDE perd.":
        "Recovers the reproducibility the plain SDE version loses.",
    "Reste stochastique dans son principe : il lui faut des pas.":
        "Still stochastic in principle: it needs steps.",
    "Pseudo-multi-pas amélioré, sans bruit ajouté.":
        "Improved pseudo-multistep, with no added noise.",
    "Sobre et déterministe ; monte en qualité dès une dizaine de pas.":
        "Sober and deterministic; quality climbs from about ten steps up.",
    "Historique à construire, comme toute méthode multi-pas.":
        "A history to build, like every multistep method.",
    "iPNDM à coefficients variables.":
        "iPNDM with variable coefficients.",
    "Un peu plus stable qu'iPNDM sur les schedules irréguliers.":
        "Slightly more stable than iPNDM on irregular schedules.",
    "Même réserve sur le nombre de pas.":
        "Same reservation about the step count.",
    "Échantillonneur des modèles distillés **par Latent Consistency**.":
        "The sampler for models distilled **by Latent Consistency**.",
    "Excellent — sur un modèle LCM.":
        "Excellent — on an LCM model.",
    "Ni Flux.2 Klein ni Krea 2 Turbo ne sont distillés en LCM. Leur appliquer son parcours donne un rendu délavé.":
        "Neither Flux.2 Klein nor Krea 2 Turbo is LCM-distilled. Applying its trajectory to them gives a washed-out render.",
    "DDIM avec alignement des timesteps « trailing ».":
        "DDIM with “trailing” timestep alignment.",
    "Utile sur les modèles où la fin du parcours est mal échantillonnée.":
        "Useful on models whose end of trajectory is poorly sampled.",
    "Pensé pour la diffusion classique ; sans objet sur du flow matching.":
        "Designed for classic diffusion; moot on flow matching.",
    "Comme LCM : réservé aux modèles distillés **en TCD**.":
        "Like LCM: reserved for models distilled **in TCD**.",
    "Très peu de pas — sur un modèle TCD.":
        "Very few steps — on a TCD model.",
    "Nos modèles ne le sont pas.":
        "Ours are not.",
    "Intégrateur exponentiel multi-pas.":
        "An exponential multistep integrator.",
    "Très bonne précision sur les modèles de flow, à pas moyens.":
        "Very good accuracy on flow models, at medium step counts.",
    "Multi-pas : bridé quand les pas manquent. Le candidat le plus crédible pour essayer autre chose dès qu'il y en a.":
        "Multistep: hobbled when steps are missing. The most credible candidate for trying something other than Euler as soon as there are some.",
    "Intégrateur exponentiel à un pas, ordre 2.":
        "A single-step, second-order exponential integrator.",
    "Précis dès les premiers pas, sans historique à constituer — ce qui le rend, lui, compatible avec un budget serré.":
        "Accurate from the very first steps, with no history to build — which makes this one compatible with a tight budget.",
    "Deux évaluations par pas : à durée égale, Euler en fait deux fois plus.":
        "Two evaluations per step: for the same wall time, Euler does twice as many.",
    "Solveur SDE à réversibilité exacte.":
        "An exactly reversible SDE solver.",
    "Le plus rigoureux des stochastiques.":
        "The most rigorous of the stochastic methods.",
    "Stochastique : il lui faut des pas pour donner sa mesure.":
        "Stochastic: it needs steps to show what it can do.",
    "Euler avec la correction de guidage « CFG++ ».":
        "Euler with the “CFG++” guidance correction.",
    "Enlève les sur-saturations dues à un CFG élevé.":
        "Removes the over-saturation caused by a high CFG.",
    "**Nos deux modèles tournent à CFG 1.0** : il n'y a aucun guidage à corriger. Cette variante n'a rien à faire ici.":
        "**Both of our models run at CFG 1.0**: there is no guidance to correct. This variant has no business here.",
    "La version ancestrale de la précédente : correction CFG++ plus réinjection de bruit à chaque pas.":
        "The ancestral version of the above: CFG++ correction plus a noise injection at every step.",
    "Aucun ici : la correction CFG++ est neutre à CFG 1.0, il ne reste que le bruit ancestral, qu'Euler Ancestral fournit déjà.":
        "None here: the CFG++ correction is a no-op at CFG 1.0, leaving only the ancestral noise, which Euler Ancestral already provides.",
    "Cumule l'inutilité du CFG++ à CFG 1.0 et le bruit ancestral, qui demande un budget de pas confortable pour se résorber.":
        "It stacks the uselessness of CFG++ at CFG 1.0 with ancestral noise, which needs a comfortable step budget to settle.",
    "Euler à extrapolation de gradient (paramètre `gamma`).":
        "Euler with gradient extrapolation (the `gamma` parameter).",
    "Peut resserrer le trait quand les pas sont très comptés — le seul du lot à viser explicitement ce régime.":
        "Can tighten the result when steps are very scarce — the only one of the lot explicitly aimed at that regime.",
    "Non exposé ici : `gamma` se règle via `--extra-sample-args`, et sans lui l'effet est marginal.":
        "Not exposed here: `gamma` is set through `--extra-sample-args`, and without it the effect is marginal.",
    "Multi-pas linéaire classique (`lms_divisions`, défaut 1000).":
        "Classic linear multistep (`lms_divisions`, default 1000).",
    "Ajout récent de sd.cpp ; méthode éprouvée sur de longs parcours.":
        "A recent sd.cpp addition; a method proven on long runs.",
    "Multi-pas : sans un vrai budget de pas, l'historique n'existe pas.":
        "Multistep: without a real step budget, the history never exists.",
    "Auto (modèle)":
        "Auto (model)",
    "Laisse le moteur choisir d'après le modèle chargé.":
        "Lets the engine choose according to the loaded model.",
    "Toujours cohérent avec le modèle : `flux2` pour Flux.2 Klein, `discrete` pour Krea 2. C'est le réglage documenté par sd.cpp.":
        "Always consistent with the model: `flux2` for Flux.2 Klein, `discrete` for Krea 2. This is the setting sd.cpp documents.",
    "Aucun — sauf si vous voulez expérimenter en connaissance de cause.":
        "None — unless you want to experiment knowingly.",
    "Répartition uniforme sur les sigmas du modèle.":
        "A uniform spread over the model's sigmas.",
    "Neutre et sans surprise. C'est ce que « Auto » choisit sur Krea 2.":
        "Neutral and unsurprising. This is what “Auto” picks on Krea 2.",
    "Rien de particulier ; simplement pas optimisé pour un modèle donné.":
        "Nothing in particular; simply not tuned for any one model.",
    "Répartition concentrant les pas vers les bas sigmas.":
        "A spread that concentrates the steps towards the low sigmas.",
    "La référence sur SD1.5 / SDXL, où elle gagne beaucoup.":
        "The reference on SD1.5 / SDXL, where it gains a lot.",
    "Conçue pour la diffusion **EDM à prédiction d'epsilon**. Nos modèles sont en flow matching : la courbe ne correspond pas au parcours.":
        "Designed for **EDM epsilon-prediction diffusion**. Our models are flow matching: the curve does not match the trajectory.",
    "Décroissance exponentielle des sigmas.":
        "Exponential decay of the sigmas.",
    "Simple, parfois utile sur les modèles à v-prediction.":
        "Simple, occasionally useful on v-prediction models.",
    "Même inadéquation que Karras vis-à-vis du flow matching.":
        "The same mismatch as Karras with respect to flow matching.",
    "Répartition optimisée par NVIDIA pour les **petits budgets de pas**.":
        "A spread optimised by NVIDIA for **small step budgets**.",
    "Pensée exactement pour le régime 8-12 pas — l'idée est bonne ici.":
        "Designed for exactly the 8-12 step regime — the idea is sound here.",
    "Ses tables sont calibrées sur SD1.5/SDXL, pas sur nos modèles : le transfert est plausible mais non garanti. À essayer sur Krea 2.":
        "Its tables are calibrated on SD1.5/SDXL, not on our models: the transfer is plausible but not guaranteed. Worth trying on Krea 2.",
    "Répartition issue d'une recherche sur graphe.":
        "A spread derived from a graph search.",
    "Bons résultats publiés à faible nombre de pas.":
        "Good published results at low step counts.",
    "Même réserve qu'AYS : calibrée ailleurs.":
        "Same reservation as AYS: calibrated elsewhere.",
    "Courbe lissée aux deux extrémités.":
        "A curve smoothed at both ends.",
    "Transitions douces, peu d'à-coups en début de parcours.":
        "Soft transitions, few jolts at the start of the run.",
    "Effet discret ; rien qui compense un scheduler adapté au modèle.":
        "A subtle effect; nothing that makes up for a model-appropriate scheduler.",
    "Uniforme, à la façon des implémentations SGM.":
        "Uniform, in the style of the SGM implementations.",
    "Proche de Discrete, comportement prévisible.":
        "Close to Discrete, predictable behaviour.",
    "Aucun avantage identifié sur nos modèles.":
        "No identified advantage on our models.",
    "Répartition linéaire élémentaire.":
        "An elementary linear spread.",
    "Robuste, sans paramètre. Défaut de DDIM Trailing.":
        "Robust, parameter-free. The default for DDIM Trailing.",
    "Grossière quand les pas sont peu nombreux.":
        "Coarse when the steps are few.",
    "Répartition minimisant une divergence KL le long du parcours.":
        "A spread minimising a KL divergence along the trajectory.",
    "Bien fondée théoriquement, correcte à pas moyens.":
        "Theoretically well founded, correct at medium step counts.",
    "Gain non démontré quand les pas sont comptés.":
        "No demonstrated gain when steps are scarce.",
    "Répartition des modèles Latent Consistency.":
        "The spread for Latent Consistency models.",
    "Indispensable — avec l'échantillonneur LCM.":
        "Indispensable — with the LCM sampler.",
    "Hors de ce couple, elle écrase le parcours et délave le rendu.":
        "Outside that pairing it crushes the trajectory and washes the render out.",
    "Courbe en tangente, très marquée.":
        "A tangent curve, very pronounced.",
    "Effet stylistique parfois intéressant.":
        "An occasionally interesting stylistic effect.",
    "Empirique, sans fondement pour nos modèles.":
        "Empirical, with no grounding for our models.",
    "Répartition **taillée pour Flux.2**.":
        "A spread **cut for Flux.2**.",
    "Ce que « Auto » sélectionne sur Flux.2 Klein : le bon choix, explicitement.":
        "What “Auto” selects on Flux.2 Klein: the right choice, made explicit.",
    "Sur Krea 2, rien ne dit qu'elle transfère.":
        "On Krea 2, nothing says it transfers.",
    "Répartition des sigmas taillée pour les modèles **Flux.1**, avec le décalage (shift) propre à cette génération.":
        "A sigma spread cut for the **Flux.1** models, with the shift specific to that generation.",
    "Reste une courbe de flow matching cohérente : elle ne casse rien, et donne un rendu légèrement plus contrasté sur les gros plans.":
        "It remains a coherent flow-matching curve: it breaks nothing, and gives a slightly more contrasted render on close-ups.",
    "Flux.2 a la sienne ; utiliser celle de Flux.1 revient à prendre l'ancienne version d'un réglage taillé sur mesure.":
        "Flux.2 has its own; using the Flux.1 one amounts to picking the previous version of a bespoke setting.",
    "Répartition suivant une loi Beta (paramètres `alpha`, `beta`).":
        "A spread following a Beta law (`alpha`, `beta` parameters).",
    "Très modulable — via `--extra-sample-args`.":
        "Highly tunable — through `--extra-sample-args`.",
    "Sans réglage de ses paramètres, aucun intérêt par rapport à Discrete.":
        "Without tuning its parameters, no benefit over Discrete.",
    "Répartition logit-normale, celle utilisée à l'entraînement de beaucoup de modèles de flow.":
        "A logit-normal spread, the one used to train many flow models.",
    "Cohérente avec la façon dont ces modèles ont été entraînés — la piste la plus défendable après « Auto ».":
        "Consistent with how these models were trained — the most defensible avenue after “Auto”.",
    "Ses paramètres (`mu`, `std`) ne sont pas exposés ici.":
        "Its parameters (`mu`, `std`) are not exposed here.",
    "**Recommandé** pour ce modèle.":
        "**Recommended** for this model.",
    "Utilisable, sans avantage net ici.":
        "Usable, with no clear advantage here.",
    "Peu adapté à ce modèle.":
        "Poorly suited to this model.",
    "**Déconseillé** avec ce modèle.":
        "**Discouraged** with this model.",
    "Sur **{model}**, trois propriétés du modèle décident presque tout — et elles\nécartent des familles entières d'options, pas une ou deux au cas par cas.\n\n**1. C'est un modèle de *flow matching*.** sd.cpp le fait tourner en mode\n« Flux FLOW ». Les schedulers **Karras** et **Exponential**, qui font gagner\nbeaucoup sur SD 1.5 et SDXL, ont été conçus pour une autre mécanique (diffusion\nEDM à prédiction d'epsilon) : leur répartition de sigmas ne correspond pas au\nparcours suivi ici.\n\n**2. Il est distillé à CFG 1.0.** Il n'y a donc **aucun guidage à corriger** :\ntoute la famille **CFG++** (`Euler CFG++`, `Euler Ancestral CFG++`) n'a\nlittéralement rien à faire. C'est aussi pourquoi le **prompt négatif est\nignoré**, quel que soit l'échantillonneur choisi.\n\n**3. Il tourne en {steps} pas.** C'est très peu, et ça disqualifie deux familles :\n\n- les méthodes **ancestrales** et **stochastiques** (`Euler Ancestral`,\n  `DPM++ 2S Ancestral`, les `SDE`, `ER SDE`) réinjectent du bruit à chaque pas.\n  Ce bruit doit ensuite être reconvergé — il n'y a pas le budget pour ça, et le\n  rendu ressort mou ou bruité ;\n- les méthodes **multi-pas** (`DPM++ 2M`, `iPNDM`, `Res Multistep`, `LMS`)\n  doivent d'abord accumuler un historique d'évaluations. Sur {steps} pas, une\n  bonne partie du parcours se fait avant que cet historique existe.\n\nEnfin, **LCM** et **TCD** ne sont pas des options générales : ce sont les\néchantillonneurs de modèles distillés *par ces méthodes-là*. Ce modèle ne l'est\npas ; les appliquer délave le rendu.\n\n---\n\n**Ce qu'il reste, en pratique :** `Euler` + `Auto`. {advice}\n\n*Ces verdicts sont raisonnés à partir des propriétés du modèle, pas tirés d'un\nbanc d'essai — ils disent où porter vos essais, pas ce que votre œil va\npréférer.*":
        "On **{model}**, three properties of the model decide almost everything — and\nthey rule out entire families of options, not one or two case by case.\n\n**1. It is a *flow matching* model.** sd.cpp runs it in “Flux FLOW” mode. The\n**Karras** and **Exponential** schedulers, which gain a lot on SD 1.5 and SDXL,\nwere designed for another mechanism (EDM epsilon-prediction diffusion): their\nsigma spread does not match the trajectory followed here.\n\n**2. It is distilled at CFG 1.0.** There is therefore **no guidance to\ncorrect**: the whole **CFG++** family (`Euler CFG++`, `Euler Ancestral CFG++`)\nhas literally nothing to do. This is also why the **negative prompt is\nignored**, whichever sampler you pick.\n\n**3. It runs in {steps} steps.** That is very few, and it disqualifies two\nfamilies:\n\n- the **ancestral** and **stochastic** methods (`Euler Ancestral`,\n  `DPM++ 2S Ancestral`, the `SDE` ones, `ER SDE`) inject noise at every step.\n  That noise then has to be reconverged — there is no budget for it, and the\n  render comes out soft or noisy;\n- the **multistep** methods (`DPM++ 2M`, `iPNDM`, `Res Multistep`, `LMS`) must\n  first accumulate a history of evaluations. Over {steps} steps, a good part of\n  the run happens before that history exists.\n\nFinally, **LCM** and **TCD** are not general-purpose options: they are the\nsamplers of models distilled *by those very methods*. This model is not;\napplying them washes the render out.\n\n---\n\n**What is left, in practice:** `Euler` + `Auto`. {advice}\n\n*These verdicts are reasoned from the model's properties, not drawn from a\nbenchmark — they say where to aim your experiments, not what your eye will\nprefer.*",
    "Sur 4 pas, il n'y a pratiquement rien à gagner ailleurs ; si vous voulez expérimenter, `Res 2S` est le seul autre à être précis sans historique à constituer.":
        "Over 4 steps there is virtually nothing to gain elsewhere; if you want to experiment, `Res 2S` is the only other one that is accurate without a history to build.",
    "Sur 8 pas, la marge est un peu plus large : `Res Multistep`, `DPM++ 2M` et le scheduler `AYS` (pensé pour les petits budgets de pas) valent un essai comparatif à seed fixe.":
        "Over 8 steps the margin is a little wider: `Res Multistep`, `DPM++ 2M` and the `AYS` scheduler (designed for small step budgets) are worth a side-by-side try at a fixed seed.",
    "🎛️ Matériel & optimisation":
        "🎛️ Hardware & optimization",
    "⚡ Accélération (avancé)":
        "⚡ Acceleration (advanced)",
    "🧪 Mesurer cette machine":
        "🧪 Measure this machine",
    "🌍 Interface, réseau & comptes":
        "🌍 Interface, network & accounts",
    "Le test génère la même image en **512×512, 4 pas, seed 424242** avec chaque placement disponible. Il mesure le temps et le pic VRAM, conserve les images pour comparaison et ne modifie aucun réglage tant que vous ne cliquez pas sur **Appliquer**.":
        "The test generates the same image at **512×512, 4 steps, seed 424242** through every available placement. It measures time and peak VRAM, keeps the images for comparison, and changes no setting until you click **Apply**.",
    "📋 Exporter le rapport système":
        "📋 Export the system report",
    "⏱️ Tester les placements GPU":
        "⏱️ Test the GPU placements",
    "Appliquer le profil le plus rapide":
        "Apply the fastest profile",
    "⏹️ Arrêter le test":
        "⏹️ Stop the test",
    "Rapport JSON":
        "JSON report",
    "Journal du test":
        "Test log",
    "Streaming des couches depuis la RAM":
        "Layer streaming from RAM",
    "Requiert les poids de diffusion en RAM, mais pas de budget --max-vram. Très dépendant du PCIe.":
        "Requires the diffusion weights in RAM, but no --max-vram budget. Very PCIe-dependent.",
    "Encodeur sur la 2e carte, poids en RAM":
        "Encoder on the 2nd card, weights in RAM",
    "Format du modèle de diffusion":
        "Diffusion model format",
    "GGUF — recommandé et éprouvé":
        "GGUF — recommended and proven",
    "INT8 ConvRot — expérimental RTX 30xx":
        "INT8 ConvRot — experimental, RTX 30xx",
    "Comparez les deux avec le test A/B des Réglages.":
        "Compare the two with the A/B test in Settings.",
    "Restauration diffusion **SeedVR2** : récupère des détails plus naturels qu'ESRGAN tout en restant plus fidèle que l'upscale créatif SDXL. Le calcul reste sur la RTX 3060 ; la GTX 1080 Ti peut servir de réserve pour les poids. Le 3B suffit dans la plupart des cas ; le 7B garde mieux les textures fines (visages, tissus) mais prend le double de temps.":
        "**SeedVR2** diffusion restoration: recovers more natural detail than "
        "ESRGAN while staying more faithful than the creative SDXL upscale. "
        "Compute stays on the RTX 3060; the GTX 1080 Ti can hold the weights. "
        "The 3B is enough most of the time; the 7B keeps fine textures (faces, "
        "fabric) better but takes twice as long.",
    "3B Q8 — valeur sûre, la plus rapide":
        "3B Q8 — safe default, fastest",
    "3B Q4 — repli si la mémoire manque":
        "3B Q4 — fallback when memory runs short",
    "7B Q4 — plus de détails, environ 2× plus lent":
        "7B Q4 — more detail, about 2× slower",
    "7B Q4 « sharp » — le plus net (peut durcir le grain)":
        "7B Q4 “sharp” — sharpest (can harden grain)",
    "Les poids se téléchargent tout seuls au premier usage (4,8 Go pour un 7B).":
        "Weights download themselves on first use (4.8 GB for a 7B).",
    "16 recommandé avec 12 Go ; 24 puis 36 si OOM.":
        "16 is right with 12 GB; try 24 then 36 if you hit OOM.",
    "📁 Restaurer un dossier en une fois":
        "📁 Restore a whole folder at once",
    "Sélectionnez un dossier d'images. SeedVR2 charge le modèle **une seule fois**, le garde en cache et traite tous les fichiers sans modifier les originaux. Les résultats vont dans un sous-dossier horodaté de `outputs/`.":
        "Pick a folder of images. SeedVR2 loads the model **once**, keeps it cached and processes every file without touching the originals. Results go to a timestamped subfolder of `outputs/`.",
    "Dossier d'images":
        "Image folder",
    "🌱 Restaurer tout le dossier":
        "🌱 Restore the whole folder",
    "Résultats du lot":
        "Batch results",
    "Journal du lot":
        "Batch log",
    "Sélectionnez un dossier d'images.":
        "Pick a folder of images.",
    "Installez d'abord SeedVR2.":
        "Install SeedVR2 first.",
    "🎭 Préréglage perso":
        "🎭 Custom preset",
    "⚙️ Réinstaller / réparer {title}":
        "⚙️ Reinstall / repair {title}",
    "🎨 Styles — préréglages, photo, artistiques ({n} styles)":
        "🎨 Styles — presets, photo, artistic ({n} styles)",
    "📷 Photo ({n})":
        "📷 Photo ({n})",
    "🖍️ Artistiques ({n})":
        "🖍️ Artistic ({n})",
    "### ⚠️ Aucune carte NVIDIA détectée\nL'application tournera sur le processeur : c'est **très lent** (des minutes par image). Vérifiez vos pilotes, ou tapez `nvidia-smi` dans un terminal.":
        "### ⚠️ No NVIDIA card detected\nThe app will run on the processor: that is **very slow** (minutes per image). Check your drivers, or type `nvidia-smi` in a terminal.",
    "### Votre matériel":
        "### Your hardware",
    "**Ce que l'application en fait, sans rien vous demander :**":
        "**What the app does with it, without asking you anything:**",
    "- l'accélération « flash attention » est active (votre carte la gère) ;":
        "- “flash attention” acceleration is on (your card supports it);",
    "- « flash attention » reste désactivée : votre carte n'a pas les unités qu'il faut, l'activer ralentirait ;":
        "- “flash attention” stays off: your card lacks the units for it, turning it on would slow things down;",
    "- l'image finale est assemblée par morceaux, pour ne pas saturer la carte au dernier moment.":
        "- the final image is assembled in pieces, so the card is not saturated at the last moment.",
    "- l'image finale est assemblée d'un seul tenant : vous avez la place, autant éviter les jointures.":
        "- the final image is assembled in one piece: you have the room, so no seams.",
    "---\n### La seule question qu'on vous pose\nTout le reste se calcule à partir de votre carte. Ceci ne se calcule pas, parce que c'est une préférence : voulez-vous de la **marge** (ça passe toujours) ou du **détail** (c'est plus fin, mais plus juste en mémoire) ?":
        "---\n### The one question we ask you\nEverything else is computed from your card. This one cannot be, because it is a preference: do you want **headroom** (it always fits) or **detail** (finer, but tighter on memory)?",
    "> **Un souci précis ?**  \n> *« Erreur de mémoire / la génération s'arrête »* → prenez **🪶 Plus de marge mémoire** ci-dessus.  \n> *« C'est trop lent »* → cela ne se joue pas ici mais dans l'onglet de génération : baissez le **nombre de pas** et la **taille de l'image**, qui pèsent bien plus lourd.  \n> *« Mes images sont fades »* → là non plus : c'est le **prompt** et les **styles**, pas un réglage matériel.":
        "> **Something specific going wrong?**  \n> *“Out of memory / generation stops”* → pick **🪶 More memory headroom** above.  \n> *“It is too slow”* → not settled here but in the generation tab: lower the **step count** and the **image size**, which weigh far more.  \n> *“My images look dull”* → not here either: that is the **prompt** and the **styles**, not a hardware setting.",
    "⏹️ Arrêt demandé — la mesure en cours se termine.":
        "⏹️ Stop requested — the run in progress will finish first.",
    "**{name}** — {vram} Go de mémoire vidéo · {ram} Go de RAM":
        "**{name}** — {vram} GB of video memory · {ram} GB of RAM",
    "- le modèle d'image est chargé en `{quant}` — le meilleur compromis qui tienne dans {vram} Go ;":
        "- the image model is loaded as `{quant}` — the best trade-off that fits in {vram} GB;",
    "- l'analyse de votre texte se fait en `{enc}`, **rangée en RAM** : elle ne prend pas de place sur la carte ;":
        "- your text is analysed as `{enc}`, **kept in RAM**: it takes no room on the card;",
    "  \n→ modèle chargé en `{quant}` au lieu de `{ref}`.":
        "  \n→ model loaded as `{quant}` instead of `{ref}`.",
    "  \n→ modèle chargé en `{quant}`.":
        "  \n→ model loaded as `{quant}`.",
    "La plus puissante est prise par défaut.":
        "The most powerful one is used by default.",
    "---\n### Vous avez deux cartes\nIl n'y a pas de bonne réponse universelle : cela dépend autant du **port PCIe** de la seconde carte que de sa mémoire. Plutôt que de vous faire deviner, l'application peut **mesurer**.":
        "---\n### You have two cards\nThere is no universally right answer: it depends as much on the second card's **PCIe slot** as on its memory. Rather than make you guess, the app can **measure**.",
    "⏱️ Mesurer sur ma machine":
        "⏱️ Measure on my machine",
    "Appliquer le plus rapide":
        "Apply the fastest",
    "⏹️ Arrêter":
        "⏹️ Stop",
    "Détail de la mesure (journal et rapport)":
        "Measurement details (log and report)",
    "🔧 Expert — options brutes de sd.cpp (facultatif)":
        "🔧 Expert — raw sd.cpp options (optional)",
    "⚠️ **Rien ici n'est nécessaire.** Ces options existent parce que sd.cpp les expose, pas parce qu'il faut y toucher. Elles se règlent en mesurant, pas en devinant — et le curseur ci-dessus couvre déjà les cas courants. Toucher à cette section **désactive le réglage automatique**.":
        "⚠️ **Nothing here is required.** These options exist because sd.cpp exposes them, not because you should touch them. They are set by measuring, not by guessing — and the slider above already covers the usual cases. Touching this section **turns off automatic tuning**.",
    "**Quantification imposée** — « auto » = laisser l'application décider d'après la carte.":
        "**Forced quantization** — “auto” = let the app decide from the card.",
    "**Options mémoire du moteur.**":
        "**Engine memory options.**",
    "---\n**Cache entre les pas** — réutilise des calculs d'un pas de diffusion au suivant. Ne gagne quelque chose qu'au-delà de ~10 pas ; nos modèles en font 4 à 8, donc **laissez désactivé** sauf mesure contraire.":
        "---\n**Cache between steps** — reuses computations from one diffusion step to the next. Only pays off above ~10 steps; our models run 4 to 8, so **leave it off** unless a measurement says otherwise.",
    "---\n**Convolution directe** — supprime un gros tampon intermédiaire. Gain de mémoire certain ; effet sur la vitesse **imprévisible** (parfois mieux, parfois moins bien). À chronométrer, pas à cocher les yeux fermés.":
        "---\n**Direct convolution** — removes a large intermediate buffer. The memory gain is certain; the speed effect is **unpredictable** (sometimes better, sometimes worse). To be timed, not ticked blindly.",
    "---\n**Découpage du calcul** — autorise le moteur à découper son graphe pour tenir dans un budget au lieu d'échouer. **C'est plus lent** : à réserver aux résolutions qui ne passent pas autrement. L'onglet 🚀 HD s'en sert déjà tout seul.":
        "---\n**Splitting the computation** — lets the engine cut its graph to fit a budget instead of failing. **It is slower**: keep it for resolutions that will not fit otherwise. The 🚀 HD tab already uses it on its own.",
    "⚡ Profil RTX 3060 + GTX 1080 Ti":
        "⚡ RTX 3060 + GTX 1080 Ti profile",
    "🌍 Langue, thème et comptes":
        "🌍 Language, theme and accounts",
    "Thème enregistré. **Redémarrez l'application** pour l'appliquer.":
        "Theme saved. **Restart the app** to apply it.",
    "Endpoint enregistré.":
        "Endpoint saved.",
    "Jeton Civitai enregistré.":
        "Civitai token saved.",
    "❌ Aucune configuration n'a pu être mesurée (voir le journal).":
        "❌ No configuration could be measured (see the log).",
    "Deuxième carte disponible : {other}.":
        "Second card available: {other}.",
    "---\n### 🧪 Dans le doute, mesurez\nL'application génère la **même image** {n} fois par configuration (plus une première jetée, le temps que tout soit chargé) et garde la médiane. Elle ne change **aucun réglage** : elle vous dit lequel est le plus rapide, vous décidez ensuite.":
        "---\n### 🧪 When in doubt, measure\nThe app generates the **same image** {n} times per configuration (plus a first one thrown away, the time for everything to load) and keeps the median. It changes **no setting**: it tells you which is fastest, then you decide.",
    "Carte de l'améliorateur enregistrée.":
        "Enhancer card saved.",
    "Réglage expert appliqué (automatique désactivé).":
        "Expert setting applied (automatic tuning off).",
    "✅ Médiane sur {n} mesures · le plus rapide : **{best}**":
        "✅ Median over {n} runs · fastest: **{best}**",
    "⏳ Mesure en cours…":
        "⏳ Measuring…",
    "❌ Lancez d'abord la mesure.":
        "❌ Run the measurement first.",
    "Profil deux cartes appliqué : la RTX 3060 dessine, la 1080 Ti lit votre texte.":
        "Two-card profile applied: the RTX 3060 draws, the 1080 Ti reads your text.",
    "Tout sur une seule carte — le plus fiable":
        "Everything on one card — the most reliable",
    "La 2e carte s'occupe du texte — libère de la mémoire pour l'image":
        "The 2nd card handles the text — frees memory for the image",
    "Répartir automatiquement — à mesurer avant d'y croire":
        "Spread it automatically — measure before believing it",
    "Exige que le modèle soit rangé en RAM. Sans cela, le moteur ignore l'option.":
        "Requires the model to be kept in RAM. Without that, the engine ignores the option.",
    "Langue enregistrée. **Redémarrez l'application** (`run.bat` / `run.sh`) pour appliquer « {lang} ».":
        "Language saved. **Restart the app** (`run.bat` / `run.sh`) to apply “{lang}”.",
    "⏹️ Mesure interrompue — aucun réglage modifié.":
        "⏹️ Measurement interrupted — no setting changed.",
    "Auto — mémoire libre moins 1 Go":
        "Auto — free memory minus 1 GB",
    "Priorité appliquée : **{label}**.":
        "Priority applied: **{label}**.",
    "Carte de génération : #{idx}.":
        "Generation card: #{idx}.",
    "Configuration mesurée appliquée : **{mode}**.":
        "Measured configuration applied: **{mode}**.",
    "La même que pour l'image":
        "The same one as for the image",
    "{name} : pas de tensor cores → flash-attention désactivé (elle n'apporte rien ici), génération plus lente.":
        "{name}: no tensor cores → flash-attention disabled (it gains nothing here), slower generation.",
    "🪶 Plus de marge mémoire":
        "🪶 More memory headroom",
    "Si vous voyez des erreurs de mémoire, ou si vous générez en grand format. Le modèle est compressé d'un cran de plus et l'application économise partout où elle peut.":
        "If you get out-of-memory errors, or if you generate at large sizes. The model is compressed one notch further and the app saves memory wherever it can.",
    "⚖️ Équilibré (recommandé)":
        "⚖️ Balanced (recommended)",
    "Ce que votre carte peut tenir sans se battre. C'est le bon choix tant que rien ne vous gêne.":
        "What your card can hold without a fight. This is the right choice as long as nothing bothers you.",
    "🎨 Plus de détail":
        "🎨 More detail",
    "Un cran de compression en moins : l'image gagne un peu de finesse, et la carte a moins de marge. À prendre si tout passe déjà confortablement.":
        "One notch less compression: the image gains a little fineness, and the card has less headroom. Take it if everything already fits comfortably.",
    "Tout se fera sur une seule carte.":
        "Everything will run on a single card.",
    "La 2e carte lira votre texte ; l'image reste sur la première.":
        "The 2nd card will read your text; the image stays on the first.",
    "Répartition automatique activée — mesurez-la avant d'y croire.":
        "Automatic spreading enabled — measure it before believing it.",
    "Assemblage sur le processeur":
        "Assembly on the processor",
    "Convolution directe — modèle d'image":
        "Direct convolution — image model",
    "Convolution directe — assemblage":
        "Direct convolution — assembly",
    "Image assemblée par morceaux":
        "Image assembled in pieces",
    "Modèle rangé en RAM":
        "Model kept in RAM",
    "Texte sur le processeur":
        "Text on the processor",
    "Budget mémoire du calcul":
        "Memory budget for the computation",
    "Carte utilisée pour générer":
        "Card used for generating",
    "Modèle d'image":
        "Image model",
    "Analyse du texte":
        "Text analysis",
    "Priorité":
        "Priority",
    "Répartition":
        "Split",
    "Option (vide = défauts)":
        "Option (blank = defaults)",
    "Mode de cache":
        "Cache mode",
    "Carte pour l'améliorateur de prompt":
        "Card for the prompt enhancer",
    "Journal du test":
        "Test log",
    "Rapport JSON":
        "JSON report",
    "📝 Image → prompt":
        "📝 Image → prompt",
    "Image → prompt":
        "Image → prompt",
    "Donnez une image, récupérez le **prompt** qui permettrait de la refaire. Ce n'est pas une légende : un modèle de vision dirait « une photo d'un chat sur un canapé », ce qui, collé dans le champ Prompt, donne une image plate. Ici on nomme le **médium**, la **lumière**, l'**objectif**, la **palette** et le **cadrage** — les mots qui pilotent réellement la diffusion. Toujours en **anglais** : c'est la langue des modèles.":
        "Hand it an image, get back the **prompt** that would recreate it. This is not a caption: a vision model would say “a photo of a cat on a sofa”, which pasted into the Prompt field gives a flat image. Here we name the **medium**, the **light**, the **lens**, the **palette** and the **framing** — the words that actually steer diffusion. Always in **English**: that is the models' language.",
    "Modèle de vision-langage **Qwen2.5-VL-3B** (~7,5 Go), même famille que l'améliorateur de prompt. Chargé puis déchargé à chaque appel : **aucun conflit de VRAM** avec la génération. ⚠️ Licence *Qwen Research* — usage non commercial, comme l'améliorateur déjà installé.":
        "**Qwen2.5-VL-3B** vision-language model (~7.5 GB), same family as the prompt enhancer. Loaded then unloaded on each call: **no VRAM conflict** with generation. ⚠️ *Qwen Research* licence — non-commercial, like the enhancer already installed.",
    "Image à lire":
        "Image to read",
    "Ce que vous voulez en tirer":
        "What you want out of it",
    "📸 Refaire cette image — sujet ET style":
        "📸 Recreate this image — subject AND style",
    "🎨 Juste le style — à appliquer à autre chose":
        "🎨 Style only — to apply to something else",
    "🔍 Décrire simplement — ce qu'il y a dedans":
        "🔍 Plain description — what is in it",
    "Sujet, décor, lumière, couleurs, médium — de quoi refaire une image proche sur un autre modèle.":
        "Subject, setting, light, colours, medium — enough to recreate a close image on another model.",
    "**Aucun mot sur le sujet** : ni le chat, ni la voiture, ni le lieu. Seulement le rendu, à coller devant votre propre sujet.":
        "**Not a word about the subject**: not the cat, not the car, not the place. Only the look, to paste in front of your own subject.",
    "Deux ou trois phrases, sans vocabulaire de prompt. Pour savoir ce qu'il y a dans l'image, pas pour la regénérer.":
        "Two or three sentences, no prompt vocabulary. To know what is in the image, not to regenerate it.",
    "📝 Lire l'image":
        "📝 Read the image",
    "Prompt obtenu":
        "Resulting prompt",
    "Le texte apparaîtra ici — relisez-le avant de l'envoyer, c'est un point de départ, pas un verdict.":
        "The text will appear here — read it before sending, it is a starting point, not a verdict.",
    "**L'envoyer directement dans un onglet de génération** — le prompt y remplace le champ, vous générez ensuite quand vous voulez.":
        "**Send it straight to a generation tab** — the prompt replaces the field there, and you generate whenever you like.",
    "→ ⚡ Krea 2 Turbo":
        "→ ⚡ Krea 2 Turbo",
    "→ 🟣 Flux.2 Klein":
        "→ 🟣 Flux.2 Klein",
    "Journal":
        "Log",
    "Chargez d'abord une image.":
        "Load an image first.",
    "Installez d'abord « Image → prompt ».":
        "Install “Image → prompt” first.",
    "Lisez d'abord une image.":
        "Read an image first.",
}

_EN_INV: dict[str, str] = {v: k for k, v in _EN.items()}
