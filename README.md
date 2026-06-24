# 🟢 GEN.Ai Image Workshop

Studio d'inférence d'images **local**, moderne et léger, pensé pour les artistes.
Génère avec **Flux.2 Klein 9B** (GGUF), avec **bibliothèque de modèles à la
demande**, **optimisations automatiques selon votre carte RTX et votre RAM**,
**LoRA**, **presets sampler/scheduler**, **styles enregistrés**, **upscale
créatif façon Magnific** et un **Toolkit** (profondeur, détourage).

Aucun ComfyUI, aucune usine à gaz : une interface web claire.

| Onglet | Rôle |
|---|---|
| 🟣 **Flux.2 Klein** | text-to-image & édition d'image, presets, styles, LoRA, modèle perso |
| 📚 **Bibliothèque** | catalogue, recommandations selon le matériel, téléchargement à la demande |
| ✨ **Upscale créatif** | façon Magnific : ré-invente le détail (SDXL + ControlNet Tile, par tuiles) |
| 🧰 **Toolkit** | carte de profondeur · suppression d'arrière-plan (PNG transparent) |
| ⚙️ **Réglages** | matériel détecté, quantification, optimisations (auto/manuel) |

---

## Installation (portable)

### Windows (cartes RTX)
```bat
install.bat      ::  Python portable + dépendances + moteur GGUF (CUDA)
run.bat          ::  lance l'interface sur http://127.0.0.1:7860
```

### Linux
```bash
./install.sh
./run.sh
```

> L'installation ne télécharge **pas** les modèles : on les récupère ensuite à la
> demande depuis l'onglet **Bibliothèque** (comme une médiathèque). Tout reste
> dans le dossier du projet.

---

## Comment ça marche

### Moteur
La génération passe par **[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)**
(natif CUDA, format GGUF) — pas de PyTorch lourd pour la génération. Flux.2 Klein
9B se compose de :
- diffusion — [`unsloth/FLUX.2-klein-9B-GGUF`](https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF) (distillée, 4 pas)
- VAE flux2 — [`Comfy-Org/flux2-klein-9B`](https://huggingface.co/Comfy-Org/flux2-klein-9B) (ungated)
- encodeur de texte — [`bartowski/mlabonne_Qwen3-8B-abliterated-GGUF`](https://huggingface.co/bartowski/mlabonne_Qwen3-8B-abliterated-GGUF)
  (Qwen3-8B **abliteré / non censuré**, via `--llm`, déchargé en RAM)

> Klein est un modèle d'**édition** : fournissez une image de référence (`-r`)
> et décrivez la modification dans le prompt. L'encodeur abliteré réduit les
> refus ; pour un encodeur standard, déposez un `Qwen3-8B` GGUF dans
> `models/custom/` et choisissez-le comme *Encodeur (local)*.

### Optimisation automatique
L'application détecte votre GPU (via `nvidia-smi`) et votre RAM, puis choisit seul :
- **la quantification** du modèle de diffusion selon la VRAM
  (`<8 Go → Q4_K_S`, `8–12 → Q4_K_M`, `12–16 → Q5_K_M`, `16–24 → Q6_K`, `≥24 → Q8_0`) ;
- **la quantification de l'encodeur** selon la RAM (il est déchargé en RAM, donc
  sans coût VRAM) ;
- **les optimisations** : flash-attention (Turing/RTX 20xx et plus), offload CPU,
  VAE tiling, CLIP/VAE sur CPU — activées progressivement quand la VRAM diminue ;
- carte Pascal (GTX 10xx) → flash-attention désactivé automatiquement.

Multi-GPU : la plus grosse carte est utilisée par défaut, modifiable dans
**Réglages**. Tout est surchargeable manuellement (mode auto décochable).

### Presets & styles
Un menu **Préréglage** propose des combos éprouvés (Flux.2 Klein → 4 pas /
CFG 1.0 / **euler + simple**). L'accordéon **Prompt système / style** permet
d'enregistrer des préfixes de style réutilisables (persistés dans `userdata/`).

### LoRA
Déposez vos `.safetensors` / `.gguf` dans le dossier **`loras/`**, puis
sélectionnez-les (jusqu'à 2) avec leur poids dans l'onglet Génération. La syntaxe
`<lora:nom:poids>` est transmise au moteur.

### Upscale (onglet ✨) — deux méthodes
- **⚡ PiD** (recommandé) — décodeur de diffusion en espace pixel de NVIDIA,
  **natif stable-diffusion.cpp** : encode l'image puis décode/agrandit vers ~2K
  en **4 pas, 100% GPU, sans PyTorch ni tuiles**. Rapide. Modèles :
  [`Comfy-Org/PixelDiT`](https://huggingface.co/Comfy-Org/PixelDiT) (décodeur +
  Gemma-2-2B) + VAE FLUX.1. (doc : `leejet/stable-diffusion.cpp` → `docs/pid.md`)
- **🎨 Créatif (SDXL + ControlNet Tile)** — façon Magnific : ré-invente le détail
  **par tuiles** (diffusers, ~9 Go, 100% GPU). Plus lent, plus « créatif ».
  Curseurs *créativité* / *fidélité*. ×2 conseillé ; sortie plafonnée à 4096 px.

### Toolkit (onglet 🧰)
Outils PyTorch installables en 1 clic (modèles téléchargés depuis Hugging Face) :
- **Profondeur** — *Depth Anything V2* (carte de profondeur).
- **Sans arrière-plan** — *RMBG-1.4* (détourage → PNG transparent ; licence non
  commerciale).

### Prompts sauvegardés
Chaque image générée est accompagnée d'un `.txt` (style A1111) dans `outputs/`
avec le prompt, le négatif, le modèle, le sampler/scheduler, le seed et les
dimensions.

---

## Partager avec des collègues (réseau local)

Vos collègues peuvent générer des images depuis leur **Mac/PC**, en utilisant
**votre** machine (et ses GPU), sans rien installer : juste un lien dans Safari.

1. Sur votre PC, lancez **`run-lan.bat`** (au lieu de `run.bat`).
2. L'adresse à partager s'affiche, par ex. :
   ```
   →  http://192.168.1.42:7860
   ```
3. Vos collègues (sur le **même Wi-Fi/réseau**) ouvrent cette adresse dans leur
   navigateur. C'est tout.

Options :
- **Mot de passe** : `run-lan.bat --auth nom:motdepasse` (demandé à la connexion).
- **Pare-feu** : au premier lancement, Windows peut demander d'autoriser Python —
  acceptez (réseaux privés). Sinon, autorisez le port 7860 dans le pare-feu.
- **Raccourci sur le Mac** : dans Safari, *Partager → Ajouter au Dock* (ou un
  marque-page) pour un accès « façon application ».

> Les générations tournent **sur votre PC** : ne l'éteignez pas pendant l'usage.
> Une seule génération à la fois est traitée (file d'attente automatique).

---

## Distribuer à vos amis (paquet portable)

Pour partager le GUI sans que vos amis aient à installer quoi que ce soit :

1. Sur une machine où **tout fonctionne déjà** (Python + moteur `bin\` en place),
   lancez **`make_portable.bat`**.
2. Cela crée `GEN-Ai-Image-Workshop-portable.zip` contenant le code, le
   **Python portable** et le **moteur** — mais pas les modèles.
3. Vos amis **décompressent** et lancent **`run.bat`**. Aucun téléchargement
   GitHub : ils récupèrent seulement les **modèles** depuis l'onglet
   Bibliothèque (via Hugging Face).

Ainsi, même si le réseau de l'un d'eux filtre GitHub, ça marche — le moteur est
déjà inclus dans le ZIP.

---

## Architecture

```
app.py                       # entrée Gradio
config/models.yaml           # bibliothèque : sources, défauts, reco (source de vérité)
atelier/
  settings.py                # chemins + préférences persistées (userdata/)
  hardware.py                # détection GPU/RAM + profils d'optimisation
  registry.py                # catalogue, résolution des fichiers, statut, reco
  downloader.py              # téléchargement HF à la demande
  styles.py                  # presets de prompt système / style
  engine/
    sdcpp.py                 # construction/exécution des commandes sd-cli (+ LoRA)
    generate.py              # pipeline de génération (modèle + matériel + LoRA)
    tools.py                 # outils PyTorch (profondeur, détourage, upscale) en sous-process
  ui/
    theme.py                 # thème clair moderne + CSS
    generate_tab.py · creative_tab.py · library_tab.py · toolkit_tab.py · settings_tab.py
scripts/
  get_sdcpp.py               # télécharge le binaire stable-diffusion.cpp
  _torch_setup.py            # helpers d'installation PyTorch CUDA (partagés)
  setup_tools.py             # installe les outils PyTorch (depth, rembg, upscale SDXL)
  tools/run_*.py             # runners d'inférence (sous-process : depth, rembg, upscale)
```

---

## Dépannage

- **« Binaire sd-cli introuvable »** → relancez `install.bat`, ou téléchargez le
  moteur manuellement. ⚠️ Sur une install portable Windows, `python` global
  n'existe pas : utilisez le Python embarqué :
  `python\python.exe scripts\get_sdcpp.py --variant cuda`
  (récupère la build **win-cuda12** *et* le runtime **cudart** côte à côte).
- **« Aucun GPU NVIDIA détecté »** → vérifiez les pilotes / `nvidia-smi`.
- **Modèle « à télécharger »** → onglet Bibliothèque → bouton Télécharger.
- **Out of memory** → Réglages : baissez la quantification, activez offload/tiling,
  ou réduisez la résolution / le facteur d'upscale créatif.
- **Upscale créatif en OOM** → baissez le facteur (×2) ou la résolution source.
