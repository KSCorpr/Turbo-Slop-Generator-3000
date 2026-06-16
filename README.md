# 🎨 Atelier

Studio d'inférence d'images **local**, moderne et léger, pensé pour les artistes.
Génère avec **Ideogram 4** et **Z-Image Turbo** au format **GGUF**, avec
**bibliothèque de modèles à la demande**, **optimisations automatiques selon
votre carte RTX et votre RAM**, **LoRA**, et **upscale SeedVR2 / NVIDIA PiD**.

Aucun ComfyUI, aucune usine à gaz : une interface web claire en quatre onglets.

| Onglet | Rôle |
|---|---|
| 🎨 **Génération** | text-to-image & image-to-image, réglages auto par modèle, LoRA |
| 📚 **Bibliothèque** | catalogue, recommandations selon le matériel, téléchargement à la demande |
| 🔍 **Upscale** | SeedVR2-3B ou NVIDIA PiD |
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
(natif CUDA, format GGUF) — pas de PyTorch lourd pour la génération. Les modèles
viennent de :
- Ideogram 4 — [`leejet/ideogram-4-GGUF`](https://huggingface.co/leejet/ideogram-4-GGUF)
  + encodeur [`unsloth/Qwen3-VL-8B-Instruct-1M-GGUF`](https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-1M-GGUF)
- Z-Image Turbo — [`unsloth/Z-Image-Turbo-GGUF`](https://huggingface.co/unsloth/Z-Image-Turbo-GGUF)
  + encodeur [`Qwen/Qwen3-4B-GGUF`](https://huggingface.co/Qwen/Qwen3-4B-GGUF)

### Optimisation automatique
Atelier détecte votre GPU (via `nvidia-smi`) et votre RAM, puis choisit seul :
- **la quantification** du modèle de diffusion selon la VRAM
  (`<8 Go → Q4_K_S`, `8–12 → Q4_K_M`, `12–16 → Q5_K_M`, `16–24 → Q6_K`, `≥24 → Q8_0`) ;
- **la quantification de l'encodeur** selon la RAM (il est déchargé en RAM, donc
  sans coût VRAM) ;
- **les optimisations** : flash-attention (Turing/RTX 20xx et plus), offload CPU,
  VAE tiling, CLIP/VAE sur CPU — activées progressivement quand la VRAM diminue ;
- carte Pascal (GTX 10xx) → flash-attention désactivé automatiquement.

Multi-GPU : la plus grosse carte est utilisée par défaut, modifiable dans
**Réglages**. Tout est surchargeable manuellement (mode auto décochable).

### Réglages auto par modèle
Sélectionner un modèle pré-remplit ses bons réglages (steps, CFG, sampler,
résolution). Ex. Z-Image Turbo → 8 étapes / CFG 1.0 ; Ideogram 4 → 28 / 4.0.

### LoRA
Déposez vos `.safetensors` / `.gguf` dans le dossier **`loras/`**, puis
sélectionnez-les (jusqu'à 2) avec leur poids dans l'onglet Génération. La syntaxe
`<lora:nom:poids>` est transmise au moteur.

### Upscale
Installation séparée (PyTorch) car ce sont de gros modèles :
```bash
python scripts/setup_upscalers.py seedvr2      # SeedVR2-3B (qualité max)
python scripts/setup_upscalers.py nvidia-pid   # NVIDIA PiD (rapide, 4K)
```
- **SeedVR2-3B** — [`ByteDance-Seed/SeedVR2-3B`](https://huggingface.co/ByteDance-Seed/SeedVR2-3B)
- **NVIDIA PiD** — [`nvidia/PiD`](https://huggingface.co/nvidia/PiD) · code [`nv-tlabs/PiD`](https://github.com/nv-tlabs/PiD)

---

## Architecture

```
app.py                       # entrée Gradio (4 onglets)
config/models.yaml           # bibliothèque : sources, défauts, reco (source de vérité)
atelier/
  settings.py                # chemins + préférences persistées (userdata/)
  hardware.py                # détection GPU/RAM + profils d'optimisation
  registry.py                # catalogue, résolution des fichiers, statut, reco
  downloader.py              # téléchargement HF à la demande
  engine/
    sdcpp.py                 # construction/exécution des commandes sd-cli (+ LoRA)
    generate.py              # pipeline de génération (modèle + matériel + LoRA)
    upscalers.py             # SeedVR2 / NVIDIA PiD
  ui/
    theme.py                 # thème sombre moderne + CSS
    generate_tab.py · library_tab.py · upscale_tab.py · settings_tab.py
scripts/
  get_sdcpp.py               # télécharge le binaire stable-diffusion.cpp
  setup_upscalers.py         # installe SeedVR2 / NVIDIA PiD
  upscalers/run_*.py         # runners d'inférence des upscalers
```

---

## Dépannage

- **« Binaire sd-cli introuvable »** → `python scripts/get_sdcpp.py --variant cuda`
  (ou `install.bat`).
- **« Aucun GPU NVIDIA détecté »** → vérifiez les pilotes / `nvidia-smi`.
- **Modèle « à télécharger »** → onglet Bibliothèque → bouton Télécharger.
- **Out of memory** → Réglages : baissez la quantification, activez offload/tiling,
  ou réduisez la résolution (Z-Image Turbo + ≤ 768 px sur petite VRAM).
- **Upscaler en erreur** → l'API des dépôts officiels évolue ; ajustez
  `scripts/upscalers/run_seedvr2.py` ou `run_pid.py` selon leur README.
