# 🟢 GEN.Ai Image Workshop

Studio d'inférence d'images **local**, moderne et léger, pensé pour les artistes.
Génère avec **Flux.2 Klein 9B** et **Krea 2 Turbo** (GGUF), avec **catalogue de
modèles à la demande**, **optimisations automatiques selon votre carte RTX et
votre RAM**, **LoRA**, **presets sampler/scheduler**, **résolutions natives par
modèle**, **styles enregistrés**, **améliorateur de prompt (IA)** et un
**Toolkit** (profondeur, détourage, SAM).

Aucun ComfyUI, aucune usine à gaz : une interface web claire.

| Onglet | Rôle |
|---|---|
| 🟣 **Flux.2 Klein** | rapide (4 pas) · text-to-image & **édition multi-référence**, presets, styles, LoRA |
| ⚡ **Krea 2 Turbo** | photoréalisme rapide (8 pas, GGUF, encodeur Qwen3-VL, VAE WAN 2.1) |
| 📚 **Catalogue de modèles** | recommandations selon le matériel, téléchargement / suppression à la demande |
| 🧰 **Toolkit** | profondeur · suppression d'arrière-plan · détourage au clic (SAM) |
| 🎬 **Vidéo (LTX-2.3)** | texte→vidéo, image→vidéo, début→fin (sd.cpp, ⚠️ 22B très lourd) |
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

### Optimisation par génération de carte (1 clic)
Dans **Réglages**, quatre boutons appliquent un préréglage curaté selon la
génération : **RTX 20xx** (Turing), **RTX 30xx** (Ampere), **RTX 40xx** (Ada),
**RTX 50xx** (Blackwell). Le préréglage est calé sur la **VRAM réelle** de la
carte sélectionnée (quantification + offload + VAE tiling), avec un léger biais
*vitesse* sur les anciennes générations et *qualité* sur les récentes. Pratique
pour basculer vite d'une machine à l'autre (ex. 2080 Ti → Q4_K_M, 4090 → Q8_0).

### Presets & styles
Un menu **Préréglage** propose des combos éprouvés (Flux.2 Klein → 4 pas /
CFG 1.0 / **euler + simple**). L'accordéon **Prompt système / style** permet
d'enregistrer des préfixes de style réutilisables (persistés dans `userdata/`).

### LoRA
Déposez vos `.safetensors` / `.gguf` dans le dossier **`loras/`**, puis
sélectionnez-les (jusqu'à 2) avec leur poids dans l'onglet Génération. La syntaxe
`<lora:nom:poids>` est transmise au moteur.

### Résolutions natives
Chaque modèle propose des **formats alignés sur ses résolutions d'entraînement**
(le rendu est meilleur dessus) :
- **Flux.2 Klein** : ~1 MP, grille 32 px — 1024², 1248×832, 1184×880, 1392×752,
  1568×672… (+ option 2K).
- **Krea 2** : famille 1024 (multiples de 64) — 1024², 1216×832, 1152×896,
  1344×768… (+ option 2K).

### Améliorateur de prompt (IA)
Bouton **« ✨ Améliorer le prompt »** dans l'onglet de génération : un petit LLM
(*Qwen2.5-3B-Instruct*, PyTorch ~6 Go, installable en 1 clic) réécrit votre idée
en un prompt **anglais** détaillé (sujet, lumière, cadrage, style), prêt à
générer. Le modèle est chargé puis déchargé à chaque appel → **aucun conflit de
VRAM** avec la génération.

### Édition multi-référence (Flux.2 Klein)
Dans l'accordéon *Images de référence*, chargez l'image à éditer et jusqu'à **2
images de référence** supplémentaires, puis décrivez la modification ou la
combinaison voulue dans le prompt (chaque image est passée au moteur via `-r`).

### Toolkit (onglet 🧰)
Outils utilitaires installables en 1 clic (modèles téléchargés depuis Hugging Face) :
- **Profondeur** — *Depth Anything V2* (carte de profondeur).
- **Sans arrière-plan** — *RMBG-1.4* (détourage → PNG transparent ; licence non
  commerciale).
- **Détourage au clic** — *Segment Anything* (`facebook/sam-vit-base`).
- **Agrandir (ESRGAN)** — upscale **simple** par réseau ESRGAN GGUF, **natif
  sd.cpp** (`--mode upscale`) : déterministe, **100% GPU**, aucun PyTorch ni
  prompt. Choix parmi tous les modèles de `wbruna/upscalers-sdcpp-gguf`
  (×2/×4 selon le modèle ; « Répéter ×2 » enchaîne deux passes).

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
    tools.py                 # outils PyTorch (profondeur, détourage, SAM, prompt) en sous-process
  ui/
    theme.py                 # thème clair moderne + CSS
    generate_tab.py · library_tab.py · toolkit_tab.py · video_tab.py · settings_tab.py
scripts/
  get_sdcpp.py               # télécharge le binaire stable-diffusion.cpp
  _torch_setup.py            # helpers d'installation PyTorch CUDA (partagés)
  setup_tools.py             # installe les outils PyTorch (depth, rembg, sam, enhance)
  tools/run_*.py             # runners d'inférence (sous-process : depth, rembg, sam, enhance)
```

---

## Dépannage

- **« Binaire sd-cli introuvable »** → relancez `install.bat`, ou téléchargez le
  moteur manuellement. ⚠️ Sur une install portable Windows, `python` global
  n'existe pas : utilisez le Python embarqué :
  `python\python.exe scripts\get_sdcpp.py --variant cuda`
  (récupère la build **win-cuda12** *et* le runtime **cudart** côte à côte).
- **« Aucun GPU NVIDIA détecté »** → vérifiez les pilotes / `nvidia-smi`.
- **Modèle « à télécharger »** → onglet Catalogue de modèles → bouton Télécharger.
- **Out of memory** → Réglages : baissez la quantification, activez offload/tiling,
  ou réduisez la résolution.
