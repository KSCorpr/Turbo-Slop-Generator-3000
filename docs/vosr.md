# VOSR 2.0 dans TurboSlop

Lancer `update.bat`, redémarrer puis ouvrir **Models** et rechercher **VOSR**.
Sélectionner **VOSR 2.0 · image restoration → Install / repair selected package**.
Le même installateur est disponible dans **Tools & 3D → VOSR 2.0 upscale**.

Importer une image, choisir **×1**, **×2** ou **×4**, puis **Restore with VOSR**.
L'aperçu indique la taille exacte après application de l'orientation EXIF.
Le résultat est un PNG téléchargeable. Le canal alpha éventuel est agrandi avec
Lanczos et réattaché ; VOSR restaure les couleurs RGB, sans générer d'alpha.
Les images animées et TIFF multipages sont traités sur leur première image.
Les vidéos ne sont pas prises en charge.

## Réglages

| Contrôle | Par défaut | Fonction |
| --- | --- | --- |
| Facteur | ×4 | ×1 restaure sans agrandissement ; ×2 et ×4 multiplient chaque dimension. |
| Précision | FP16 | DiT et DINO en FP16 ; VAE toujours en FP32. BF16 nécessite un GPU compatible. |
| Color alignment | Wavelet | Conserve les basses fréquences de couleur de la source. AdaIN ajuste les statistiques ; None garde les couleurs du modèle. |
| DiT tile / overlap | 512 / 64 px | Tuiles dans l'image **après** agrandissement, avec fusion gaussienne des recouvrements. |
| VAE tile / overlap | 512 / 64 px | Limite les activations de l'encodage/décodage FP32. 1024/128 est un réglage amont courant, mais 512/64 est retenu ici pour les cartes 11–12 Go. |
| Offload | Activé | DiT, DINO et VAE passent sur le GPU à tour de rôle ; les autres restent en RAM. |
| Seed | 42 | Bruit initial reproductible avec le même environnement et les mêmes paramètres. |

Si la VRAM manque, conserver l'offload et réduire la tuile VAE à **256**,
puis éventuellement la tuile DiT. Le recouvrement doit rester inférieur à la
tuile. Les sorties sont limitées à 32 millions de pixels pour borner les
allocations pleine image. Les tuiles réduisent les activations, mais le bruit,
les latents et les accumulateurs ont toujours une taille liée à l'image finale.
Les 12 Go d'une RTX 3060 ne constituent donc pas une garantie pour toute entrée.

VOSR est une restauration générative en un passage : il peut reconstruire ou
modifier les textures, les traits fins et le texte. Comparer les résultats à
100 % sur les images où la fidélité est essentielle.

## Installation et stockage

L'environnement `tools_repo/vosr/.venv` utilise Python 3.12, **Torch
2.7.1+cu126**, safetensors et einops. Il ne modifie ni le Python principal ni
les environnements des autres outils. Il n'installe ni ComfyUI, ni Diffusers,
ni xFormers, ni compilateur Triton. Le CUDA Toolkit n'est pas nécessaire ;
le pilote doit prendre en charge le runtime CUDA 12.6.

Les modules neuronaux et le traitement par tuiles proviennent de
[`ylchen333/ComfyUI-VOSR2`](https://github.com/ylchen333/ComfyUI-VOSR2), commit
`6a21810b3f4f1a0ffc477c55330db0ac03c671f0`. L'installateur conserve la licence
et le code amont, remplace explicitement les couches `comfy.ops` par leurs
équivalents `torch.nn`, retire l'enregistrement des nœuds et fournit un journal
de progression. Les opérations neuronales et la fusion des tuiles sont
conservées. Le padding de bord utilise `replicate` pour accepter les images
très étroites. La construction sur `meta` et le chargement strict avec
`assign=True` évitent une seconde allocation initialisée de tous les poids.

Le triplet de poids est figé sur
[`CSWRY/VOSR`](https://huggingface.co/CSWRY/VOSR/tree/f24b3061b7f350b81e1907bdcfc27f71fb4ff3f3),
révision `f24b3061b7f350b81e1907bdcfc27f71fb4ff3f3`, dans **un seul dossier**
`tools_repo/vosr/models` :

| Fichier | Octets |
| --- | ---: |
| `VOSR2/args.json` | 816 |
| `VOSR2/checkpoints/ema_model.safetensors` | 5 576 385 424 |
| `Qwen-Image-vae-2d/config.json` | 811 |
| `Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors` | 178 418 892 |
| `torch_cache/checkpoints/dinov2_vitl14_pretrain.pth` | 1 217 586 395 |

Total : environ **6,97 Go**, en plus du runtime. Le VAE **2D** fait partie
du format latent de VOSR ; les VAE de Qwen Image Turbo, Qwen Layered et Flux
ne peuvent pas le remplacer. DINO conserve son checkpoint original, chargé
avec `weights_only=True` et mmap ; aucune copie safetensors supplémentaire.
Les fichiers identiques déjà indexés sont réutilisés par le téléchargement
commun, qui respecte le plafond de bande passante et reprend les transferts.

L'installateur teste CUDA et SDPA avant de télécharger les modèles. Il ne
marque l'installation complète qu'après les cinq transferts. Le traitement
ne télécharge rien : il fonctionne avec les fichiers locaux, en mode hors
ligne. **Stop** arrête aussi les processus enfants. Une erreur, une annulation
ou une sortie invalide n'exporte pas d'image partielle.

## Vérification

Les tests d'intégration couvrent téléchargement, routage catalogue, isolation,
contrôles, sélection GPU, chemins Unicode, orientation EXIF, arrêt et validation
des sorties. `tests/verify_vosr_runtime.py --source tools_repo/vosr/source`
compare les **849 tenseurs DiT et 186 tenseurs VAE** aux en-têtes officiels,
vérifie le chargement réel des deux formats et exécute le pipeline neuronal
avec de petits réseaux aléatoires sur CPU. Ce contrôle vérifie l'exécution et
les dimensions ; il ne mesure pas la qualité des poids préentraînés, la vitesse
GPU ni la compatibilité d'exécution complète sous Windows.
