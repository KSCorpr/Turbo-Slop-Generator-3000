# SeedVR2, LTX 2.5 et capture Gaussian Splatting

Ces trois fonctions sont disponibles dans `main` et dans le téléchargement Studio.
Le canal de mise à jour suit `main`.

## Utilisation

### SeedVR2

Ouvrir **Tools & 3D → SeedVR2 upscale**, puis **Install / repair SeedVR2**.
Importer une image ou une vidéo, choisir 3B (Q8) ou 7B (Q4), sélectionner **×2**
ou **×4**, régler **Added detail strength (%)**, puis **Restore**.
Le premier traitement avec une force supérieure à 0 télécharge automatiquement
le DiT choisi et le VAE.
Les fichiers importés sont copiés sans recompression sous un nom temporaire
simple, pour contourner les problèmes OpenCV avec les noms Unicode sous
Windows et conserver l'entrée pendant les traitements longs. Cette copie
est supprimée à la fin, même en cas d'échec ou d'annulation.
L'image source s'affiche dès l'import, avec ses dimensions. Cet aperçu est
réduit uniquement pour l'affichage ; l'entrée du moteur reste intacte.
L'image restaurée et son téléchargement apparaissent à la fin du traitement.
La CLI SeedVR2 utilisée ici ne produit pas d'aperçu intermédiaire à chaque pas.
**×2 / ×4** calcule automatiquement la résolution depuis les dimensions de
l'image ou des images de la vidéo. Exemple : 1024×768 → ×2 : 2048×1536 ;
×4 : 4096×3072. La taille cible s'affiche dès l'import d'une image ; les dimensions
d'une vidéo sont lues au lancement, sans transcodage pendant l'import.
**Custom** affiche le champ de résolution du petit côté (256–8192 pixels).
Si un facteur dépasse cette plage, le programme demande une taille personnalisée
au lieu de réduire le facteur silencieusement. Le rapport d'aspect est conservé,
sous réserve de l'alignement des dimensions imposé par le modèle et l'encodage vidéo.

Le curseur **Added detail strength (%)** dose le résultat de façon explicite :
- **0 %** : agrandissement Lanczos de la source, sans charger SeedVR2 ;
- **100 %** : résultat complet du moteur, comme avant ;
- **entre les deux** : mélange de la source agrandie et du résultat restauré.
  Par exemple, 25 % conserve 25 % du résultat SeedVR2 et 75 % de la source agrandie.

Ce réglage réduit aussi les autres modifications de la restauration (couleurs,
textures et formes) ; il ne compte pas les détails inventés et ne garantit pas
la fidélité géométrique. Les paramètres de bruit natifs restent à zéro : ce bruit
sert à varier ou adoucir la reconstruction, pas à régler précisément sa force.
À 100 %, les images restent les PNG du moteur et la vidéo n'est pas réencodée
pour ce réglage. Les vidéos à force intermédiaire passent par un mélange FFmpeg
encodé en H.264, puis la piste audio éventuelle est réattachée en AAC.

Le profil utilise le déchargement CPU, BlockSwap, l'attention PyTorch SDPA et des
tuiles VAE de 512 pixels. Les vidéos passent par lots de 5 images et segments de
33 images, avec recouvrement temporel. La piste audio éventuelle est réattachée
en AAC ; les images restaurées ne sont pas réencodées pendant ce raccord.
Un modèle génératif peut modifier les petits détails : comparer les résultats
sur les sources professionnelles, notamment typographie et architecture.

### LTX 2.5

Ouvrir **Video → LTX 2.5**, choisir la précision, puis **Download / prepare LTX 2.5**.
Le programme télécharge le modèle dev GGUF, le VAE convolutionnel et l'encodeur
Gemma 4 spécifique à LTX. L'encodeur BF16 est converti une fois en Q4_0 via sd.cpp.
Il reste conservé, pour éviter de supprimer un fichier que l'utilisateur pourrait
vouloir employer dans un autre outil. Prévoir au moins 60 Go libres pour commencer.
Les téléchargements soumis à conditions utilisent l'authentification Hugging Face
déjà configurée sur la machine ; les termes du modèle doivent avoir été acceptés.

Commencer par 512×288, 33 images, sur les GPU de 11–12 Go. Le modèle dépasse la
VRAM : auto-fit et la RAM système sont indispensables. Les autres choix sont
768×448 / 960×544 et 65 / 97 / 121 images. Ce sont des profils à valider sur le GPU,
pas une garantie de temps de calcul ou d'absence de manque de mémoire.

Cette intégration utilise **dev, 20 étapes, Euler, CFG 3**, une première image
facultative et une dernière image facultative si la première est fournie.
Elle produit une vidéo MP4 sans audio et un fichier JSON de paramètres.
Elle n'utilise pas le décodeur par diffusion, le modèle distilled, les passes DFR,
les images clés intermédiaires ni les sorties HDR/EXR du pipeline Python LTX.
Si l'encodage MP4 échoue, l'AVI reste disponible via la récupération de vidéos
dans l'onglet MiniMax H3.

### Capture 3D

Ouvrir **Tools & 3D → Capture → splats**, puis **Install / repair COLMAP + Brush**.
Le binaire de Brush 0.3 porte le nom `brush_app.exe` sous Windows et `brush_app`
sous Linux/macOS. Si une installation précédente a affiché « Brush executable
missing from release archive », mettre le code à jour, relancer TurboSlop puis
cliquer de nouveau sur **Install / repair COLMAP + Brush**.
Importer **soit** 8–300 photos, **soit** une vidéo. Pour un objet, tourner autour
d'un sujet immobile ; pour une scène, déplacer la caméra avec beaucoup de
recouvrement. Éviter reflets, surfaces uniformes, flou, zoom variable et objets
mobiles. Un plateau tournant devant un fond immobile demande des masques qui ne
sont pas automatisés dans cette première intégration.

Les photos sont copiées, orientées selon EXIF et réduites à 1600 pixels maximum.
Les vidéos fournissent au maximum les 300 premières images échantillonnées au
rythme choisi. COLMAP calcule les caméras sur CPU, sélectionne la composante
contenant le plus de vues alignées, puis corrige la distorsion.
Le journal indique les vues retenues et celles qui ne contribuent pas.

Brush 0.3 entraîne ensuite un modèle avec au maximum 500 000 splats pour un objet
ou 1 million pour une scène. Les profils Preview / Standard / Fine exécutent
5 000 / 15 000 / 30 000 étapes. L'objet n'est pas détouré automatiquement.
L'aperçu Gradio sait afficher les Gaussian splats PLY ; le téléchargement conserve
le PLY complet. Ce fichier n'est pas un maillage avec UV pour 3ds Max.

Les données restent dans `outputs/capture-*.work/` : images de travail, base
COLMAP, caméras, images corrigées, rapport d'alignement et export. Une capture
peut occuper plusieurs Go. Ne supprimer ce dossier qu'après avoir récupéré
l'export et les données à conserver. La reprise après interruption nécessite
actuellement une utilisation manuelle des fichiers conservés ; le bouton de
reconstruction lance un nouveau travail.

Brush choisit son GPU via WebGPU. L'option avancée `Brush adapter index` correspond
à `CUBECL_DEFAULT_DEVICE`, pas forcément à l'index NVIDIA utilisé par les autres
outils. Vérifier le GPU annoncé dans le journal, en particulier avec un iGPU.

## Isolation et versions

- SeedVR2 CLI : `numz/ComfyUI-SeedVR2_VideoUpscaler` au commit
  `4490bd1f482e026674543386bb2a4d176da245b9`.
- Environnement SeedVR2 séparé : Python 3.12 géré par uv, Torch 2.7.1 /
  torchvision 0.22.1 CUDA 12.6. Aucune modification du Torch des outils existants.
- COLMAP : wheel `pycolmap==3.12.0`, exécution CPU, environnement séparé.
- Brush : binaire officiel `v0.3.0`, empreinte SHA-256 vérifiée avant extraction.
- LTX : modèle `vantagewithai/LTX-2.5-GGUF`, composants `Lightricks/LTX-2.5`,
  moteur sd.cpp déjà utilisé par l'application. Licence des poids : LTX Community.

Les boutons d'installation et d'inférence partagent la file GPU du studio.
Stop termine aussi les sous-processus (FFmpeg, installateur et moteurs) et empêche
le passage à l'étape suivante. Aucun serveur tiers ni service payant n'est utilisé
pour le traitement. Les téléchargements initiaux nécessitent Internet.

## Validation

Tests automatisés : commandes, fichiers incompatibles, préparation des photos,
chemins des exports, interruption de processus réels et isolation de la file GPU.
La wheel Windows de pycolmap 3.12.0/Python 3.12 a été vérifiée disponible.
Extraction SIFT et correspondances COLMAP ont été exécutées sur CPU.
Les tests d'interface avec moteurs simulés vérifient le câblage, pas la qualité
des modèles. Les inférences SeedVR2/LTX et l'entraînement Brush sur GPU Windows
nécessitent encore une validation sur matériel réel.

Sources des interfaces :
- https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler
- https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/ltx2.md
- https://github.com/ArthurBrussee/brush/releases/tag/v0.3.0
- https://github.com/colmap/colmap
