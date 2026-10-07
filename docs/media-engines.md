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

Le profil utilise le déchargement CPU, BlockSwap et des tuiles VAE de 512 pixels.
Sans optimisation installée, **Attention → Auto** utilise l'attention PyTorch
SDPA. Les vidéos passent par lots de 5 images et segments de
33 images, avec recouvrement temporel. La piste audio éventuelle est réattachée
en AAC ; les images restaurées ne sont pas réencodées pendant ce raccord.
Un modèle génératif peut modifier les petits détails : comparer les résultats
sur les sources professionnelles, notamment typographie et architecture.

#### Optimisations Windows

Après l'installation de SeedVR2, cliquer sur **Install optimizations** dans le
même onglet. Le bouton utilise le GPU choisi dans les réglages du studio et
installe uniquement dans l'environnement Python de SeedVR2. Il conserve Torch
2.7.1 / CUDA 12.6, vérifie les empreintes SHA-256 des wheels et ne lance aucune
compilation de paquet ni installation de Visual Studio / CUDA Toolkit.
Triton compile néanmoins ses petits kernels au premier usage ; cette première
vérification peut prendre du temps. **Stop** interrompt aussi l'installation.

SeedVR2 sélectionne explicitement **TinyCC**, livré avec Triton, et sa chaîne
CUDA locale pour ses sous-processus. Cela évite qu'un environnement Visual
Studio incomplet (`Failed to find Windows SDK`) ou un autre CUDA Toolkit
installé sur Windows prenne la priorité. Aucun réglage système n'est modifié.
Si une optimisation échoue, le journal affiche le diagnostic et SeedVR2 reste
disponible avec **SDPA** ; l'échec d'une option ne marque plus le moteur entier
comme inutilisable. Après une erreur de SDK, mettre à jour TurboSlop, le
relancer, puis réessayer **Install optimizations**. On peut aussi sélectionner
**Attention → SDPA** pour traiter immédiatement sans ces modules.

| GPU sélectionné | Installation proposée |
| --- | --- |
| RTX 30xx / 40xx, A100, H100 | Triton Windows 3.3.1.post21, SageAttention 2.2.0, FlashAttention 2.8.3 |
| RTX 20xx / GTX 16xx (Turing), calcul BF16 | SDPA ; aucun téléchargement d'optimisation incompatible |
| Turing, calcul FP16 détecté par SeedVR2 | Essai de SageAttention avec Triton Windows 3.2.0.post21 ; FlashAttention 2 exclu |
| GTX 10xx (Pascal) / GPU non pris en charge | Aucune installation automatique ; SDPA conservé |

SeedVR2 choisit sa précision par un petit calcul CUBLAS, et non par la génération
du GPU. Ce calcul peut accepter BF16 sur une RTX 2080 Ti alors que les conversions
BF16 du kernel Triton de SageAttention nécessitent `sm_80` ou supérieur.
TurboSlop utilise le même test de précision et conserve **SDPA** sur Turing
(`sm_75`) dans ce cas, même si SageAttention et Triton sont déjà installés ou
qu'un ancien rapport les indiquait disponibles. Aucune réinstallation du moteur
ni suppression de paquets n'est nécessaire. Après une erreur `ptxas fatal`,
mettre TurboSlop à jour puis relancer l'application ; **Attention → SDPA** permet
aussi de relancer le traitement directement.

Triton 3.3 a supprimé la prise en charge de Turing. Si SeedVR2 détecte FP16,
la combinaison Triton 3.2 / Torch 2.7 reste expérimentale et n'est activée
qu'après un calcul de vérification réussi avec cette précision.
Les wheels Windows SageAttention et FlashAttention sont des builds communautaires
des projets libres ; leurs versions CUDA 12.8 peuvent cohabiter avec le runtime
Torch CUDA 12.6, sous réserve du pilote et du test réel sur la machine.
La wheel FlashAttention est construite pour Torch 2.7.0 : le choix de wheel amont
se fait par version majeure/mineure de Torch, puis nous vérifions le calcul sur
le runtime 2.7.1. Aucun changement de Torch ne compense un test échoué.

**Auto** reteste les API d'attention à longueurs variables utilisées par SeedVR2
sur le GPU sélectionné avant chaque restauration IA. Le test utilise la précision
réelle de SeedVR2, calcule de petits tenseurs, vérifie leurs valeurs et les compare
à SDPA ; un simple import ne
suffit pas. Il choisit SageAttention si disponible, sinon FlashAttention, sinon
SDPA. Le journal indique le GPU, le backend choisi et les raisons d'un repli.
Les choix explicites **SageAttention 2 / FlashAttention 2** effectuent aussi ce
test ; **SDPA** permet d'éviter les modules optionnels. À 0 % de détails ajoutés,
aucun de ces tests GPU ni modèle IA n'est chargé.

Ces backends sont des alternatives, leurs gains ne s'additionnent pas.
Installer Triton n'active pas `torch.compile` : la compilation du modèle reste
désactivée dans le profil avec BlockSwap. Les tests d'attention ne garantissent
ni un gain chronométré ni la compatibilité de toutes les tailles d'image.
Si un backend optionnel échoue pendant la restauration avec une erreur de
compilation PTX/Triton reconnue, TurboSlop supprime sa sortie incomplète et
relance **une seule fois** dans un nouveau processus avec SDPA, en conservant
la source, la seed et les réglages. Une annulation, un manque de VRAM ou une
autre erreur de traitement ne déclenche pas ce repli ; un échec SDPA reste visible.
Le message amont « optimizations check » indique des paquets optionnels absents,
pas une erreur bloquante ; TurboSlop le remplace par le résultat de sa vérification.

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
Pour SeedVR2, la CI Windows vérifie les DLL, la compilation TinyCC et la
compilation PTX du véritable kernel SageAttention à deux tailles de tête :
FP16 sur `sm_75`, refus BF16 attendu sur `sm_75`, et BF16 sur `sm_80`.
Ces vérifications compilent le code sans charger ni exécuter de kernel GPU.
Les tests d'interface avec moteurs simulés vérifient le câblage, pas la qualité
des modèles. Les inférences SeedVR2/LTX et l'entraînement Brush sur GPU Windows
nécessitent encore une validation sur matériel réel.

Sources des interfaces :
- https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler
- https://github.com/woct0rdho/SageAttention/releases/tag/v2.2.0-windows
- https://github.com/triton-lang/triton-windows
- https://github.com/kingbri1/flash-attention/releases/tag/v2.8.3
- https://github.com/Dao-AILab/flash-attention
- https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
- https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/ltx2.md
- https://github.com/ArthurBrussee/brush/releases/tag/v0.3.0
- https://github.com/colmap/colmap
