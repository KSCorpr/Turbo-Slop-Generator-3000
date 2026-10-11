# Remplacer sd.cpp avec un moteur invisible dans l'interface

Oui, le studio peut conserver ses écrans et utiliser ComfyUI uniquement comme
moteur local. L'éditeur de nœuds n'est pas nécessaire : le studio envoie les
paramètres à des workflows internes et reçoit les images, les aperçus et les
événements de progression par API. Cette migration complète n'est pas réalisée
dans la modification du catalogue et de SeedVR2 ConvRot.

## Choix du moteur

| Option | Intérêt pour ce studio | Coût principal |
| --- | --- | --- |
| ComfyUI derrière l'interface existante | Chargeurs quantifiés, modèles récents, gestion mémoire, workflows réutilisables | Versions des extensions à épingler et vérifier |
| Pipelines Diffusers directs | Contrôle Python du pipeline, sans graphe interne | Intégration et optimisation propres à chaque modèle ; certains formats ConvRot nécessitent des extensions |
| sd.cpp conservé provisoirement | Runtime natif existant et GGUF, utile sur matériel ancien | Disponibilité de chaque nouveau modèle dépendante du portage amont |

ComfyUI est le premier candidat à essayer. Cela facilite l'intégration des
modèles et chargeurs publiés pour cet écosystème ; cela ne prouve pas un gain
de vitesse face à sd.cpp. Un moteur résident peut éviter de relire les poids à
chaque image, mais un modèle trop gros ou un changement fréquent de modèle
peut provoquer du déchargement et annuler cet avantage.

## Migration proposée

1. Introduire une interface de backend pour la génération, l'édition, la
   progression, l'annulation et les erreurs. Garder le catalogue, les styles,
   les LoRA, les exports et la galerie du studio.
2. Lancer ComfyUI dans un environnement dédié, lié à 127.0.0.1, avec
   `--disable-auto-launch`. Le studio reste la seule interface affichée.
3. Fournir un workflow interne validé par modèle. Le prompt, les références,
   la résolution, la seed et les réglages autorisés remplissent ces workflows.
   L'utilisateur ne manipule aucun nœud. Qwen Turbo conserve ses huit étapes,
   CFG 1 et sigmas officiels.
4. Utiliser `POST /prompt`, les événements WebSocket, `/history` et `/view`
   pour suivre le travail et récupérer les sorties. Importer les références
   via `/upload/image`. Associer chaque tâche à son identifiant, et gérer
   l'annulation dans un serveur local réservé au studio.
5. Garder les modèles en mémoire lorsque la VRAM le permet ; mesurer séparément
   le premier chargement et les générations suivantes. Sur 12 Go, essayer
   le 3B W4A8 SeedVR2 et les quantifications compatibles avec chaque modèle.
   Une GTX 1080 Ti ne remplace pas les Tensor Cores d'une RTX 3060 et ne doit
   pas recevoir automatiquement les mêmes kernels.
6. Tester les aperçus, les éditions avec alpha, les LoRA, l'outpaint et les
   annulations avant de remplacer chaque parcours. Pour SDXL, conserver le
   redimensionnement, le tuilage et ControlNet Tile.
7. Épingler les versions du moteur, des extensions et de Torch. Tester une
   mise à jour dans un environnement séparé, puis conserver une version
   fonctionnelle pour le retour arrière.

## Mesures avant de choisir définitivement

Comparer sur la même carte, les mêmes poids ou quantifications équivalentes,
la même résolution, la même seed et la même qualité visuelle cible. Relever
temps de chargement, temps par génération à chaud, pic de VRAM, RAM, aperçu,
annulation et comportement lors du changement de modèle. Comparer aussi le
SeedVR2 ConvRot intégré ici au moteur classique. Aucun gain chiffré n'est
annoncé avant ces essais.

Références : [communications du serveur ComfyUI](https://docs.comfy.org/development/comfyui-server/comms_overview),
[routes](https://docs.comfy.org/development/comfyui-server/comms_routes),
[Qwen Image 2.1 Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo),
[SeedVR2 ConvRot](https://huggingface.co/ussoewwin/SeedVR2-ConvRot-INT8-and-w4a8).
