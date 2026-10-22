# Fibo 2.0 dans TurboSlop

Lancer **`update.bat`**, puis redémarrer TurboSlop. L'installation du moteur
prend la dernière release sd.cpp disponible et refuse une release officielle
antérieure à **985**, qui a introduit Fibo 2.0. La génération vérifie aussi la
version avant de résoudre les poids : le seul fait d'avoir les flags habituels
sur un ancien binaire ne prouve pas qu'il connaît cette architecture.

Dans **Models**, rechercher **Fibo**, choisir la variante puis **Download
selected image weights**. Les deux entrées sont indépendantes : télécharger
Dev ne télécharge pas Edit Turbo, et inversement.

| Onglet Images | Usage | Étapes | CFG | Sampler / scheduler |
| --- | --- | ---: | ---: | --- |
| Fibo 2.0 Dev | Génération texte → image et édition avec références | 20 | 5 | Euler / Auto |
| Fibo 2.0 Edit Turbo | Édition uniquement ; au moins une référence | 4 | 1 | Euler / Auto |

Ces paramètres reprennent les [recettes du serveur sd.cpp](https://github.com/leejet/stable-diffusion.cpp/tree/a77d3623180b2bc1886f3dd04c0aa04156a987ef/examples/server/recipes).
**Flow shift : 0** laisse le moteur choisir son décalage natif de **3**.
Les dimensions doivent être des multiples positifs de **16**. Commencer à
1024×1024 et conserver le placement mémoire automatique sur les cartes 12 Go.
Les négatifs sont disponibles pour Dev ; à CFG 1 ils n'ont pas d'effet.

## Édition et captions JSON

Dans **Reference images**, importer une image et décrire la modification.
Deux autres images ont leurs champs propres ; **Additional references** permet
d'aller jusqu'à **10 références ordonnées dans TurboSlop**. Il s'agit d'une
limite pratique de l'interface, pas d'une limite architecturale annoncée.
Leur ordre correspond à `<image_1>`, `<image_2>`, etc. Les entrées sont copiées
en PNG temporaires puis supprimées après la génération ; les originaux restent
intacts. Ajouter des références augmente le coût mémoire.

Le moteur applique automatiquement le preset de référence `fibo2`. Avec une
référence, il redimensionne pour couvrir la sortie et recadre au centre ;
choisir un rapport d'aspect proche de l'entrée limite la perte des bords. Avec
plusieurs références, il choisit des dimensions proches d'un mégapixel selon
leur rapport d'aspect. Le VAE et l'encodeur vision voient les mêmes recadrages.

Les prompts en langage naturel et les objets JSON sont acceptés. Par exemple :

```json
{
  "structured_caption": {
    "short_description": "A clean travel poster with a turquoise background",
    "text_render": []
  },
  "edit_instruction": "Replace the red background with turquoise"
}
```

TurboSlop vérifie la syntaxe puis transmet l'objet **sans ajouter de styles**
ni de texte système. Le parsing des captions est effectué par le moteur natif.
L'intégration sd.cpp utilise le chemin caption-only : elle ne lance pas Bria
VGL, ne développe pas automatiquement le prompt et ne fabrique pas d'images de
glyphes auxiliaires. Un JSON structuré ne garantit donc pas à lui seul un
rendu typographique parfait.

Edit Turbo sans référence est refusé avant le lancement du moteur, y compris
quand une entrée `init_image` classique lui est envoyée. Le masque natif Fibo
n'est pas relié à `--mask` dans sd.cpp : les appels masqués et ADetailer sont
refusés explicitement. Fibo n'apparaît pas dans le choix ADetailer. Edit Turbo
est aussi exclu des parcours qui génèrent sans image de référence, comme le
benchmark texte seul et le générateur d'image initiale pour la 3D.

## Poids et partage

Les checkpoints [Comfy-Org/Fibo-2.0](https://huggingface.co/Comfy-Org/Fibo-2.0)
ont été vérifiés à la révision
`b2bffaf57d48825a6989b986f24d5cfb8c71231b` :

| Composant | Fichier | Taille |
| --- | --- | ---: |
| Dev par défaut | `diffusion_models/fibo2_dev_int8_convrot.safetensors` | 8 408 562 474 octets |
| Edit Turbo par défaut | `diffusion_models/fibo2_edit_turbo_int8_convrot.safetensors` | 8 408 562 474 octets |
| Dev BF16, sélection explicite | `diffusion_models/fibo2_dev_bf16.safetensors` | 16 452 934 194 octets |
| Edit Turbo BF16, sélection explicite | `diffusion_models/fibo2_edit_turbo_bf16.safetensors` | 16 452 934 194 octets |

Les INT8 sont chargés directement ; aucune conversion ou copie BF16 n'est
créée. Le chemin accéléré CUDA ConvRot demande une capacité de calcul 7.5+
(RTX 2080 Ti / RTX 3060 notamment). Les backends non compatibles peuvent
revenir au calcul CPU. Aucun environnement Torch supplémentaire n'est installé.

Le VAE conserve le dossier historique commun
`Comfy-Org/flux2-klein-9B/split_files/vae/flux2-vae.safetensors` ; son SHA-256
est identique au VAE republié avec Fibo. L'encodeur est celui déjà utilisé par
Krea 2 et Flux.3 : `Qwen/Qwen3-VL-4B-Instruct-GGUF`, avec la quantification
choisie par le profil ou l'utilisateur. Q4_K_M pèse 2,50 Go ; Q8_0, 4,28 Go.
Un encodeur compatible déjà présent est conservé.

Le projecteur vision conserve également le chemin Krea 2
`mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf` (453 974 304 octets). Dev peut générer
sans ce fichier, mais son édition le demande ; il est requis pour qu'Edit
Turbo soit marqué prêt. Il n'est chargé que lorsqu'une référence est passée.
Le tokenizer Qwen est intégré au GGUF et ne demande pas de fichier séparé.

Le placement automatique tient compte du DiT, du VAE et du projecteur pour
l'édition. Il peut garder l'image sur le GPU avec les poids texte en RAM, ou
utiliser l'auto-fit natif quand le budget ne suffit pas. BF16 nécessite de
décharger davantage sur une carte 12 Go. Les tailles des checkpoints ne
représentent pas la mémoire totale du traitement.

## Licence et vérification

Les poids publiés sont **CC BY-NC 4.0** : l'usage commercial nécessite un
[accord distinct avec BRIA](https://platform.bria.ai/licensing/pricing?model=fibo-2).
Le catalogue et les onglets de génération affichent cette condition.

Les tests couvrent le partage des fichiers, INT8/BF16, le modèle/encodeur/VAE
réellement transmis au CLI, le projecteur conditionnel, l'ordre des références,
la conservation des captions JSON, les limites de version/dimensions/masque et
le placement d'une RTX 3060 12 Go avec des fichiers factices de taille exacte.
Le binaire officiel Linux `master-987-a77d362` a été exécuté pour vérifier les
options CLI. Les tests navigateur simulent la génération. La qualité, la
vitesse GPU et l'inférence complète sous Windows n'ont pas été mesurées ici.
