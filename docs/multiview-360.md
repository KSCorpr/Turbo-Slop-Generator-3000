# Dix vues à partir d'une image et panorama 360°

Ouvrir **Tools & 3D → 10 views / 360°**, importer une image et choisir un moteur.
Cliquer **Generate 10 views**. Les images apparaissent progressivement ; le ZIP
contient les dix PNG numérotés, une planche de contact, la référence utilisée et
un `manifest.json` avec les angles, les prompts, les seeds et le statut du lot.
Les fichiers restent dans `outputs/views-…/`. **Stop views** interrompt la tâche
et conserve les vues terminées dans un ZIP partiel.

## Deux mouvements de caméra

- **Object** : huit angles autour du sujet par pas de 45°, une vue en plongée
  à 45° et une vue basse à −20°. La caméra regarde le centre de l'objet.
- **360 scene** : la caméra reste au même endroit ; elle tourne vers huit
  directions par pas de 45°, puis vers le zénith et le nadir. Chaque consigne
  demande une vue perspective carrée de 100° pour ménager des recouvrements.

Le yaw 0 correspond à l'orientation de la référence. Les angles sont des
**consignes génératives**, pas des poses de caméra mesurées ni contraintes par
un moteur 3D. Une image ne contient pas ce qu'il y a derrière un objet ou hors
champ : les modèles inventent ces parties et peuvent modifier la géométrie.

## Compatibilité des moteurs

| Moteur du catalogue actuel | Comment la source est utilisée | Limite |
| --- | --- | --- |
| Flux.2 Klein | Référence native, instruction de changement de vue | Recommandé ; identité et angles restent approximatifs |
| Qwen Image 2.1 | Référence native, instruction de changement de vue | Recommandé ; plus de pas par défaut |
| Krea 2 Turbo | Img2img classique | Pas de LoRA d'édition ajouté automatiquement ; grandes rotations peu fiables |
| Z-Image Turbo | Img2img classique | Le débruitage peut préserver l'angle d'origine ou réinventer le sujet |
| Ming Image | Description de la source par le module local Image → prompt, puis génération texte | Aucune référence visuelle directe ; identité moins fidèle |

Cocher **Advanced and conversion tools** pour afficher la force img2img,
le seed, le module de description et la conversion de vues existantes.
Pour Ming, installer **Image → prompt** depuis le bouton de cet onglet si
nécessaire. Le modèle de vision Qwen2.5-VL-3B déjà employé par le Toolkit est
chargé une fois pour lire l'image, puis déchargé avant les dix générations.
Le bouton installe ce module et ses poids, pas Ming : le moteur image se
télécharge dans **Models**. Une description complémentaire peut être saisie
pour préciser le sujet ou les éléments à garder. L'option de lecture
automatique de la source est également disponible avec les autres moteurs.

Les nouveaux modèles du catalogue emprunteront le chemin déclaré par leurs
capacités : édition native, img2img, ou description visuelle. Tous utilisent
le pipeline image existant, leurs pas/CFG/sampler/scheduler par défaut et les
préférences système. Les poids doivent être installés normalement.

La taille par vue est de 512, 768 ou 1024 pixels. Commencer à 512 ou 768 sur
une carte de 11–12 Go. Dix vues sont dix générations **séquentielles**, pas dix
images simultanément en VRAM. Le seed aléatoire est résolu et sauvegardé ;
chaque vue utilise le seed de base + son index. Avec Krea / Z-Image, la force
img2img règle le compromis entre changement d'angle et fidélité.

## Sortie équirectangulaire

En mode **360 scene**, cocher **Also create an equirectangular 360×180 panorama**.
Choisir 2048×1024, 4096×2048 ou 8192×4096. Le fichier
`equirectangular-360.png` couvre 360° horizontalement et 180° verticalement,
avec le yaw 0 au centre, le zénith en haut et le nadir en bas. Il peut servir
de texture d'environnement ou être ouvert dans un lecteur panoramique 360°.
C'est une image RGB PNG, pas une carte HDR ni une source stéréoscopique.

La conversion projette les rayons sphériques dans chacune des dix vues,
échantillonne en bilinéaire et fond les zones communes en privilégiant le
centre des vues. Elle ne se contente pas d'étirer une image au ratio 2:1.
Les pôles sont alimentés par les vues zénith/nadir. Une image finale plus
grande ne crée pas de détails supplémentaires dans les vues sources.

**Cette projection est correcte pour les poses et le champ de vue demandés.**
Le modèle peut cependant ignorer ces consignes : le résultat peut conserver
des raccords, des doublons ou des formes incompatibles entre vues. Le fondu
ne résout pas ces contradictions et ne constitue pas un recalage
photogrammétrique. Il faut examiner le résultat avant un usage en production.

Pour convertir à nouveau sans régénérer : extraire les dix PNG du ZIP,
ouvrir **Convert an existing 360 scene batch**, importer les dix fichiers
numérotés et cliquer **Convert views to 360 PNG**. L'ordre de sélection est
indifférent ; les noms d'export déterminent leurs orientations. Les vues
d'orbite d'objet sont refusées car elles n'ont pas le même centre de caméra.

## Validation

La reprojection est testée contre un environnement analytique dont la couleur
encode la direction du rayon : axes, pôles, recouvrement et fermeture à 180°.
Les chemins des cinq moteurs, les exports et l'annulation entre vues sont
testés sans poids GPU. Le parcours de l'interface est vérifié dans Chromium
avec un moteur simulé. La fidélité des dix angles avec de vrais modèles n'est
pas mesurée dans cet environnement, qui n'a pas de GPU adapté.

Ces vues générées ne remplacent pas des photos réelles pour COLMAP + Brush :
des détails incompatibles peuvent empêcher le calibrage ou dégrader les splats.

Références techniques : [édition d'image sd.cpp](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/edit.md)
et [projections sphériques / perspectives](https://paulbourke.net/panorama/cubemaps/).
