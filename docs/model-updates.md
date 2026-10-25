# Vérifier les versions des modèles installés

Après installation de cette version, **`update.bat`** enchaîne : code,
maintenance, moteur sd.cpp, moteur 3D puis contrôle des modèles installés.
Le contrôle peut être lancé seul avec **`update.bat --models`**, également
disponible sous Linux/macOS avec `./update.sh --models`.

Le script compare les empreintes des **fichiers présents**, avec une requête
de métadonnées par dépôt/version. Un nouveau README ou un commit dans un
autre fichier ne produit pas une alerte sur des poids inchangés. Un fichier
de même taille dont les octets ont changé est détecté.

Les nouveaux téléchargements enregistrent le dépôt, le nom exact, la branche
ou révision demandée, le commit servi et l'empreinte vérifiée dans
`userdata/model-downloads.json`. Une branche comme `main` reste suivie même
si le téléchargement d'un snapshot utilise un commit immuable pour ses fichiers.

Les installations antérieures sont reconnues par les dossiers historiques,
l'index des poids partagés et les métadonnées Hugging Face existantes. Si aucune
empreinte fiable n'existe, **le premier contrôle lit le fichier local une fois**
pour l'identifier. Cela peut prendre du temps pour plusieurs dizaines de Go,
mais ne télécharge pas ces poids. Les contrôles suivants réutilisent cette
empreinte tant que la taille, la date et l'identité du fichier correspondent.
Les petits fichiers Git utilisent leur hash de blob Git ; les gros fichiers
LFS utilisent SHA-256.

Le contrôle couvre les dossiers des modèles d'image et upscalers, les outils
HF du Toolkit, SeedVR2, VOSR, SAM 2, HiVG, OmniSVG, Qwen Layered, TRELLIS et
Pixal3D, ainsi que les LoRA dont la provenance HF est enregistrée. Un fichier
personnel ou provenant d'une release GitHub non identifié est signalé comme
`unmapped`, sans inventer de version amont.

| Résultat | Signification |
| --- | --- |
| `up_to_date` | Le fichier installé correspond au contenu publié. |
| `update_available` | Le fichier de la branche suivie a un contenu différent. |
| `pinned_update` | Une version épinglée est installée ; le fichier de `main` diffère. |
| `local_modified` | Le fichier a changé localement depuis sa dernière identification. |
| `missing_upstream` | Le nom installé n'est plus présent dans la révision interrogée. |
| `unavailable` | Réseau, autorisation ou mode hors ligne empêchant la vérification. |
| `unverified` | Les métadonnées ne permettent pas de conclure. |
| `unmapped` | La provenance distante de ce fichier est inconnue. |

Les résultats détaillés sont enregistrés dans
**`userdata/model-update-report.json`**, avec la date du contrôle, les chemins
locaux et les empreintes comparées. Une panne réseau ne remplace ni ne
supprime un modèle et n'annule pas la mise à jour du code/des moteurs.

**Le contrôle ne remplace aucun poids.** Pour une branche suivie, utiliser
**Models → Download selected image weights** ou **Install / repair** selon
le package. Le téléchargement vérifie les fichiers, conserve ceux qui sont
identiques et reprend les transferts avec le plafond réseau habituel.

Les révisions épinglées des toolkits ont été choisies avec leur runtime et
leurs configurations : une différence sur `main` est une information, pas
l'autorisation de substituer automatiquement un checkpoint. Leur évolution
demande une intégration compatible dans TurboSlop.

`--code`, `--engine`, `--trellis` et `--clean` restent limités à leurs étapes,
sauf si `--models` est également indiqué. `--check` et `--rollback` ne lancent
aucun contrôle de modèles.
