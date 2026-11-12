# TurboSlop Rust pour Windows 11 x64

## Démarrer

1. Extrais l’archive dans un dossier où tu peux écrire et lance `TurboSlop.exe`.
2. Dans **Réglages**, sélectionne le dossier principal de TurboSlop si tu as déjà des poids et des LoRA. L’application retrouve `models/`, `loras/`, `outputs/` et `bin/sd-cli.exe`.
3. Pour une nouvelle installation, choisis **NVIDIA CUDA 12**, **Vulkan** ou **CPU**, puis **Installer / mettre à jour**. Un pilote graphique compatible est requis pour CUDA et Vulkan. L’installateur télécharge le binaire officiel de sd.cpp et ses DLL CUDA si nécessaire.
4. Dans **Modèles**, sélectionne le modèle et les variantes à télécharger. Les téléchargements publics ne demandent aucun token. Une interruption laisse le fichier partiel pour la reprise.
5. Dans **Images**, saisis ton prompt, ajoute les références éventuelles, choisis le preset et la variante installée, puis clique sur **Générer**. `Ctrl+Entrée` lance aussi la génération.

Les images apparaissent dans `outputs/`. Les préférences Rust et le moteur installé depuis cette interface résident dans `userdata-rust/`. Le même dossier de travail peut être utilisé par les deux éditions.

## Périmètre

Cette première version couvre le catalogue des neuf modèles d’image, la génération par lots, l’édition par références, l’img2img, les LoRA locaux, les adaptateurs des presets DMD et Krea, l’aperçu par étape via sd.cpp et la galerie. Le calcul d’inférence reste confié au binaire sd.cpp. L’interface, le catalogue, les transferts et le pilotage du moteur sont natifs et fonctionnent sans Python ni serveur web.

Les outils de restauration vidéo, d’upscale, de segmentation, d’outpaint et de reconstruction 3D restent disponibles dans l’édition Python. Le mode LoRA de référence spécifique à Krea et les décodeurs TAESD restent également dans l’édition Python. La génération complète avec les poids réels demande un GPU et un budget mémoire adaptés au modèle.

## Compiler sur Windows

Avec Rust stable, le compilateur MSVC et le Windows SDK :

```powershell
cargo test --locked --manifest-path rust/Cargo.toml
cargo build --locked --release --manifest-path rust/Cargo.toml --target x86_64-pc-windows-msvc
```

L’exécutable se trouve dans `rust/target/x86_64-pc-windows-msvc/release/turboslop.exe`. Le workflow GitHub Actions `Rust Windows x64` publie une archive portable comme artefact de build.
