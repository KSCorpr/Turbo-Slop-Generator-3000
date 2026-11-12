# TurboSlop Rust 0.2 pour Windows 11 x64

TurboSlop exécute la génération dans `TurboSlop.exe`. Le moteur Rust utilise Burn et Vulkan pour l’encodage du prompt, la diffusion et le décodage. Il faut un pilote graphique Vulkan à jour. L’exécutable fonctionne sans Python ni stable-diffusion.cpp.

## Démarrer

1. Décompresse l’archive dans un dossier accessible en écriture et lance `TurboSlop.exe`.
2. Dans **Paramètres**, utilise **Tester le GPU** pour vérifier le périphérique Vulkan. Si deux cartes sont présentes, le test affiche le nom de la carte sélectionnée.
3. Dans **Modèles**, télécharge FLUX.2 Klein 4B. Les trois fichiers publics occupent 10,42 Go. Le téléchargement peut reprendre et l’application vérifie les sommes SHA-256.
4. Dans **Générer**, saisis ton prompt, choisis la taille et clique sur **Générer**. L’image PNG et ses paramètres sont dans `outputs`.

L’application utilise le dossier de l’EXE par défaut. Tu peux choisir un autre dossier de travail dans **Paramètres** pour partager le dossier `models` ou la galerie d’une installation existante. Les réglages de l’édition native ont leur propre dossier.

## Modèle disponible

FLUX.2 Klein 4B distillé, texte vers image, quatre étapes Flow Match Euler. Les dimensions vont de 256 à 768 pixels par côté, par multiples de 16. L’encodeur Qwen3 4B est au format GGUF Q4_K_M. Le transformeur BF16 est quantifié en Q8 au chargement. La VRAM requise varie selon la taille et le pilote.

L’édition native prend en charge la génération texte vers image avec ce modèle. L’édition d’images, les autres familles de modèles, les LoRA et les modules vidéo, 3D et upscale relèvent des autres éditions de TurboSlop.

## Compiler

Depuis le dossier du dépôt, sur Windows avec Rust stable et les outils de compilation MSVC :

```powershell
$env:RUSTFLAGS = "-C target-feature=+crt-static"
cargo build --locked --release --manifest-path rust/Cargo.toml --target x86_64-pc-windows-msvc
```

L’exécutable est dans `rust/target/x86_64-pc-windows-msvc/release/turboslop.exe`. Les poids sont disponibles sur [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) sous licence Apache 2.0, ainsi que [l’encodeur Qwen3 GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF).
