"""Onglet « 🧹 Gestion & aide » : inventaire de ce qui occupe le disque
(moteurs, modèles, add-ons, données), suppression sélective, et documentation
de toutes les options de l'application.
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr

from .. import inventory, settings, storage
from ..i18n import t

CATEGORIES = ["Moteurs", "Modèles", "Add-ons Toolkit", "Vos données"]


def _choices_and_summary():
    """Cases à cocher (installés seulement) + tableau récapitulatif."""
    its = inventory.items()
    choices: list[tuple[str, str]] = []
    rows: list[str] = []
    total = 0
    for cat in CATEGORIES:
        cat_items = [i for i in its if i.category == cat]
        if not cat_items:
            continue
        rows.append(f"\n**{cat}**\n")
        rows.append("| | Élément | Taille |")
        rows.append("|---|---|---|")
        for i in cat_items:
            size = i.size
            total += size
            mark = "✅" if i.installed else "—"
            warn = " ⚠️" if (i.protected and i.installed) else ""
            rows.append(f"| {mark} | {i.label}{warn} | "
                        f"{inventory.human(size)} |")
            if i.installed:
                label = f"{i.label} — {inventory.human(size)}"
                if i.protected:
                    label = "⚠️ " + label
                choices.append((label, i.key))
    rows.append(f"\n**Total occupé : {inventory.human(total)}**")
    return choices, "\n".join(rows)


def _move_choices() -> list[tuple[str, str]]:
    """Éléments (dossiers) déplaçables individuellement, avec leur état."""
    out: list[tuple[str, str]] = []
    for i in inventory.items():
        dirs = [p for p in i.paths if p.is_dir()]
        if not dirs:
            continue
        moved = [p for p in dirs if storage.is_link(p)]
        if moved:
            tgt = storage.link_target(moved[0])
            label = f"↗️ {i.label} — déplacé vers {tgt}"
        else:
            label = f"{i.label} — {inventory.human(i.size)}"
        out.append((label, i.key))
    return out


def build_manage_tab():
    with gr.Tab("🧹 Gestion & aide"):
        # ------------------------------------------------------------------ #
        #  Inventaire & suppression
        # ------------------------------------------------------------------ #
        gr.Markdown(
            "### Gestion de l'espace disque\n"
            "Tout ce que l'application a téléchargé, avec sa taille. Coche ce "
            "que tu veux supprimer puis confirme. Les éléments marqués "
            "**⚠️** sont **tes données** (LoRA, modèles perso, créations) — "
            "réfléchis à deux fois. Tout le reste est **re-téléchargeable** "
            "depuis l'app.")

        _choices, _summary = _choices_and_summary()
        summary_md = gr.Markdown(_summary)
        picks = gr.CheckboxGroup(choices=_choices, value=[],
                                 label="Éléments à supprimer")
        with gr.Row():
            confirm = gr.Checkbox(
                value=False,
                label="Je confirme vouloir supprimer les éléments cochés")
            refresh = gr.Button("↻ Rafraîchir les tailles", size="sm")
        delete_btn = gr.Button("🗑️ Supprimer la sélection", variant="stop")
        result = gr.Markdown("")

        def _refresh():
            ch, summ = _choices_and_summary()
            return (gr.update(choices=ch, value=[]), gr.update(value=summ),
                    gr.update(value=False), "")

        refresh.click(_refresh,
                      outputs=[picks, summary_md, confirm, result])

        def _delete(keys, ok):
            if not keys:
                return (gr.update(), gr.update(), gr.update(),
                        t("Rien de coché."))
            if not ok:
                return (gr.update(), gr.update(), gr.update(),
                        t("⚠️ Coche **« Je confirme »** pour supprimer."))
            msgs, freed = inventory.delete(list(keys))
            ch, summ = _choices_and_summary()
            report = "\n".join(msgs) + \
                f"\n\n**{inventory.human(freed)} libéré(s).**"
            return (gr.update(choices=ch, value=[]), gr.update(value=summ),
                    gr.update(value=False), report)

        delete_btn.click(_delete, inputs=[picks, confirm],
                         outputs=[picks, summary_md, confirm, result])

        gr.Markdown(
            f"*Emplacements : modèles `{settings.MODELS_DIR.name}/`, moteurs "
            f"`{settings.BIN_DIR.name}/`, add-ons `tools_repo/`, LoRA "
            f"`{settings.LORA_DIR.name}/`, sorties "
            f"`{settings.OUTPUT_DIR.name}/`.*")

        # ------------------------------------------------------------------ #
        #  Emplacement des modèles (dossier externe)
        # ------------------------------------------------------------------ #
        gr.Markdown("---\n### 📁 Emplacement des modèles")
        with gr.Accordion("Déplacer les modèles vers un autre disque "
                          "(ex. NVMe rapide)", open=False):
            gr.Markdown(
                "Les modèles peuvent vivre **hors du dossier du projet** : "
                "utile pour les mettre sur un **NVMe** (chargements bien plus "
                "rapides) ou sur un disque plus grand. Les lectures répétées "
                "**n'usent pas** un SSD — seules les écritures comptent.\n\n"
                "⚠️ Le changement est pris en compte **au redémarrage** de "
                "l'application.")
            loc_now = gr.Markdown(
                f"**Dossier actuel :** `{storage.current()}`"
                + ("  *(défaut du projet)*" if storage.is_default() else ""))
            dest_box = gr.Textbox(
                value=storage.configured(),
                label="Nouveau dossier (chemin absolu)",
                placeholder=r"ex. D:\IA\models  ou  /mnt/nvme/models")
            with gr.Row():
                move_btn = gr.Button("📦 Déplacer les modèles ici",
                                     variant="primary")
                point_btn = gr.Button("🔗 Pointer ici sans déplacer")
                reset_btn = gr.Button("↩️ Revenir au dossier du projet")
            loc_log = gr.Textbox(label="Journal", lines=10, autoscroll=True,
                                 elem_classes="log-box")

            def _do_move(raw):
                dest, err = storage.validate(raw)
                if err:
                    yield gr.update(), f"❌ {err}"
                    return
                lines: list[str] = []
                yield gr.update(), t("⏳ Déplacement en cours… (long si les "
                                     "disques diffèrent — ne fermez pas)")
                for msg in storage.move(dest):
                    lines.append(msg)
                    yield gr.update(), "\n".join(lines)
                yield (gr.update(value=f"**Dossier actuel :** `{storage.current()}` "
                                       "— *redémarrez pour appliquer*"),
                       "\n".join(lines))

            move_btn.click(_do_move, inputs=[dest_box],
                           outputs=[loc_now, loc_log])

            def _do_point(raw):
                dest, err = storage.validate(raw)
                if err:
                    return gr.update(), f"❌ {err}"
                msg = storage.save(dest)
                return (gr.update(value=f"**Dossier actuel :** `{storage.current()}` "
                                        "— *redémarrez pour appliquer*"),
                        msg + "\n\n*(Aucun fichier déplacé : le dossier indiqué "
                        "doit déjà contenir vos modèles, sinon ils seront "
                        "re-téléchargés.)*")

            point_btn.click(_do_point, inputs=[dest_box],
                            outputs=[loc_now, loc_log])

            def _do_reset():
                return (gr.update(value=f"**Dossier actuel :** "
                                        f"`{settings.DEFAULT_MODELS_DIR}` "
                                        "— *redémarrez pour appliquer*"),
                        storage.save(None), gr.update(value=""))

            reset_btn.click(_do_reset, outputs=[loc_now, loc_log, dest_box])

        with gr.Accordion("Déplacer SEULEMENT certains éléments "
                          "(ex. les 16 Go de la 3D)", open=False):
            gr.Markdown(
                "Déplace **les éléments cochés** vers un autre disque et laisse "
                "un **lien** à leur place : l'application continue de les "
                "trouver, **sans redémarrage ni réglage**. Idéal pour sortir "
                "les gros modèles 3D tout en gardant les modèles d'image sur "
                "le NVMe.\n\n"
                "⚠️ Le disque de destination doit rester **branché** — sinon "
                "les éléments déplacés deviennent introuvables.")
            sel_picks = gr.CheckboxGroup(choices=_move_choices(), value=[],
                                         label="Éléments à déplacer / ramener")
            sel_dest = gr.Textbox(
                label="Dossier de destination (chemin absolu)",
                placeholder=r"ex. E:\IA-gros-modeles  ou  /mnt/hdd/ia")
            with gr.Row():
                sel_move = gr.Button("📦 Déplacer + créer le lien",
                                     variant="primary")
                sel_back = gr.Button("↩️ Ramener dans le projet")
                sel_refresh = gr.Button("↻ Rafraîchir", size="sm")
            sel_log = gr.Textbox(label="Journal", lines=10, autoscroll=True,
                                 elem_classes="log-box")

            def _sel_paths(keys):
                out = []
                for k in keys or []:
                    it = inventory.by_key(k)
                    if it:
                        out += [p for p in it.paths if p.is_dir()]
                return out

            def _sel_move(keys, raw):
                paths = _sel_paths(keys)
                if not paths:
                    yield gr.update(), t("Rien de coché.")
                    return
                raw = (raw or "").strip().strip('"')
                if not raw or not Path(raw).expanduser().is_absolute():
                    yield gr.update(), t("❌ Indiquez un dossier de destination "
                                         "en chemin **absolu**.")
                    return
                lines: list[str] = []
                yield gr.update(), t("⏳ Déplacement en cours…")
                for msg in storage.relocate(paths,
                                            Path(raw).expanduser()):
                    lines.append(msg)
                    yield gr.update(), "\n".join(lines)
                yield gr.update(choices=_move_choices(), value=[]), \
                    "\n".join(lines)

            sel_move.click(_sel_move, inputs=[sel_picks, sel_dest],
                           outputs=[sel_picks, sel_log])

            def _sel_back(keys):
                paths = _sel_paths(keys)
                if not paths:
                    yield gr.update(), t("Rien de coché.")
                    return
                lines: list[str] = []
                yield gr.update(), t("⏳ Retour en cours…")
                for msg in storage.restore(paths):
                    lines.append(msg)
                    yield gr.update(), "\n".join(lines)
                yield gr.update(choices=_move_choices(), value=[]), \
                    "\n".join(lines)

            sel_back.click(_sel_back, inputs=[sel_picks],
                           outputs=[sel_picks, sel_log])
            sel_refresh.click(lambda: (gr.update(choices=_move_choices(),
                                                 value=[]), ""),
                              outputs=[sel_picks, sel_log])

        # ------------------------------------------------------------------ #
        #  Documentation des options
        # ------------------------------------------------------------------ #
        gr.Markdown("---\n### 📖 Aide — que fait chaque option ?")

        with gr.Accordion("🎨 Onglets de génération (Flux.2 / Krea 2)",
                          open=False):
            gr.Markdown(
                "**Prompt** — ta description. Sur un **modèle d'édition** "
                "(Flux.2), décris la *modification* à appliquer.  \n"
                "**Prompt négatif** — ce qu'on ne veut pas. **Affiché seulement "
                "si le modèle en tient compte** (CFG > 1) : les modèles "
                "distillés tournent en CFG 1.0 et l'ignorent.  \n"
                "**🎭 Prompt système / style** — préfixe ajouté à chaque "
                "génération ; « — Aucun — » le retire.  \n"
                "**📷 Styles photo / 🎨 Styles artistiques** — banques de styles "
                "**cumulables**. Ton sujet est inséré dans chaque style coché ; "
                "🎲 en tire un au hasard.  \n"
                "**✨ Améliorer le prompt** — un petit LLM local réécrit ton "
                "idée en prompt anglais détaillé (add-on à installer).\n\n"
                "**Format / Largeur / Hauteur** — préréglages calés sur les "
                "**résolutions natives** du modèle (il rend mieux dessus).  \n"
                "**Étapes** — nombre de pas de débruitage. Les modèles "
                "distillés sont calibrés pour peu de pas (4–8) : au-delà, on "
                "gagne peu.  \n"
                "**CFG** — force du guidage par le prompt. **1.0 = désactivé** "
                "(normal en distillé). >1 = suit davantage le prompt (et "
                "active le négatif). 0 peut **ignorer** le prompt.  \n"
                "**Préréglage** — combo sampler/scheduler/pas recommandé par "
                "la doc du modèle. En cas de doute, garde-le.  \n"
                "**Sampler / Scheduler** — algorithmes d'échantillonnage. "
                "« Auto » laisse le moteur choisir : le plus sûr.  \n"
                "**Flow shift** — 0 = auto (recommandé). Trop bas = grain en "
                "haute résolution.  \n"
                "**Seed** — graine aléatoire. -1 = aléatoire ; une valeur fixe "
                "**rejoue la même image**.  \n"
                "**Images** — nombre d'images générées (seeds consécutifs).\n\n"
                "**🖼️ Image de référence** — *édition* (Flux.2 : jusqu'à 3 "
                "images, piloté par le prompt) ou *image-to-image* (**force de "
                "transformation** : bas = fidèle à l'original, haut = "
                "réinventé).  \n"
                "**🧩 Outpaint** — agrandit la toile et laisse le modèle "
                "remplir les bords (expérimental).  \n"
                "**🧩 LoRA** — styles/concepts additionnels, avec un poids "
                "(≈ 0.6–1.0 en général). Doivent correspondre à "
                "**l'architecture du modèle**.  \n"
                "**🚀 PiD ×4** — décode la sortie en haute résolution par "
                "diffusion pixel (poids non commerciaux).  \n"
                "**📂 Fichiers locaux** — utiliser un modèle déposé à la main "
                "au lieu de celui du catalogue.")

        with gr.Accordion("⚙️ Réglages (matériel & optimisation)", open=False):
            gr.Markdown(
                "**Automatique (recommandé)** — déduit tout du matériel "
                "détecté : quantification de diffusion selon la **VRAM**, "
                "quantification de l'encodeur selon la **RAM**, et les flags "
                "mémoire. À laisser coché en temps normal.  \n"
                "**Presets par génération de carte** — applique en 1 clic un "
                "profil curaté pour GTX 10xx → RTX 50xx (désactive l'auto et "
                "remplit les champs manuels).\n\n"
                "**Quant. diffusion / encodeur** — précision des poids. "
                "`Q8_0` ≈ sans perte (gros) → `Q4_K_M` bon compromis → `Q3` "
                "agressif. Plus bas = tient dans moins de VRAM, moins fidèle.  \n"
                "**Flash attention** — attention optimisée (Turing/RTX 20xx et "
                "plus). Plus rapide, moins de VRAM. Désactivée sur Pascal.  \n"
                "**Offload CPU** — garde les poids en RAM et ne monte que le "
                "nécessaire en VRAM : économise beaucoup de VRAM, coût faible.  \n"
                "**VAE tiling** — décode l'image par tuiles : évite les pics "
                "mémoire en haute résolution (un peu plus lent).  \n"
                "**CLIP / VAE sur CPU** — derniers recours quand la VRAM est "
                "vraiment juste (lent).\n\n"
                "**🧮 Multi-GPU** (si ≥ 2 cartes) — *Une seule carte* "
                "(recommandé) ; *Encodeur sur la 2e carte* (libère de la VRAM "
                "sur la principale) ; *Auto-fit* (répartit tout — ⚠️ force la "
                "VRAM, risque d'OOM).  \n"
                "**⚡ Cache** — réutilise des calculs entre les pas. Utile "
                "au-delà de ~10 pas ; sur un modèle distillé (4–8 pas), gain "
                "faible et artefacts possibles. Laisser désactivé.")

        with gr.Accordion("🧊 Image → 3D (trellis)", open=False):
            gr.Markdown(
                "**Résolution géométrie** — **512** = « light », le seul mode "
                "qui tienne sur une carte ≤ 12 Go. **1024/1536** demandent "
                "~16 Go+ : en dessous, la géométrie sort souvent **corrompue "
                "(« blobs »)** au lieu d'échouer proprement.  \n"
                "**Seed** — -1 = aléatoire ; valeur fixe = même objet rejoué.  \n"
                "**Détourage** — *BiRefNet* (qualité) ou *Seuil* (rapide).  \n"
                "**⚡ Serveur résident** — garde le moteur 3D en vie entre les "
                "générations (évite ~30 s de rechargement) **mais occupe la "
                "VRAM** : à arrêter avant de générer des images.  \n"
                "**Décimation** — nombre de faces cible (plus bas = maillage "
                "plus léger).  \n"
                "**Atlas UV** — résolution de la texture (1024 → 4096).  \n"
                "**Géométrie seule** — sans texture, plus rapide.  \n"
                "**Carte utilisée** — ⚠️ en multi-GPU, laisser le calcul "
                "déborder sur une carte **Pascal (GTX 10xx)** peut corrompre "
                "la géométrie : **épingle la carte la plus récente**.  \n"
                "**Exiger le GPU** — empêche un repli CPU silencieux (qui "
                "prendrait des heures). À laisser coché.")

        with gr.Accordion("🔧 Convertir en GGUF · 🧰 Toolkit", open=False):
            gr.Markdown(
                "**Convertir en GGUF** — quantifie un modèle déposé dans "
                "`models/custom/` vers un GGUF plus léger (100 % CPU). sd.cpp "
                "n'accepte que `q8_0 / q5_1 / q5_0 / q4_1 / q4_0 / f16` "
                "(**pas** de k-quants ici). `q5_1` est un bon défaut.\n\n"
                "**Toolkit** — *Profondeur* (carte de profondeur), *Sans "
                "arrière-plan* (détourage auto), *SAM* (détourage au clic), "
                "*Agrandir ESRGAN* (rapide, déterministe, 100 % GPU), "
                "*Upscale créatif SDXL* (ré-invente du détail ; la "
                "**créativité** contrôle la liberté prise avec l'original). "
                "Chaque outil s'installe en 1 clic (PyTorch, à la demande).")

        with gr.Accordion("🌐 Réseau, partage & maintenance", open=False):
            gr.Markdown(
                "**Endpoint Hugging Face** — miroir alternatif si HF est "
                "bloqué/lent sur ton réseau.  \n"
                "**Jeton Civitai** — nécessaire pour importer certains LoRA "
                "protégés.  \n"
                "**Partage réseau** — `run-lan.bat` expose l'app aux machines "
                "du réseau local (pense au pare-feu).  \n"
                "**maintenance.bat** — après une mise à jour par copier-coller : "
                "supprime les fichiers obsolètes, purge les caches et vérifie "
                "que tout compile. Ne touche jamais à tes modèles/sorties.  \n"
                "**update-engine.bat** — met à jour le moteur sd.cpp "
                "(binaire officiel) ; **update-engine-ci.bat** installe notre "
                "build maison (compilé pour nos cartes).  \n"
                "**update-trellis.bat** — met à jour le moteur **3D** "
                "(trellis.cpp). Les modèles 3D (~16 Go) ne sont pas "
                "re-téléchargés.  \n"
                "**📁 Emplacement des modèles** (ci-dessus) — déplace les "
                "modèles vers un autre disque (NVMe, disque plus grand). "
                "*Déplacer* transfère les fichiers ; *Pointer sans déplacer* "
                "réutilise un dossier qui les contient déjà. Effet au "
                "**redémarrage**.")
