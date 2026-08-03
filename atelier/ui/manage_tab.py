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

        with gr.Accordion("🖌️ Boogu Edit Turbo 10B — édition par instruction",
                          open=False):
            gr.Markdown(
                "Le seul modèle du catalogue conçu pour **éditer sur "
                "instruction** — *« enlève la voiture »*, *« mets un fond de "
                "plage »*, *« remplace le texte par … »* — plutôt que pour "
                "fabriquer une image à partir de rien. C'est ce qui se rapproche "
                "le plus d'un Gemini / ChatGPT parmi les poids ouverts qui "
                "tiennent sur 11-12 Go. 10,3 milliards de paramètres, licence "
                "Apache 2.0, **distillé : 4 pas, CFG 1.0** — même vitesse que "
                "Flux.2 Klein.\n\n"
                "**Place disque** ≈ 17 Go : diffusion 7,4 Go (Q4_1) ou 8,6 Go "
                "(Q5_1) selon ta VRAM, encodeur Qwen3-VL-8B 8,7 Go déchargé en "
                "RAM, projecteur vision 0,75 Go, VAE 0,17 Go.\n\n"
                "💡 **Sur deux cartes** : c'est le cas d'usage idéal du réglage "
                "*Encodeur sur la 2e carte* — la diffusion sur la 3060, "
                "l'encodeur Qwen3-VL sur la 1080 Ti.\n\n"
                "**Une seule image de référence.** La doc de sd.cpp ne montre "
                "qu'une image pour Boogu (Flux.2 en accepte trois). Les "
                "emplacements 2 et 3 restent utilisables, mais non documentés. "
                "Le **prompt négatif est masqué** : à CFG 1.0 il serait ignoré, "
                "comme sur Flux.2 Klein et Krea 2.\n\n"
                "---\n\n"
                "### ⚠️ Poids antérieurs au correctif du 8 juillet 2026\n"
                "L'équipe Boogu a publié ce jour-là un correctif de l'Edit-Turbo "
                "réglant « une dégradation sévère de la qualité d'image et de "
                "mauvaises performances **sur la suppression d'objets** et "
                "d'autres tâches d'édition ». **Aucun GGUF de la version "
                "corrigée n'a été publié** : les deux dépôts GGUF existants sont "
                "antérieurs, et le seul dérivé du correctif est dans un format "
                "que sd.cpp ne sait pas charger. On a pris la version "
                "pré-correctif pour avoir les 4 pas — choix assumé.\n\n"
                "**Si un « enlève ceci » donne un résultat médiocre, deux "
                "solutions.** Les deux réutilisent le même encodeur, le même "
                "mmproj et le même VAE — seul le fichier de diffusion change, "
                "donc rien d'autre à retélécharger :\n\n"
                "1. **La variante non-Turbo**, jamais signalée défectueuse. "
                "Télécharge un `boogu-edit-dit-*.gguf` depuis "
                "`realrebelai/Boogu-Image-Edit_GGUFs`, dépose-le dans le dossier "
                "des modèles, et désigne-le via **📂 Fichiers locaux → "
                "diffusion** dans le même onglet. Pense à monter les curseurs : "
                "**25 à 50 pas, CFG 2 à 5** (elle n'est pas distillée).\n"
                "2. **Les poids corrigés en bf16** (20,6 Go) depuis "
                "`Comfy-Org/Boogu-Image`, fichier "
                "`boogu_image_edit_turbo_hotfix_1k_20260708_bf16.safetensors`, "
                "à quantifier ensuite dans l'onglet **🔧 Convertir en GGUF** "
                "(en Q5_1 ou Q4_1). Plus long à mettre en place, mais tu gardes "
                "les 4 pas **et** le correctif.")

        with gr.Accordion("🎨 Onglets de génération (Flux.2 / Krea 2 / Boogu)",
                          open=False):
            gr.Markdown(
                "**Prompt** — ta description. Sur un **modèle d'édition** "
                "(Flux.2), décris la *modification* à appliquer.  \n"
                "**Prompt négatif** — ce qu'on ne veut pas. **Affiché seulement "
                "si le modèle en tient compte** (CFG > 1) : les modèles "
                "distillés tournent en CFG 1.0 et l'ignorent.  \n"
                "**🎨 Styles** — un seul repli qui regroupe les trois banques, "
                "toutes **cumulables** entre elles :  \n"
                "  · **🎭 Préréglage perso** — un préfixe que tu écris et "
                "enregistres, ajouté en tête de chaque génération ; "
                "« — Aucun — » le retire. Les préréglages sont **globaux** : "
                "enregistrés une fois, ils apparaissent dans les trois onglets. "
                "Certains sont **livrés avec l'app** (comme *📷 France "
                "provinciale 1995-2005*) : ils survivent aux mises à jour et ne "
                "peuvent pas être supprimés, mais enregistrer un style du même "
                "nom crée ta propre version, qui prend le dessus — la supprimer "
                "rétablit l'originale.  \n"
                "  · **📷 Styles photo** (139) — qualité, lumière, objectif, "
                "pellicule, ambiance. Ton sujet est inséré *dans* chaque style "
                "coché. Banque © ghleg, MIT.  \n"
                "  · **🖍️ Styles artistiques** (397) — anime, cartoon, BD, "
                "dessin, design, peinture ; ajoutés *après* ton sujet. "
                "🎲 en tire un au hasard (wildcard).  \n"
                "**🎨 Générer / ⏹️ Stop / 🗑️ Effacer** — juste sous le prompt. "
                "*Effacer* vide le prompt, le négatif et toute trace "
                "d'amélioration.  \n"
                "**↩️ Rétablir le prompt d'origine** — apparaît après une "
                "amélioration et remet exactement ce que tu avais écrit, "
                "paramètres `--` compris.  \n"
                "ℹ️ **Un préréglage ne traduit rien** : c'est un préfixe collé "
                "devant ton texte. Si tu écris en français, tu obtiens du "
                "français avec une entête anglaise. C'est le bouton "
                "**✨ Améliorer** qui traduit et met en forme — et il tient "
                "désormais compte du préréglage actif : il décrit le sujet sans "
                "ajouter d'appareil, d'objectif, de lumière ni de traitement qui "
                "le contrediraient. Ordre conseillé : **choisir le préréglage, "
                "puis améliorer**.\n\n"
                "**✨ Améliorateur de prompt** — un petit LLM local réécrit ton "
                "idée en prompt anglais (add-on à installer). **Intensité** "
                "règle l'ampleur de la réécriture, **Propositions** en génère "
                "plusieurs d'un coup (en un seul chargement du modèle) : "
                "clique celle que tu préfères. La **ligne grise** sous les "
                "menus résume ce que le bouton va faire avant que tu cliques. "
                "Réglages **et** installation sont dans le même repli.\n\n"
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
                "**🧩 Outpaint centré** — agrandit la toile symétriquement et "
                "laisse le modèle remplir les bords (expérimental, modèles "
                "d'édition seulement). Pour un outpaint **directionnel**, voir "
                "l'onglet « 🖼️ Outpaint ».  \n"
                "**🧩 LoRA** — styles/concepts additionnels, avec un poids "
                "(≈ 0.6–1.0 en général). Doivent correspondre à "
                "**l'architecture du modèle**.  \n"
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

        with gr.Accordion("🖼️ Outpaint (étendre une image)", open=False):
            gr.Markdown(
                "Étend une image **à gauche, à droite, en haut, en bas ou tout "
                "autour**, façon Midjourney. Ça marche avec **n'importe quel "
                "modèle** du catalogue et **sans prompt** — aucun modèle "
                "d'inpainting n'est nécessaire.\n\n"
                "**⚠️ À utiliser avec un modèle d'ÉDITION** (Flux.2 Klein, Boogu "
                "Edit). Ce n'est pas une préférence : c'est ce qui fait que ça "
                "marche. Un modèle d'édition reçoit la toile agrandie en "
                "**référence** — son conditionnement image lui dit ce que "
                "contient la scène — plus une **consigne d'extension écrite "
                "automatiquement**. Un modèle de text-to-image ordinaire, lui, "
                "ne reçoit qu'un latent bruité en image-to-image : il ne sait "
                "pas ce qu'il prolonge, donc il **réinvente**. Aucun réglage de "
                "force, de fondu ou de remplissage ne corrige ça — c'est une "
                "limite de méthode. Le repli est là pour ne pas bloquer, pas "
                "parce qu'il donne un bon résultat.\n\n"
                "C'est aussi ça, le « sans prompt » : tu n'écris rien, mais le "
                "modèle reçoit une consigne précise qui nomme les côtés étendus "
                "et lui demande de continuer perspective, lumière, palette et "
                "style sans toucher à l'original ni dupliquer de sujet.\n\n"
                "**Comment ça marche** — la toile est agrandie, la nouvelle zone "
                "est remplie (gris neutre pour un modèle d'édition : une zone "
                "franchement vide se lit « à remplir »), le modèle génère, la "
                "**tonalité du neuf est recalée** sur celle de l'original, puis "
                "**l'image d'origine est recollée** par-dessus.\n\n"
                "⚠️ Le recalage de tonalité n'est pas un détail : en "
                "image-to-image le modèle re-rend TOUTE la toile plus "
                "contrastée. Sans lui, l'original recollé apparaît comme un "
                "rectangle plus terne au milieu — et le fondu n'y peut rien, "
                "l'écart étant global et non local.\n\n"
                "**Direction** — un ou plusieurs côtés à étendre.  \n"
                "**Extension par côté** — proportion ajoutée de chaque côté "
                "choisi (0.25 = +25 %). La nouvelle taille s'affiche en dessous ; "
                "elle est alignée sur 16 px et plafonnée à **2048 px** de côté "
                "(au-delà, les marges sont réduites automatiquement).  \n"
                "**Modèle** — n'importe quel modèle installé. Le **sampler, le "
                "CFG et les étapes** sont repris de ses réglages recommandés.  \n"
                "**Prompt** — *facultatif*. À laisser vide pour une extension "
                "neutre ; le remplir sert seulement à orienter ce qui apparaît "
                "dans la nouvelle zone.  \n"
                "**Remplissage des bords** — *Gris neutre* (défaut sur un modèle "
                "d'édition) : la zone à remplir est sans ambiguïté. *Étirement "
                "flou* prolonge les pixels du contour, ce qui ne transmet que la "
                "couleur, aucune forme. *Miroir* donne une continuité parfaite "
                "sur un motif régulier, **mais reflète un sujet proche du "
                "bord** — et le modèle transforme volontiers ce reflet en un "
                "second objet bien réel. À réserver aux fonds uniformes.  \n"
                "**Force de génération** — **repli seulement**, grisée sur un "
                "modèle d'édition (celui-ci est piloté par la consigne, pas par "
                "une force). Haut = invente librement ; bas = reste proche du "
                "pré-remplissage.  \n"
                "**Fondu de raccord** — largeur du dégradé à la jonction avec "
                "l'original. ~24 px efface la couture. Attention : il mélange "
                "une bande d'environ **2× sa valeur** *à l'intérieur* du bord de "
                "l'original — c'est précisément ce qui fait disparaître la "
                "couture. **0 = collage net**, original strictement intact "
                "partout.  \n"
                "**Recalage de tonalité** — 0 à 1 : à quel point le neuf est "
                "ramené sur le contraste et la couleur de l'original. À baisser "
                "seulement si la correction exagère sur une image inhabituelle.  \n"
                "**Seed** — -1 = aléatoire ; une valeur fixe rejoue la même "
                "extension.  \n"
                "**♻️ Ré-étendre le résultat** — recharge le résultat comme "
                "nouvelle image d'entrée, pour enchaîner les extensions "
                "(droite, puis haut, etc.).\n\n"
                "Chaque résultat est enregistré dans `outputs/` avec un fichier "
                "`.txt` à côté qui récapitule modèle, seed et réglages.")

        with gr.Accordion("🧊 Image → 3D (trellis)", open=False):
            gr.Markdown(
                "**Poids utilisés** — variante des modèles : **f16** (~16,5 Go), "
                "**q8** (~9,9 Go) ou **q4** (~6 Go). ⚠️ **f16 est le plus "
                "RAPIDE** quand il tient en mémoire : en ggml, les poids "
                "quantifiés sont déquantifiés à la volée pendant le calcul, "
                "et ce surcoût n'est PAS amorti sur une charge 3D "
                "(compute-bound) — contrairement aux LLM. q8/q4 ne servent "
                "donc qu'à faire tenir un mode (1024/1536) qui déborderait "
                "autrement. Plusieurs variantes peuvent coexister ; on "
                "bascule à la génération.  \n"
                "**Compléter en carré** — TRELLIS pré-traite l'entrée en "
                "**carré** : une image 16:9 envoyée telle quelle sort "
                "**déformée** (écrasée). Coché (défaut), des bandes neutres "
                "sont ajoutées pour garder les proportions ; le détourage "
                "les retire.  \n"
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
                "puis **trois agrandisseurs**, du plus fidèle au plus inventif : "
                "*ESRGAN* (rapide, déterministe, 100 % GPU ; choisis un modèle "
                "🎨 **dessin/anime** pour de la BD ou de l'illustration — un "
                "modèle photo pose des halos sur les traits et invente du grain "
                "dans les aplats ; tu peux déposer tes propres `.pth` "
                "d'OpenModelDB dans le dossier des upscalers), "
                "*SeedVR2* (restaure du détail plausible), *Upscale créatif "
                "SDXL* (ré-invente ; la **créativité** contrôle la liberté "
                "prise avec l'original). "
                "Chaque outil s'installe en 1 clic (PyTorch, à la demande).\n\n"
                "**🎯 SeedVR2 en détail** — **un seul pas de diffusion, sans "
                "prompt** : il n'y a quasiment rien à régler, juste le **côté "
                "court visé**. Zone de confort **×2 à ×4** ; au-delà la qualité "
                "décroche. Léger : 2,9 Go de poids, ~4,6 Go de VRAM au pic pour "
                "un 512→2048.  \n"
                "*Correction colorimétrique* recale les couleurs sur "
                "l'original — « lab » convient presque toujours.  \n"
                "**VAE par tuiles : laisse-le coché.** Ce qui pèse, ce n'est pas "
                "ton image d'entrée mais la **taille visée** : SeedVR2 "
                "redimensionne d'abord, puis le VAE encode à cette taille, et "
                "ses convolutions 3D causales répliquent l'image dans le temps — "
                "c'est ça qui fait exploser la mémoire. Sur 11–12 Go, sans "
                "tuiles ça déborde vers 1440 px. Si ça déborde quand même : "
                "baisse la taille visée, puis descends la tuile à 256. Le "
                "curseur est plafonné à 512 exprès (auto-attention sur la tuile "
                "entière, coût en O(n²) : plus gros = plus lent ET plus "
                "lourd).  \n"
                "*D'où vient le code ?* Il n'existe ni paquet pip ni pipeline "
                "diffusers pour SeedVR2. On clone le dépôt d'inférence de "
                "référence, qui est distribué comme nœud ComfyUI mais fournit "
                "un `inference_cli.py` prévu pour tourner **sans ComfyUI** — "
                "on n'installe ni ne lance ComfyUI, on appelle ce script en "
                "sous-process. Le VAE (~0,5 Go) est récupéré automatiquement au "
                "premier agrandissement.")

        with gr.Accordion("🔄 Mettre à jour l'app (copier-coller du ZIP)",
                          open=False):
            gr.Markdown(
                "**Après CHAQUE mise à jour par copier-coller : lance "
                "`maintenance.bat`.** Extraire un ZIP par-dessus *ajoute et "
                "écrase, mais ne supprime jamais* : les fichiers retirés en "
                "amont restent en orphelins, et un `__pycache__` périmé peut "
                "faire tourner de l'ancien code.\n\n"
                "**Ce que le ZIP remplace… et ce qu'il ne touche pas :**\n\n"
                "| | Remplacé ? |\n| --- | --- |\n"
                "| Code de l'app, catalogue de modèles, docs | ✅ oui |\n"
                "| Moteur `bin/` | ❌ non — `update-engine.bat` seulement quand "
                "une nouveauté de sd.cpp est nécessaire |\n"
                "| Modèles, LoRA, sorties, préférences | ❌ non — c'est voulu |\n"
                "| Add-ons du Toolkit (`tools_repo/`) | ❌ **non, et c'est le "
                "piège** |\n\n"
                "⚠️ **Le piège.** Certains add-ons ne sont pas que des poids : "
                "**SeedVR2 applique des correctifs au code qu'il télécharge** "
                "(config d'architecture, sélection du modèle, versions "
                "épinglées). Quand une mise à jour change *la façon dont un "
                "add-on s'installe*, mettre à jour l'app laisse l'add-on figé "
                "dans son ancien état — et ça ne se voit qu'à l'usage suivant. "
                "Dans ce cas : **reclique sur son bouton « Installer »**. Les "
                "poids déjà téléchargés ne sont pas repris.\n\n"
                "`maintenance.bat` vérifie ça pour toi et nomme précisément ce "
                "qui est périmé — tu n'as pas à deviner.\n\n"
                "**Résumé** : `maintenance.bat` à chaque fois ; "
                "`update-engine.bat` seulement sur demande ; réinstaller un "
                "add-on seulement si la maintenance ou un message d'erreur le "
                "réclame.")

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
