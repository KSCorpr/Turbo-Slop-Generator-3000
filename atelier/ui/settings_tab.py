"""Onglet Réglages.

CE QUE CET ONGLET REFUSE DE FAIRE : poser des questions auxquelles personne
ne peut répondre. « Flash attention ? » « Convolution directe ? » « Budget
VRAM du graphe ? » — une case qu'on ne sait pas cocher n'est pas un réglage,
c'est une source de doute. Il y en avait une vingtaine, réparties sur quatre
onglets ; on ne savait ni par où commencer ni ce qu'on risquait.

Trois principes, dans cet ordre :

1. **La machine décide ce que la machine sait.** Quantification, offload,
   tiling, flash-attention : tout se déduit du matériel détecté. On l'affiche
   en clair — non pour le faire régler, mais pour dire ce qui a été décidé, et
   en CONSÉQUENCES observables plutôt qu'en noms d'options.
2. **Il reste UNE question, et c'est un goût** : plus de marge mémoire, ou plus
   de détail ? Le matériel ne peut pas y répondre à votre place. Un curseur à
   trois crans, et c'est tout.
3. **Quand on ne sait pas, on mesure.** Le placement multi-GPU ne se devine pas
   (il dépend du lien PCIe autant que de la mémoire), donc un bouton le mesure
   au lieu de demander de parier.

Le reste — les options brutes de sd.cpp — vit sous un seul repli « Expert »,
annoncé comme facultatif.

Il n'y a PAS de bouton « Enregistrer » : chaque changement s'applique aussitôt
et le dit. Un bouton d'enregistrement est une occasion de plus de se demander
si ça a été pris en compte — et il n'existait déjà pas pour la langue et le
thème, ce qui rendait le reste ambigu.
"""
from __future__ import annotations

import json
import queue
import threading

import gradio as gr

from .. import benchmark, diagnostics, hardware, settings
from ..engine import engine_build_source, resident_engine
from ..i18n import t
from . import widgets

QUANTS = ["Q3_K_S", "Q3_K_M", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M",
          "Q6_K", "Q8_0"]
LANGS = [("Français", "fr"), ("English", "en")]

# Préfixe des confirmations. Elles sont au passé et concrètes — « appliqué »,
# pas « enregistré » : ce qu'on veut savoir, c'est que c'est FAIT.
_OK = "✅ "


def _resident_reason() -> str:
    """Pourquoi le moteur résident n'est pas proposable — ou "" s'il l'est.

    Première version : la case disparaissait purement et simplement. Résultat,
    on cherche dans les Réglages une case dont on vient de lire la description,
    sans jamais savoir ce qui manque. Une option absente doit dire ce qui
    l'empêche, et le geste qui la débloque.
    """
    server = resident_engine()
    if server is None:
        return t("⚠️ **Moteur résident indisponible** : le fichier "
                 "`atelier/engine/sdserver.py` manque. Votre copie de "
                 "l'application est incomplète — retéléchargez l'archive, "
                 "**fermez l'application**, puis ré-extrayez-la.")
    if server.available():
        return ""
    if engine_build_source() == "custom-ci":
        return t("⚠️ **Moteur résident indisponible** : `sd-server` n'est pas "
                 "dans `bin/`. Votre moteur vient du build maison du projet, "
                 "qui n'empaquetait que `sd.exe`. Relancez le workflow "
                 "« Build sd.cpp (Windows CUDA) » (il empaquette désormais les "
                 "deux) puis `update-engine-ci.bat` — ou passez au binaire "
                 "officiel avec `update-engine.bat`.")
    return t("⚠️ **Moteur résident indisponible** : `sd-server` n'est pas dans "
             "`bin/`. Lancez `update-engine.bat` pour réinstaller le moteur "
             "complet.")


def _said(msg: str):
    """Affiche une confirmation — et fait EXISTER la ligne à ce moment-là.

    Un Markdown vide n'est pas invisible dans Gradio : son conteneur garde son
    cadre. Une bande vide en permanence au-dessus des réglages, c'est du bruit
    qui ressemble à un message qu'on n'arrive pas à lire. La ligne apparaît donc
    au premier changement, et pas avant.
    """
    return gr.update(value=msg, visible=True)

_STRATEGY_SAID = {
    "single": "Tout se fera sur une seule carte.",
    "encoder": "La 2e carte lira votre texte ; l'image reste sur la première.",
    "autofit": "Répartition automatique activée — mesurez-la avant d'y croire.",
}


def _gpu_choices() -> list[tuple[str, int]]:
    return [(f"#{g.index} — {g.label()}", g.index)
            for g in hardware.detect_gpus()]


def _save(**changes) -> dict:
    """Applique des changements aux préférences et renvoie le tout."""
    p = settings.load_prefs()
    # Nettoyage des anciens réglages moteur (serveur/ComfyUI, retirés).
    for stale in ("engine", "use_sd_server", "sd_server_port", "comfyui_port"):
        p.pop(stale, None)
    p.update(changes)
    settings.save_prefs(p)
    return p


def _strategy_of(prefs: dict) -> str:
    if prefs.get("auto_fit"):
        return "autofit"
    eg = prefs.get("encoder_gpu_index")
    if eg is not None and eg != prefs.get("gpu_index"):
        return "encoder"
    return "single"


def _apply_strategy_prefs(choice: str, gpu_idx) -> dict:
    """Traduit le choix en préférences (les trois stratégies s'excluent)."""
    gpus = hardware.detect_gpus()
    sel = gpu_idx if gpu_idx is not None else (
        max(gpus, key=lambda x: x.vram_gb).index if gpus else None)
    other = next((g.index for g in gpus if g.index != sel), None)
    split = (choice == "encoder") and sel is not None and other is not None
    return _save(
        auto_fit=(choice == "autofit"),
        encoder_gpu_index=other if split else None,
        # Résidence explicite : le calcul ET les poids de l'encodeur vont sur
        # la 2e carte. Sans cela, un encodeur « sur la 2e carte » relirait ses
        # poids depuis la RAM à chaque image, à travers le PCIe.
        params_backend=(f"diffusion=cuda{sel},vae=cuda{sel},te=cuda{other}"
                        if split else ""))


# --------------------------------------------------------------------------- #
#  Ce que la machine a décidé — en français, pas en noms d'options
# --------------------------------------------------------------------------- #
def _headline(prefs: dict | None = None) -> str:
    """Carte lisible : le matériel, puis ce qui en découle.

    On ne recopie surtout pas la liste des flags : `vae_tiling`, `clip_on_cpu`
    ne veulent rien dire pour qui ne connaît pas sd.cpp. Chacun est traduit en
    conséquence observable.
    """
    prefs = prefs if prefs is not None else settings.load_prefs()
    bias = hardware.bias_from_prefs(prefs)
    prof = hardware.biased_profile(bias, prefs.get("gpu_index"))
    gpus = hardware.detect_gpus()

    if prof.gpu is None:
        return t("### ⚠️ Aucune carte NVIDIA détectée\n"
                 "L'application tournera sur le processeur : c'est **très "
                 "lent** (des minutes par image). Vérifiez vos pilotes, ou "
                 "tapez `nvidia-smi` dans un terminal.")

    lines = [t("### Votre matériel"),
             t("**{name}** — {vram} Go de mémoire vidéo · {ram} Go de RAM"
               ).format(name=prof.gpu.name, vram=f"{prof.gpu.vram_gb:.0f}",
                        ram=f"{prof.ram_gb:.0f}")]
    if len(gpus) > 1:
        others = ", ".join(g.name for g in gpus if g.index != prof.gpu.index)
        lines.append(t("Deuxième carte disponible : {other}.").format(
            other=others))

    lines += ["", t("**Ce que l'application en fait, sans rien vous demander :**")]
    lines.append(t("- le modèle d'image est chargé en `{quant}` — le meilleur "
                   "compromis qui tienne dans {vram} Go ;").format(
                       quant=prof.quant, vram=f"{prof.gpu.vram_gb:.0f}"))
    lines.append(t("- l'analyse de votre texte se fait en `{enc}`, **rangée en "
                   "RAM** : elle ne prend pas de place sur la carte ;").format(
                       enc=prof.enc_quant))
    lines.append(
        t("- l'accélération « flash attention » est active (votre carte la "
          "gère) ;") if prof.diffusion_fa else
        t("- « flash attention » reste désactivée : votre carte n'a pas les "
          "unités qu'il faut, l'activer ralentirait ;"))
    lines.append(
        t("- l'image finale est assemblée par morceaux, pour ne pas saturer la "
          "carte au dernier moment.") if prof.vae_tiling else
        t("- l'image finale est assemblée d'un seul tenant : vous avez la "
          "place, autant éviter les jointures."))
    return "\n".join(lines)


def _bias_note(bias: str, prefs: dict | None = None) -> str:
    """Ce que le cran choisi change — une raison, puis un chiffre."""
    prefs = prefs if prefs is not None else settings.load_prefs()
    spec = hardware.BIASES.get(bias) or hardware.BIASES["balanced"]
    idx = prefs.get("gpu_index")
    prof = hardware.biased_profile(bias, idx)
    ref = hardware.biased_profile("balanced", idx)
    detail = t(spec["why"])
    if prof.quant != ref.quant:
        detail += t("  \n→ modèle chargé en `{quant}` au lieu de `{ref}`."
                    ).format(quant=prof.quant, ref=ref.quant)
    else:
        detail += t("  \n→ modèle chargé en `{quant}`.").format(quant=prof.quant)
    return detail


def build_settings_tab():
    with gr.Tab("⚙️ Réglages"):
        prefs = settings.load_prefs()
        gpus = hardware.detect_gpus()
        multi_gpu = len(gpus) > 1

        headline = gr.Markdown(_headline(prefs))
        status = gr.Markdown("", elem_classes="feedback", visible=False)

        # ------------------------------------------------------------------ #
        #  LE réglage : marge mémoire ou détail
        # ------------------------------------------------------------------ #
        gr.Markdown(t(
            "---\n"
            "### La seule question qu'on vous pose\n"
            "Tout le reste se calcule à partir de votre carte. Ceci ne se "
            "calcule pas, parce que c'est une préférence : voulez-vous de la "
            "**marge** (ça passe toujours) ou du **détail** (c'est plus fin, "
            "mais plus juste en mémoire) ?"))
        _bias = hardware.bias_from_prefs(prefs)
        bias = gr.Radio(
            [(t(spec["label"]), key) for key, spec in hardware.BIASES.items()],
            value=_bias, label="Priorité", show_label=False)
        bias_note = gr.Markdown(_bias_note(_bias, prefs))
        # Le vrai mode d'emploi : partir du SYMPTÔME. C'est ainsi qu'on arrive
        # sur cette page — pas en se demandant « quelle quantification ? ».
        # Deux des trois réponses renvoient ailleurs, et c'est volontaire :
        # laisser croire que tout se règle ici serait la même impasse.
        gr.Markdown(t(
            "> **Un souci précis ?**  \n"
            "> *« Erreur de mémoire / la génération s'arrête »* → prenez "
            "**🪶 Plus de marge mémoire** ci-dessus.  \n"
            "> *« C'est trop lent »* → cela ne se joue pas ici mais dans "
            "l'onglet de génération : baissez le **nombre de pas** et la "
            "**taille de l'image**, qui pèsent bien plus lourd.  \n"
            "> *« Mes images sont fades »* → là non plus : c'est le **prompt** "
            "et les **styles**, pas un réglage matériel."))

        # ------------------------------------------------------------------ #
        #  Deux cartes : on ne fait pas parier, on propose de mesurer
        # ------------------------------------------------------------------ #
        gpu = gr.Dropdown(
            label="Carte utilisée pour générer", choices=_gpu_choices(),
            value=prefs.get("gpu_index"), visible=multi_gpu,
            info=t("La plus puissante est prise par défaut."))
        if multi_gpu:
            gr.Markdown(t(
                "---\n"
                "### Vous avez deux cartes\n"
                "Il n'y a pas de bonne réponse universelle : cela dépend autant "
                "du **port PCIe** de la seconde carte que de sa mémoire. Plutôt "
                "que de vous faire deviner, l'application peut **mesurer**."))
            strategy = gr.Radio(
                [(t("Tout sur une seule carte — le plus fiable"), "single"),
                 (t("La 2e carte s'occupe du texte — libère de la mémoire "
                    "pour l'image"), "encoder"),
                 (t("Répartir automatiquement — à mesurer avant d'y croire"),
                  "autofit")],
                value=_strategy_of(prefs), label="Répartition",
                show_label=False)
            tools_gpu = gr.Dropdown(
                label="Carte pour l'améliorateur de prompt",
                choices=[(t("La même que pour l'image"), None)] + _gpu_choices(),
                value=prefs.get("text_gpu_index"))
        else:
            strategy = gr.State(_strategy_of(prefs))
            tools_gpu = gr.State(prefs.get("text_gpu_index"))

        # ------------------------------------------------------------------ #
        #  Mesurer plutôt que deviner
        # ------------------------------------------------------------------ #
        gr.Markdown(t(
            "---\n"
            "### 🧪 Dans le doute, mesurez\n"
            "L'application génère la **même image** {n} fois par configuration "
            "(plus une première jetée, le temps que tout soit chargé) et garde "
            "la médiane. Elle ne change **aucun réglage** : elle vous dit "
            "lequel est le plus rapide, vous décidez ensuite."
        ).format(n=benchmark.MEASURED_RUNS))
        with gr.Row():
            bench_btn = gr.Button(t("⏱️ Mesurer sur ma machine"),
                                  variant="primary")
            apply_bench = gr.Button(t("Appliquer le plus rapide"))
            bench_stop = gr.Button(t("⏹️ Arrêter"), variant="stop")
        bench_status = gr.Markdown("")
        with gr.Accordion(t("Détail de la mesure (journal et rapport)"),
                          open=False):
            bench_file = gr.File(label="Rapport JSON", interactive=False)
            bench_log = gr.Textbox(label="Journal du test", lines=10,
                                   autoscroll=True, elem_classes="log-box")
            system_md = gr.Markdown(diagnostics.summary_markdown())
            report_btn = gr.Button(t("📋 Exporter le rapport système"),
                                   size="sm")

        # ------------------------------------------------------------------ #
        #  Expert : les options brutes de sd.cpp, sous UN seul repli
        # ------------------------------------------------------------------ #
        with gr.Accordion(
                t("🔧 Expert — options brutes de sd.cpp (facultatif)"),
                open=False):
            gr.Markdown(t(
                "⚠️ **Rien ici n'est nécessaire.** Ces options existent parce "
                "que sd.cpp les expose, pas parce qu'il faut y toucher. Elles "
                "se règlent en mesurant, pas en devinant — et le curseur "
                "ci-dessus couvre déjà les cas courants. Toucher à cette "
                "section **désactive le réglage automatique**."))

            gr.Markdown(t("**Quantification imposée** — « auto » = laisser "
                          "l'application décider d'après la carte."))
            with gr.Row():
                quant = gr.Dropdown(label="Modèle d'image",
                                    choices=["auto"] + QUANTS,
                                    value=prefs.get("quant") or "auto")
                enc_quant = gr.Dropdown(label="Analyse du texte",
                                        choices=["auto"] + QUANTS,
                                        value=prefs.get("enc_quant") or "auto")
            f = prefs.get("flags", {})
            gr.Markdown(t("**Options mémoire du moteur.**"))
            with gr.Row():
                fa = gr.Checkbox(value=f.get("diffusion_fa", True),
                                 label="Flash attention")
                offload = gr.Checkbox(value=f.get("offload_to_cpu", True),
                                      label="Modèle rangé en RAM")
                tiling = gr.Checkbox(value=f.get("vae_tiling", True),
                                     label="Image assemblée par morceaux")
            with gr.Row():
                clip_cpu = gr.Checkbox(value=f.get("clip_on_cpu", False),
                                       label="Texte sur le processeur")
                vae_cpu = gr.Checkbox(value=f.get("vae_on_cpu", False),
                                      label="Assemblage sur le processeur")

            gr.Markdown(t(
                "---\n"
                "**Cache entre les pas** — réutilise des calculs d'un pas de "
                "diffusion au suivant. Ne gagne quelque chose qu'au-delà de "
                "~10 pas ; nos modèles en font 4 à 8, donc **laissez "
                "désactivé** sauf mesure contraire."))
            with gr.Row():
                cache_mode = gr.Dropdown(
                    [(t("Désactivé (recommandé)"), ""),
                     ("easycache", "easycache"), ("dbcache", "dbcache"),
                     ("taylorseer", "taylorseer"), ("cache-dit", "cache-dit"),
                     ("spectrum", "spectrum")],
                    value=prefs.get("cache_mode", ""), label="Mode de cache")
                cache_opt = gr.Textbox(
                    value=prefs.get("cache_option", ""),
                    label="Option (vide = défauts)",
                    placeholder="ex. threshold=0.2")

            gr.Markdown(t(
                "---\n"
                "**Convolution directe** — supprime un gros tampon "
                "intermédiaire. Gain de mémoire certain ; effet sur la vitesse "
                "**imprévisible** (parfois mieux, parfois moins bien). À "
                "chronométrer, pas à cocher les yeux fermés."))
            with gr.Row():
                conv_diff = gr.Checkbox(
                    value=bool(prefs.get("conv_direct_diffusion")),
                    label="Convolution directe — modèle d'image")
                conv_vae = gr.Checkbox(
                    value=bool(prefs.get("conv_direct_vae")),
                    label="Convolution directe — assemblage")

            gr.Markdown(t(
                "---\n"
                "**Découpage du calcul** — autorise le moteur à découper son "
                "graphe pour tenir dans un budget au lieu d'échouer. **C'est "
                "plus lent** : à réserver aux résolutions qui ne passent pas "
                "autrement. L'onglet 🚀 HD s'en sert déjà tout seul."))
            with gr.Row():
                max_vram = gr.Dropdown(
                    [(t("Désactivé (recommandé)"), ""),
                     (t("Auto — mémoire libre moins 1 Go"), "auto"),
                     (t("Plafond ferme : 6 Go"), "6"),
                     (t("Plafond ferme : 8 Go"), "8"),
                     (t("Plafond ferme : 10 Go"), "10")],
                    value=prefs.get("max_vram", ""), allow_custom_value=True,
                    label="Budget mémoire du calcul")
                stream_layers = gr.Checkbox(
                    value=bool(prefs.get("stream_layers")),
                    label="Streaming des couches depuis la RAM",
                    info=t("Exige que le modèle soit rangé en RAM. Sans cela, "
                           "le moteur ignore l'option."))

            gr.Markdown(t(
                "---\n"
                "**Moteur résident** — aujourd'hui le moteur démarre, lit le "
                "modèle, fabrique l'image et s'arrête : le chargement est "
                "repayé à **chaque** image. Coché, le modèle reste chargé "
                "entre deux générations. C'est tout bénéfice quand on génère "
                "une image à la fois pour affiner un prompt.\n\n"
                "En échange : **pas d'aperçu pendant le calcul** (l'image "
                "arrive d'un coup), et le modèle occupe la carte en "
                "permanence — les outils du Toolkit le déchargent tout seuls "
                "quand ils ont besoin du GPU. Les LoRA et la passe HD "
                "repassent automatiquement par l'ancien mode."))
            _no_resident = _resident_reason()
            gr.Markdown(_no_resident, visible=bool(_no_resident))
            resident = gr.Checkbox(
                value=bool(prefs.get("resident_engine")),
                label="Garder le modèle chargé entre deux images",
                visible=not _no_resident)

            # Confirmation LOCALE : la ligne d'état du haut est hors de l'écran
            # quand on coche quelque chose ici. Un réglage qui s'applique sans
            # rien dire de visible, c'est un réglage dont on doute.
            expert_status = gr.Markdown("", elem_classes="feedback", visible=False)

            combo = hardware.rtx3060_1080ti_combo() if multi_gpu else None
            combo_btn = gr.Button(t("⚡ Profil RTX 3060 + GTX 1080 Ti"),
                                  visible=bool(combo))

        # ------------------------------------------------------------------ #
        #  Ce qui n'a rien à voir avec la génération
        # ------------------------------------------------------------------ #
        with gr.Accordion(t("🌍 Langue, thème et comptes"), open=False):
            with gr.Row():
                lang_dd = gr.Dropdown(
                    LANGS, value=prefs.get("lang", "fr"),
                    label="🌐 Langue / Language (redémarrage requis)")
                theme_dd = gr.Dropdown(
                    [(t("Clair"), "light"), (t("Sombre"), "dark")],
                    value=prefs.get("theme", "light"),
                    label="🎨 Thème (redémarrage requis)")
            hf_ep = gr.Textbox(
                value=prefs.get("hf_endpoint", "https://huggingface.co"),
                label="Endpoint Hugging Face (miroir éventuel)")
            civitai_tok = gr.Textbox(
                value=prefs.get("civitai_token", ""),
                label="Jeton Civitai (optionnel — LoRA protégés)",
                type="password")
            account_status = gr.Markdown("", elem_classes="feedback", visible=False)

        # ================================================================== #
        #  Câblage — tout s'applique à la volée
        # ================================================================== #
        def _apply_bias(choice, gpu_idx):
            prof = hardware.biased_profile(choice, gpu_idx)
            # « Équilibré » RESTE l'automatique : c'est exactement ce qu'il
            # calcule. Les deux autres crans sont un choix explicite, donc ils
            # figent quant et flags — sinon le profil automatique les
            # réécrirait au prochain démarrage et le réglage aurait l'air de
            # « ne pas tenir ».
            p = _save(auto_optimize=(choice == "balanced"),
                      quant=None if choice == "balanced" else prof.quant,
                      enc_quant=None if choice == "balanced" else prof.enc_quant,
                      flags=prof.flags())
            return (_headline(p), _bias_note(choice, p),
                    _said(_OK + t("Priorité appliquée : **{label}**.").format(
                        label=t(hardware.BIASES[choice]["label"]))))

        bias.change(_apply_bias, inputs=[bias, gpu],
                    outputs=[headline, bias_note, status])

        def _apply_gpu(idx, choice):
            _save(gpu_index=idx)
            p = _apply_strategy_prefs(choice, idx)
            return _headline(p), _said(_OK + t(
                "Carte de génération : #{idx}.").format(idx=idx))

        gpu.change(_apply_gpu, inputs=[gpu, strategy],
                   outputs=[headline, status])

        if multi_gpu:
            def _apply_strategy(choice, idx):
                p = _apply_strategy_prefs(choice, idx)
                return _headline(p), _said(_OK + t(_STRATEGY_SAID[choice]))

            strategy.change(_apply_strategy, inputs=[strategy, gpu],
                            outputs=[headline, status])

            def _apply_tools_gpu(v):
                _save(text_gpu_index=v)
                return _said(_OK + t("Carte de l'améliorateur enregistrée."))

            tools_gpu.change(_apply_tools_gpu, inputs=[tools_gpu],
                             outputs=[status])

        # ---- Expert : chaque contrôle s'applique seul --------------------- #
        def _apply_expert(q, eq, fa_, off, til, clip, vae, cm, co,
                          cd, cv, mv, sl):
            p = _save(auto_optimize=False,
                      quant=None if q == "auto" else q,
                      enc_quant=None if eq == "auto" else eq,
                      flags={"diffusion_fa": bool(fa_),
                             "offload_to_cpu": bool(off),
                             "vae_tiling": bool(til),
                             "clip_on_cpu": bool(clip),
                             "vae_on_cpu": bool(vae)},
                      cache_mode=cm or "", cache_option=(co or "").strip(),
                      conv_direct_diffusion=bool(cd),
                      conv_direct_vae=bool(cv),
                      max_vram=(mv or "").strip(),
                      stream_layers=bool(sl))
            return (_headline(p),
                    _bias_note(hardware.bias_from_prefs(p), p),
                    _said(_OK + t("Réglage expert appliqué "
                                  "(automatique désactivé).")))

        _expert = [quant, enc_quant, fa, offload, tiling, clip_cpu, vae_cpu,
                   cache_mode, cache_opt, conv_diff, conv_vae, max_vram,
                   stream_layers]
        for comp in _expert:
            comp.change(_apply_expert, inputs=_expert,
                        outputs=[headline, bias_note, expert_status])

        # Le moteur résident n'est PAS un réglage de sd.cpp : il ne doit donc
        # pas basculer l'application en mode manuel comme le fait `_apply_expert`.
        def _apply_resident(on):
            _save(resident_engine=bool(on))
            if on:
                return _said(_OK + t("Le modèle restera chargé entre deux "
                                     "images. Le premier chargement sera "
                                     "aussi long que d'habitude."))
            server = resident_engine()
            if server is not None:
                server.stop()
            return _said(_OK + t("Moteur résident désactivé, la mémoire de la "
                                 "carte est rendue."))

        resident.change(_apply_resident, inputs=[resident],
                        outputs=[expert_status])

        # ---- Langue, thème, comptes --------------------------------------- #
        def _apply_lang(lang):
            _save(lang=lang)
            disp = {v: k for k, v in LANGS}.get(lang, lang)
            return _said(_OK + t("Langue enregistrée. **Redémarrez l'application** "
                                 "(`run.bat` / `run.sh`) pour appliquer « {lang} »."
                                 ).format(lang=disp))

        def _apply_theme(th):
            _save(theme="dark" if th == "dark" else "light")
            return _said(_OK + t("Thème enregistré. **Redémarrez l'application** "
                                 "pour l'appliquer."))

        def _apply_endpoint(v):
            _save(hf_endpoint=v or "https://huggingface.co")
            return _said(_OK + t("Endpoint enregistré."))

        def _apply_token(v):
            _save(civitai_token=(v or "").strip())
            return _said(_OK + t("Jeton Civitai enregistré."))

        lang_dd.change(_apply_lang, inputs=[lang_dd], outputs=[account_status])
        theme_dd.change(_apply_theme, inputs=[theme_dd], outputs=[account_status])
        hf_ep.change(_apply_endpoint, inputs=[hf_ep], outputs=[account_status])
        civitai_tok.change(_apply_token, inputs=[civitai_tok],
                           outputs=[account_status])

        # ---- Mesure -------------------------------------------------------- #
        _bench_stop = threading.Event()

        def _verdict(data: dict) -> str:
            """Dire ce qui a été mesuré, pas seulement qui gagne."""
            rows = [r for r in data.get("results", []) if r.get("ok")]
            if not rows:
                return t("❌ Aucune configuration n'a pu être mesurée "
                         "(voir le journal).")
            lines = [f"- **{r['label']}** — {r['seconds']:.2f} s "
                     f"(± {r.get('spread_seconds', 0):.2f} s)" for r in rows]
            best = data.get("recommended_mode") or data.get("fastest_model")
            head = t("✅ Médiane sur {n} mesures · le plus rapide : **{best}**"
                     ).format(n=data.get("measured_runs", "?"), best=best)
            return head + "\n" + "\n".join(lines)

        def _do_bench():
            """Diffuse le journal au fil de l'eau.

            Rendre le résultat d'un bloc à la fin laisserait l'interface muette
            plusieurs minutes, sans dire si ça avance ni comment l'arrêter.
            """
            _bench_stop.clear()
            q: "queue.Queue[str | None]" = queue.Queue()
            state: dict = {}

            def worker():
                try:
                    state["path"] = benchmark.run_hardware_benchmark(
                        log=q.put, cancel=_bench_stop.is_set)
                except Exception as exc:  # noqa: BLE001
                    state["err"] = exc
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            logs: list[str] = []
            yield t("⏳ Mesure en cours…"), gr.update(), ""
            while True:
                line = q.get()
                if line is None:
                    break
                logs.append(line)
                yield gr.update(), gr.update(), "\n".join(logs[-500:])
            tail = "\n".join(logs[-500:])
            if isinstance(state.get("err"), benchmark.Cancelled):
                yield (t("⏹️ Mesure interrompue — aucun réglage modifié."),
                       gr.update(), tail)
                return
            if "err" in state:
                yield f"❌ {state['err']}", gr.update(), tail
                return
            path = state["path"]
            yield (_verdict(json.loads(path.read_text(encoding="utf-8"))),
                   str(path), tail)

        _bench_evt = bench_btn.click(
            _do_bench, outputs=[bench_status, bench_file, bench_log])

        def _cancel_bench() -> str:
            # On POSE le drapeau au lieu de tuer le fil : la génération en
            # cours va au bout et la mesure s'arrête proprement ensuite.
            # Couper au milieu laisserait un tir à moitié chronométré.
            _bench_stop.set()
            return t("⏹️ Arrêt demandé — la mesure en cours se termine.")

        widgets.stop_into_log(bench_stop, _cancel_bench, bench_log,
                              [_bench_evt])

        def _apply_report(raw):
            if not raw:
                return gr.update(), _said(t("❌ Lancez d'abord la mesure."))
            try:
                mode = benchmark.apply_recommendation(getattr(raw, "name", raw))
            except Exception as exc:  # noqa: BLE001
                return gr.update(), _said(f"❌ {exc}")
            return _headline(), _said(_OK + t(
                "Configuration mesurée appliquée : **{mode}**.").format(
                    mode=mode))

        apply_bench.click(_apply_report, inputs=[bench_file],
                          outputs=[headline, status])

        def _system_report():
            return (diagnostics.summary_markdown(),
                    str(diagnostics.write_system_report()))

        report_btn.click(_system_report, outputs=[system_md, bench_file])

        def _apply_combo():
            preset = hardware.rtx3060_1080ti_prefs()
            p = _save(**preset)
            return (_headline(p), gr.update(value=preset["gpu_index"]),
                    gr.update(value="encoder"),
                    _said(_OK + t("Profil deux cartes appliqué : la RTX 3060 "
                                  "dessine, la 1080 Ti lit votre texte.")))

        combo_btn.click(_apply_combo,
                        outputs=[headline, gpu, strategy, status])
