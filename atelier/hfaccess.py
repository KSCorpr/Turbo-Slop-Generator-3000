"""Répondre à « qu'est-ce qu'il me manque exactement ? » pour un dépôt fermé.

Un dépôt Hugging Face fermé échoue de QUATRE façons qui se ressemblent à
l'écran et n'appellent pas du tout la même action :

* **pas de jeton** (401) — personne n'est connecté ;
* **jeton valide, portée insuffisante** (403) — c'est le piège des jetons
  « fine-grained » : ils s'authentifient parfaitement, `whoami` répond, et ils
  n'ont pas la case « accès en lecture aux dépôts publics restreints ». Rien
  sur la page du modèle n'y changera quoi que ce soit ;
* **jeton valide, licence non acceptée** (403 aussi) — là il faut aller
  cliquer sur la page du modèle, et aucune manipulation locale n'aidera ;
* **tout va bien**.

Les deux du milieu rendent le MÊME code, et c'est pour ça qu'ils sont écrits
ensemble ici : envoyer quelqu'un accepter une licence déjà acceptée est le
genre de conseil qui fait perdre une heure. Le module dit les deux, dans
l'ordre où on peut les vérifier.

Aucune dépendance à torch : c'est une requête HTTP, et elle doit pouvoir être
lancée depuis les réglages sans charger un moteur.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from . import settings

TIMEOUT = 15.0


@dataclass(frozen=True)
class Access:
    repo: str
    ok: bool
    #  "open" | "granted" | "needs_licence" | "needs_token" | "missing" | "error"
    state: str
    detail: str = ""

    @property
    def icon(self) -> str:
        return {"open": "✅", "granted": "✅", "forbidden": "🚫",
                "needs_licence": "📝", "needs_token": "🔑",
                "missing": "❓"}.get(self.state, "⚠️")


def token() -> str:
    """Le jeton effectivement utilisable, réglages d'abord."""
    import os
    prefs = settings.load_prefs()
    return ((prefs.get("hf_token") or "").strip()
            or (os.environ.get("HF_TOKEN") or "").strip()
            or (os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip())


def _endpoint() -> str:
    prefs = settings.load_prefs()
    return (prefs.get("hf_endpoint") or "https://huggingface.co").rstrip("/")


def _get(url: str, tok: str) -> tuple[int, bytes]:
    req = urllib.request.Request(url)
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""


def whoami(tok: str | None = None) -> tuple[bool, str]:
    """Le jeton est-il valide, et à quel compte appartient-il ?"""
    tok = token() if tok is None else tok
    if not tok:
        return False, "No token saved."
    code, body = _get(f"{_endpoint()}/api/whoami-v2", tok)
    if code != 200:
        return False, (f"The token was refused (HTTP {code}). Check that it "
                       "was copied whole and has not been revoked.")
    try:
        name = json.loads(body).get("name") or "?"
    except (ValueError, AttributeError):
        name = "?"
    return True, f"Signed in as **{name}**."


def check(repo: str, probe: str = "model_index.json") -> Access:
    """L'état d'accès à CE dépôt, avec la marche à suivre s'il en manque une.

    On demande un FICHIER, et pas l'API du modèle. La distinction n'est pas
    théorique : sur un dépôt fermé, `/api/models/<id>` répond 200 sans le
    moindre jeton — seules les métadonnées publiques sortent — tandis que
    `/raw/main/<fichier>` répond 401. Interroger l'API donnait donc « accès
    accordé » à quelqu'un qui n'a pas de compte du tout, ce qui est le pire
    diagnostic possible : il envoie chercher la panne ailleurs.

    `probe` est le fichier que le moteur ira réellement lire. C'est aussi ce
    qui rend la réponse exacte plutôt que probable.
    """
    tok = token()
    url = f"{_endpoint()}/{repo}/raw/main/{probe.lstrip('/')}"
    code, _ = _get(url, tok)
    if code == 200:
        return Access(repo, True, "granted" if tok else "open",
                      "Access granted." if tok
                      else "Open — no token needed for this one.")
    if code == 401 or (code == 403 and not tok):
        return Access(repo, False, "needs_token",
                      "Gated, and no usable token is saved.")
    if code == 403:
        # 401 et 403 ne veulent PAS dire la même chose, et c'est la
        # distinction la plus utile de ce module. 401 : le Hub ne sait pas qui
        # vous êtes. 403 : il le sait très bien et refuse quand même — soit la
        # licence n'est pas acceptée, soit le jeton est « fine-grained » sans
        # la permission de lire les dépôts restreints. Les deux se corrigent
        # ailleurs et différemment.
        return Access(repo, False, "forbidden",
                      "Your token is valid — the Hub knows who you are — but "
                      "it refuses this repository. Two possible causes, and "
                      "the answer is the same code for both: the licence is "
                      "not accepted on the model's page, or the token is a "
                      "fine-grained one without “Read access to contents of "
                      "all public gated repos you can access”.")
    if code == 404:
        return Access(repo, False, "missing",
                      f"The Hub has no “{probe}” there (repository renamed?).")
    return Access(repo, False, "error", f"Unexpected answer (HTTP {code}).")


def gated_repos() -> list[tuple[str, str, str, str]]:
    """(modèle, dépôt, fichier sondé, ce qu'on y cherche).

    Construit depuis le catalogue PyTorch : une entrée qui cesse d'être fermée
    en amont disparaît d'ici toute seule, sans qu'on pense à l'enlever.
    """
    try:
        from .torchengine import catalog
    except ImportError:
        return []
    out = []
    for model in catalog.load():
        if model.config_gated and model.config_repo:
            out.append((model.id, model.config_repo,
                        "transformer/config.json",
                        "its architecture config — a few kilobytes. The "
                        "weights come from the GGUF you already have."))
        elif not model.usable and model.gated and model.repo:
            #  On ne dit PLUS « the whole model ». Ce qui manque à Krea 2 est
            #  nommé pièce par pièce dans le catalogue, et son transformer —
            #  l'essentiel du poids — se charge depuis le GGUF déjà installé.
            out.append((model.id, model.repo, "model_index.json",
                        "its pipeline settings, which only this gated "
                        "repository publishes. Its transformer already loads "
                        "from the GGUF you have."))
        elif not model.from_gguf and model.gated and model.repo:
            out.append((model.id, model.repo, "model_index.json",
                        "the whole model: no single-file loader exists for "
                        "it."))
    return out


def report() -> str:
    """Un état complet, en Markdown, pour le bouton des réglages."""
    lines = []
    ok, who = whoami()
    lines.append(f"{'✅' if ok else '🔑'} {who}")
    repos = gated_repos()
    if not repos:
        lines.append("\nNothing on this branch needs a token.")
        return "\n".join(lines)
    lines.append("")
    #  Un seul appel réseau par dépôt : le récapitulatif final se déduit de ce
    #  qu'on vient de lire, il ne le redemande pas.
    checked = [(m, why, check(repo, probe))
               for m, repo, probe, why in repos]
    for model_id, why, access in checked:
        lines.append(f"{access.icon} **{model_id}** — {access.detail}")
        lines.append(f"    · what it needs there: {why}")
        if access.state in ("needs_licence", "needs_token"):
            lines.append(f"    · page: {_endpoint()}/{access.repo}")
    blocked = [a for _m, _w, a in checked if not a.ok]
    if blocked:
        lines.append("")
        if not token():
            lines.append("Paste a token above, then accept the licence on "
                         "each page listed.")
        elif any(a.state == "forbidden" for a in blocked):
            lines.append(
                "Your token authenticates fine, so the fault is one of two "
                "things — check them in this order, the second is the one "
                "people miss:\n"
                "1. the licence, accepted on each page listed above;\n"
                "2. the **token's scope**. A fine-grained token needs "
                "*Read access to contents of all public gated repos you can "
                "access* ticked. A plain **Read** token has it by default, "
                "which is why it is the simpler choice here.")
        else:
            lines.append("The token is fine — what is missing is the licence, "
                         "and only a click on those pages can give it.")
    return "\n".join(lines)
