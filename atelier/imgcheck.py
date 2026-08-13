"""Diagnostic de l'affichage des images (icône d'image cassée).

Pourquoi ce module existe : le symptôme « j'importe une image, elle n'apparaît
pas » est un des rares qu'on ne peut PAS reproduire depuis le code. Six noms de
fichiers hostiles (accents, `#`, `%`, cyrillique, 120 caractères) passent tous
en HTTP 200 sur la machine de développement. Le problème est donc dans
l'environnement — et demander à quelqu'un d'ouvrir la console du navigateur
pour lire un code HTTP n'est pas une réponse acceptable dans une application
qui se veut utilisable sans être développeur.

D'où ce diagnostic : il refait le trajet exact d'une image importée (écriture
dans le cache de Gradio, puis affichage par le serveur web) et affiche le
résultat. Si l'image de test s'affiche, la chaîne de service fonctionne et le
problème est en amont ; si elle est cassée, le rapport dit lequel des maillons
a lâché, avec le chemin exact.

Les causes qu'il sait nommer sont celles qui frappent réellement sous Windows :
un cache posé sur un lecteur RÉSEAU ou un lecteur `subst` (les deux se
comportent mal avec les chemins résolus), un disque plein, un dossier non
inscriptible, un antivirus qui verrouille le fichier le temps de l'analyser
(visible dans le délai de relecture), et un chemin trop long pour l'API Win32.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import settings

# Longueur maximale d'un chemin dans l'API Win32 « ANSI » historique. Au-delà,
# des programmes échouent à ouvrir le fichier alors qu'il existe.
WIN_MAX_PATH = 260

# Au-delà, la relecture d'un fichier qu'on vient d'écrire n'est plus normale :
# quelque chose s'interpose (analyse antivirus, lecteur réseau lent).
SLOW_READBACK_S = 1.0


@dataclass
class Check:
    """Un point de contrôle. `ok` à None = information, pas un verdict."""
    ok: bool | None
    label: str
    detail: str = ""

    @property
    def mark(self) -> str:
        return {True: "✅", False: "❌", None: "•"}[self.ok]


def temp_dir() -> Path:
    """Le cache que Gradio utilise VRAIMENT (la variable, pas notre intention).

    `app.py` la pose avec `setdefault` : un environnement qui la définissait
    déjà garde la main, et le cache peut alors être ailleurs que sous le
    projet — donc hors des dossiers que le serveur a le droit de servir.
    """
    raw = os.environ.get("GRADIO_TEMP_DIR") or ""
    if raw:
        return Path(raw)
    import tempfile
    return Path(tempfile.gettempdir()) / "gradio"


def drive_kind(path: Path) -> str:
    """Type de lecteur Windows : « fixe », « réseau », « amovible »…

    Un cache sur un lecteur réseau ou sur un lecteur `subst` est la cause la
    plus courante d'images cassées de façon INTERMITTENTE : le chemin résolu
    n'est plus celui qu'on croit, et la latence fait échouer des lectures qui
    passeraient en local.
    """
    if sys.platform != "win32":
        return ""
    try:
        import ctypes
        drive = os.path.splitdrive(str(path.absolute()))[0]
        if not drive:
            return ""
        kinds = {0: "inconnu", 1: "inexistant", 2: "amovible", 3: "fixe",
                 4: "réseau", 5: "lecteur optique", 6: "disque en mémoire"}
        code = ctypes.windll.kernel32.GetDriveTypeW(f"{drive}\\")
        return kinds.get(int(code), "inconnu")
    except Exception:  # noqa: BLE001
        return ""


def _free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return -1.0


def cache_size(path: Path) -> tuple[int, int]:
    """(nombre de fichiers, octets) du cache — sans suivre les liens."""
    n = size = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                n += 1
                size += p.stat().st_size
    except OSError:
        pass
    return n, size


def write_test_image(dest_dir: Path) -> tuple[Path | None, float, str]:
    """Écrit un PNG puis le relit. Retourne (chemin, secondes, erreur).

    La relecture n'est pas une précaution de style : c'est elle qui révèle un
    antivirus qui verrouille le fichier le temps de l'analyser, ou un lecteur
    réseau qui n'a pas encore vu l'écriture.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - Pillow est une dépendance
        return None, 0.0, f"Pillow indisponible : {exc}"
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"diagnostic_{int(time.time())}.png"
        img = Image.new("RGB", (320, 200), (24, 132, 168))
        d = ImageDraw.Draw(img)
        d.rectangle((8, 8, 311, 191), outline=(255, 255, 255), width=3)
        d.text((24, 92), "TEST", fill=(255, 255, 255))
        start = time.time()
        img.save(dest, format="PNG")
        with open(dest, "rb") as fh:          # relecture réelle
            data = fh.read()
        elapsed = time.time() - start
        if not data.startswith(b"\x89PNG"):
            return None, elapsed, "le fichier relu n'est pas un PNG valide"
        return dest, elapsed, ""
    except OSError as exc:
        return None, 0.0, str(exc)


# Formats qu'un composant image accepte couramment. La tuile de test est
# écrite dans CHACUN : si le PNG s'affiche et pas le JPEG, le problème n'est
# pas la chaîne de service mais le type de fichier — une piste qu'aucun
# contrôle général ne peut donner.
TEST_FORMATS = [("PNG", ".png"), ("JPEG", ".jpg"), ("WEBP", ".webp"),
                ("GIF", ".gif"), ("BMP", ".bmp")]


def mime_of(suffix: str) -> str:
    """Type MIME que Python associe à une extension.

    Sous Windows, `mimetypes` s'initialise depuis la BASE DE REGISTRE. Une
    entrée absente ou détournée (un logiciel qui s'est approprié `.webp`, par
    exemple) fait renvoyer autre chose qu'un `image/…`, et Gradio sert alors
    le fichier en `application/octet-stream` avec une en-tête de
    téléchargement — le navigateur ne l'affiche plus. C'est invisible partout
    ailleurs, et ça ne touche QUE certaines extensions : exactement le profil
    d'un bug qui frappe les imports et épargne l'image de test.
    """
    import mimetypes
    return mimetypes.guess_type(f"x{suffix}")[0] or ""


def format_probe(dest_dir: Path) -> tuple[list[tuple[str, str]], list[str]]:
    """(libellés+chemins des tuiles écrites, lignes de rapport MIME)."""
    tiles: list[tuple[str, str]] = []
    lines: list[str] = []
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return tiles, ["Pillow indisponible."]
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, suffix in TEST_FORMATS:
        mime = mime_of(suffix)
        ok_mime = mime.startswith("image/")
        path = dest_dir / f"test_{name.lower()}{suffix}"
        try:
            img = Image.new("RGB", (160, 100), (24, 132, 168))
            img.save(path, format=name)
            tiles.append((str(path), f"{name} — {mime or 'MIME inconnu'}"))
            wrote = True
        except (OSError, KeyError, ValueError) as exc:
            wrote = False
            lines.append(f"❌ **{name}** — impossible à écrire : {exc}")
        if wrote:
            mark = "✅" if ok_mime else "❌"
            detail = (mime if ok_mime else
                      f"`{mime or 'aucun'}` — Windows ne reconnaît pas "
                      f"`{suffix}` comme une image (base de registre). Gradio "
                      "le sert alors en téléchargement et le navigateur "
                      "n'affiche rien.")
            lines.append(f"{mark} **{name}** ({suffix}) — {detail}")
    return tiles, lines


def recent_uploads(cache: Path, limit: int = 5) -> list[Path]:
    """Les derniers fichiers réellement déposés par le navigateur.

    C'est le contrôle qui tranche : si l'image que vous venez d'importer est
    là, le dépôt a fonctionné et seul l'affichage est en cause ; si elle n'y
    est pas, c'est l'envoi qui échoue, et regarder du côté du serveur d'images
    ne mènera nulle part.
    """
    out: list[Path] = []
    try:
        for p in cache.rglob("*"):
            # Nos propres tuiles de test ne comptent pas comme des imports.
            if p.is_file() and "diagnostic" not in p.parts:
                out.append(p)
    except OSError:
        return []
    out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return out[:limit]


# Au-delà, un navigateur peut renoncer à décoder l'image et n'afficher qu'une
# icône cassée, alors que le fichier est parfaitement valide. Repère indicatif :
# les limites réelles dépendent du navigateur et de la mémoire disponible.
HUGE_PIXELS = 80_000_000
HUGE_BYTES = 50 * 1024 * 1024


def describe_file(path: Path) -> str:
    """Ce que le fichier EST vraiment : format, taille, dimensions.

    L'extension ne prouve rien — un `.png` qui est en fait un JPEG, un fichier
    tronqué par un transfert, une image de 200 mégapixels : trois cas où le
    navigateur renonce et n'affiche qu'une icône cassée, sans que rien côté
    serveur n'ait échoué.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        return f"❌ illisible : {exc}"
    parts = [f"{size / 1024:.0f} Ko"]
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            w, h = im.size
            fmt = im.format or "?"
        parts.append(f"{w}×{h}, {fmt}")
        if fmt and f".{fmt.lower()}" != path.suffix.lower() and not (
                fmt == "JPEG" and path.suffix.lower() in (".jpg", ".jpeg")):
            parts.append(f"⚠️ l'extension `{path.suffix}` ne correspond pas au "
                         f"contenu ({fmt})")
        if w * h > HUGE_PIXELS:
            parts.append(f"⚠️ {w * h / 1e6:.0f} mégapixels — certains "
                         "navigateurs renoncent à décoder au-delà")
        if size > HUGE_BYTES:
            parts.append(f"⚠️ {size / 1e6:.0f} Mo — très lourd à transférer "
                         "puis à décoder")
    except Exception as exc:  # noqa: BLE001 - Pillow lève un peu de tout
        parts.append(f"❌ illisible même par l'application : {exc}. Le fichier "
                     "est probablement corrompu ou tronqué")
    return " · ".join(parts)


def checks() -> tuple[list[Check], Path | None]:
    """Tous les points de contrôle + l'image de test à afficher."""
    out: list[Check] = []
    cache = temp_dir()

    try:
        import gradio
        out.append(Check(None, "Version de Gradio", gradio.__version__))
    except ImportError:  # pragma: no cover
        pass

    out.append(Check(None, "Cache d'images", f"`{cache}`"))

    # 1. Le cache est-il DANS le projet ?
    #
    # Précision qui a son importance : hors du projet, ce n'est PAS un refus
    # d'autorisation — Gradio sert toujours son propre dossier de dépôt, quel
    # qu'il soit. Le danger est ailleurs : dans %TEMP%, le système se croit
    # autorisé à faire le ménage (Storage Sense, nettoyage de disque,
    # antivirus, redémarrage). La copie disparaît sous les pieds du
    # navigateur, la requête renvoie 404 et l'image casse — par intermittence,
    # ce qui est exactement la signature du symptôme.
    inside = _is_within(cache, settings.ROOT)
    out.append(Check(
        inside, "Cache dans le dossier du projet",
        "" if inside else
        "le cache est hors du projet (probablement dans le dossier temporaire "
        "du système). Windows y fait le ménage quand bon lui semble : les "
        "images déjà affichées cassent alors d'un coup. Une variable "
        "`GRADIO_TEMP_DIR` définie dans votre environnement prend le pas sur "
        "celle de l'application — supprimez-la puis relancez."))

    # 2. Type de lecteur (Windows). Réseau/amovible = cause connue.
    kind = drive_kind(cache)
    if kind:
        ok = kind == "fixe"
        out.append(Check(
            ok, f"Type de lecteur : {kind}",
            "" if ok else
            "un cache sur un lecteur réseau ou amovible donne des images "
            "cassées par intermittence. Déplacez le projet sur un disque "
            "interne, ou pointez `GRADIO_TEMP_DIR` vers un dossier local."))

    # 3. Longueur du chemin (Windows).
    if sys.platform == "win32":
        # Le fichier final ajoute le dossier de hachage (64) et le nom.
        projected = len(str(cache)) + 64 + 40
        ok = projected < WIN_MAX_PATH
        out.append(Check(
            ok, f"Longueur du chemin : ~{projected} caractères",
            "" if ok else
            f"au-delà de {WIN_MAX_PATH} caractères, Windows refuse d'ouvrir "
            "des fichiers qui existent pourtant. Placez le projet plus près "
            "de la racine du disque (ex. `C:\\TurboSlop`)."))

    # 4. Espace libre.
    free = _free_gb(cache if cache.exists() else settings.ROOT)
    if free >= 0:
        ok = free > 1.0
        out.append(Check(ok, f"Espace libre : {free:.1f} Go",
                         "" if ok else "un disque plein empêche d'écrire la "
                                       "copie que le navigateur va demander."))

    # 5. L'aller-retour écriture/relecture.
    dest, elapsed, err = write_test_image(cache / "diagnostic")
    if err:
        out.append(Check(False, "Écriture dans le cache", err))
    else:
        slow = elapsed > SLOW_READBACK_S
        out.append(Check(
            not slow, f"Écriture puis relecture : {elapsed * 1000:.0f} ms",
            "" if not slow else
            "c'est anormalement long pour 60 Ko. Un antivirus analyse "
            "probablement chaque fichier écrit : ajoutez le dossier du projet "
            "à ses exclusions."))

    # 6. Taille du cache — informatif, mais un cache énorme se nettoie.
    n, size = cache_size(cache)
    out.append(Check(None, "Contenu du cache",
                     f"{n} fichier(s), {size / 1e6:.0f} Mo"))
    return out, dest


def _is_within(path: Path, parent: Path) -> bool:
    try:
        Path(os.path.abspath(path)).resolve().relative_to(
            Path(os.path.abspath(parent)).resolve())
        return True
    except (ValueError, OSError):
        return False


@dataclass
class Report:
    markdown: str
    test_image: str | None
    tiles: list[tuple[str, str]]     # (chemin, libellé) pour la galerie
    last_upload: str | None


def report() -> Report:
    """Tout ce que l'interface affiche : rapport, image de test, tuiles de
    format, et le dernier fichier réellement importé."""
    cache = temp_dir()
    items, dest = checks()
    bad = [c for c in items if c.ok is False]

    lines = []
    for c in items:
        detail = f" — {c.detail}" if c.detail else ""
        lines.append(f"{c.mark} **{c.label}**{detail}")
    if dest is None:
        lines.append("⚠️ L'image de test n'a pas pu être écrite : c'est déjà "
                     "l'explication.")

    tiles, mime_lines = format_probe(cache / "diagnostic")
    bad_mime = [l for l in mime_lines if l.startswith("❌")]
    lines.append("\n**Types de fichiers** — chaque tuile ci-dessous est écrite "
                 "dans un format différent. Celles qui ne s'affichent pas "
                 "désignent le coupable.")
    lines += mime_lines

    ups = recent_uploads(cache)
    lines.append("\n**Derniers fichiers importés par le navigateur**")
    if ups:
        for p in ups:
            age = max(0, int(time.time() - p.stat().st_mtime))
            lines.append(f"• `{p.name}` — il y a {age // 60} min {age % 60} s "
                         f"· {mime_of(p.suffix) or 'MIME inconnu'} · "
                         f"{describe_file(p)}")
        lines.append("Le plus récent est affiché en bas. **S'il s'affiche ici "
                     "mais pas dans l'outil, le fichier est intact et le "
                     "problème est ailleurs ; s'il est cassé ici aussi, c'est "
                     "ce fichier-là que le navigateur n'arrive pas à lire.**")
    else:
        lines.append("• *Aucun.* Importez une image dans un outil, puis "
                     "relancez ce diagnostic : si rien n'apparaît ici, c'est "
                     "l'ENVOI qui échoue, pas l'affichage.")

    if bad or bad_mime:
        head = (f"### ❌ {len(bad) + len(bad_mime)} problème(s) trouvé(s)\n"
                "Les lignes ❌ ci-dessous expliquent quoi faire.")
    else:
        head = ("### ✅ Rien d'anormal détecté\n"
                "La chaîne de service fonctionne. Regardez alors les tuiles de "
                "format et le dernier fichier importé, plus bas : c'est là que "
                "se voit un problème propre à UN fichier.")
    return Report("\n\n".join([head] + lines),
                  str(dest) if dest else None,
                  tiles,
                  str(ups[0]) if ups else None)
