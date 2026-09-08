"""Lire le GGUF de Krea 2 Turbo avec diffusers — le chargeur qui manquait.

diffusers 0.40 sait charger un fichier unique pour Z-Image et Flux.2 Klein,
mais pas pour Krea 2 : `Krea2Transformer2DModel` n'est pas enregistré dans
`SINGLE_FILE_LOADABLE_CLASSES`. J'en avais conclu qu'il fallait retélécharger
le modèle entier depuis un dépôt fermé. C'était une conclusion trop rapide :
l'absence d'un convertisseur en amont ne dit pas que le fichier est illisible,
seulement que personne n'a écrit la correspondance.

Elle s'écrit, et elle est EXACTE. Vérifiée tenseur par tenseur contre l'en-tête
du vrai fichier (`krea2_turbo-Q5_K_M.gguf`) et contre le `state_dict` de la
classe diffusers : 430 tenseurs d'un côté, 430 clés de l'autre, une seule
correspondance possible pour chacune, et toutes les formes concordent. Le test
rejoue cette vérification sur un relevé conservé dans `tests/fixtures`, donc
sans réseau.

Deux tenseurs sur 432 ne correspondent à rien, et ce n'est pas un trou dans la
correspondance : `last.up.weight` et `last.down.weight`, deux matrices
6144×6144. Ni la couche finale de diffusers (`Krea2FinalLayer` : une table de
modulation, une normalisation, une projection) ni celle de stable-diffusion.cpp
(`KreaLastLayer`, mêmes trois blocs) n'en contient. Les métadonnées du fichier
disent ce qu'elles sont — `egg_w: 6144`, `egg_h: 6144`, `egg_c: 1`,
`egg_format: chw_m1p1_flat` : une image de 6144×6144 en un canal, aplatie et
cachée dans le fichier par celui qui l'a empaqueté. Une cinquantaine de
mégaoctets d'œuf de Pâques, dans chaque quantification, qu'aucun des deux
moteurs ne lit. On les écarte nommément plutôt qu'en silence : une clé
inattendue qui disparaît sans un mot est exactement la façon dont un modèle
finit par rendre des images subtilement fausses.

**Ce fichier ne suffit pas à faire tourner Krea 2 sur ce moteur.** Il règle le
transformer, c'est-à-dire l'essentiel du poids ; l'encodeur de texte et le VAE
butent ailleurs, et `catalog.py` dit où.
"""
from __future__ import annotations

#  Les deux tenseurs à écarter, et pourquoi (voir l'en-tête du module).
EASTER_EGG = ("last.up.weight", "last.down.weight")

#  Les pièces uniques, nommées une à une : il y en a dix-neuf, et une table est
#  plus lisible qu'une suite de règles à laquelle il faudrait faire confiance.
TOP_LEVEL: dict[str, str] = {
    "first.weight": "img_in.weight",
    "first.bias": "img_in.bias",
    "tmlp.0.weight": "time_embed.linear_1.weight",
    "tmlp.0.bias": "time_embed.linear_1.bias",
    "tmlp.2.weight": "time_embed.linear_2.weight",
    "tmlp.2.bias": "time_embed.linear_2.bias",
    "tproj.1.weight": "time_mod_proj.weight",
    "tproj.1.bias": "time_mod_proj.bias",
    "txtmlp.0.scale": "txt_in.norm.weight",
    "txtmlp.1.weight": "txt_in.linear_1.weight",
    "txtmlp.1.bias": "txt_in.linear_1.bias",
    "txtmlp.3.weight": "txt_in.linear_2.weight",
    "txtmlp.3.bias": "txt_in.linear_2.bias",
    "last.norm.scale": "final_layer.norm.weight",
    "last.linear.weight": "final_layer.linear.weight",
    "last.linear.bias": "final_layer.linear.bias",
    "last.modulation.lin": "final_layer.scale_shift_table",
    "txtfusion.projector.weight": "text_fusion.projector.weight",
}

#  Renommages à l'intérieur d'un bloc. L'ORDRE compte : `.attn.wo.weight` doit
#  passer avant toute règle plus courte qui le contiendrait.
INNER: tuple[tuple[str, str], ...] = (
    (".attn.qknorm.qnorm.scale", ".attn.norm_q.weight"),
    (".attn.qknorm.knorm.scale", ".attn.norm_k.weight"),
    (".attn.wq.weight", ".attn.to_q.weight"),
    (".attn.wk.weight", ".attn.to_k.weight"),
    (".attn.wv.weight", ".attn.to_v.weight"),
    (".attn.wo.weight", ".attn.to_out.0.weight"),
    (".attn.gate.weight", ".attn.to_gate.weight"),
    (".mlp.", ".ff."),
    (".prenorm.scale", ".norm1.weight"),
    (".postnorm.scale", ".norm2.weight"),
    (".mod.lin", ".scale_shift_table"),
)

PREFIXES: tuple[tuple[str, str], ...] = (
    ("blocks.", "transformer_blocks."),
    ("txtfusion.", "text_fusion."),
)

#  La table de modulation par bloc est stockée À PLAT dans le GGUF (36864) et
#  attendue en table par diffusers (6 × 6144). C'est le seul remodelage, et il
#  est déterminé par la forme cible — on ne devine rien.
FLAT_TABLES = ("scale_shift_table",)


def convert_key(name: str) -> str | None:
    """Le nom diffusers correspondant, ou None si le tenseur est à écarter."""
    if name in EASTER_EGG:
        return None
    if name in TOP_LEVEL:
        return TOP_LEVEL[name]
    out = name
    for old, new in PREFIXES:
        if out.startswith(old):
            out = new + out[len(old):]
            break
    else:
        return None
    for old, new in INNER:
        out = out.replace(old, new)
    return out


def convert_state_dict(checkpoint: dict, **kwargs) -> dict:
    """Le convertisseur, à la forme qu'attend le chargeur fichier-unique.

    Il REND compte de ce qu'il jette. Un tenseur inconnu qui disparaîtrait en
    silence est le scénario où le modèle charge sans erreur et rend des images
    subtilement fausses — le seul défaut qu'on ne verrait jamais.
    """
    out: dict = {}
    unknown: list[str] = []
    for key in list(checkpoint.keys()):
        tensor = checkpoint.pop(key)
        target = convert_key(key)
        if target is None:
            if key not in EASTER_EGG:
                unknown.append(key)
            continue
        if target.endswith(FLAT_TABLES) and tensor.ndim == 1:
            # 36864 = 6 × 6144. On déduit la largeur de la table plutôt que de
            # l'écrire : elle suit la taille du modèle.
            rows = 6
            tensor = tensor.reshape(rows, tensor.shape[0] // rows)
        out[target] = tensor
    if unknown:
        raise ValueError(
            "This Krea 2 checkpoint holds tensors the converter does not know: "
            + ", ".join(sorted(unknown)[:5])
            + (" …" if len(unknown) > 5 else "")
            + ". Loading it would silently drop them.")
    return out


def register(diffusers_module) -> bool:
    """Branche le convertisseur dans diffusers, sans jamais écraser le sien.

    Le jour où l'amont publie son propre chargeur, celui-ci doit s'effacer :
    deux correspondances pour un même modèle finiraient par diverger, et c'est
    la nôtre qui aurait tort.
    """
    from diffusers.loaders import single_file_model as sfm
    table = sfm.SINGLE_FILE_LOADABLE_CLASSES
    name = "Krea2Transformer2DModel"
    if name in table:
        return False
    if not hasattr(diffusers_module, name):
        return False
    table[name] = {"checkpoint_mapping_fn": convert_state_dict,
                   "default_subfolder": "transformer"}
    # `from_single_file` n'est offert qu'aux classes qui portent le mixin.
    from diffusers.loaders import FromOriginalModelMixin
    cls = getattr(diffusers_module, name)
    if not issubclass(cls, FromOriginalModelMixin):
        cls.__bases__ = (*cls.__bases__, FromOriginalModelMixin)
    return True
