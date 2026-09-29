"""Réglages d'affichage communs aux composants image.

Gradio 6 a remplacé les `show_*_button` par une LISTE de boutons, et en a
profité pour changer le défaut : une image de sortie affiche désormais
`["download", "share", "fullscreen"]`. Le bouton « share » partage vers les
Discussions Hugging Face Spaces — sans objet dans une application qui tourne
sur votre machine, sans compte et sans réseau. Sous Gradio 5 il n'apparaissait
pas en local (`show_share_button=None` = « seulement sur Spaces »), donc le
laisser faire serait un bouton de plus, arrivé tout seul, qui ne fait rien
d'utile.

D'où ces listes nommées : elles disent en un endroit ce qu'on veut voir, et
elles sont passées explicitement. Un composant sans `buttons=` reprendrait le
défaut de Gradio, « share » compris — c'est le piège que ce module existe pour
fermer.
"""
from __future__ import annotations

# Sorties : on télécharge et on agrandit. Rien d'autre.
IMAGE_BUTTONS: list[str] = ["download", "fullscreen"]

# Entrées et aperçus intermédiaires : agrandir suffit, il n'y a rien à garder.
IMAGE_VIEW_ONLY: list[str] = ["fullscreen"]

# Galeries : idem, plus le téléchargement groupé quand plusieurs images sortent.
GALLERY_BUTTONS: list[str] = ["download", "fullscreen"]

# Champs texte : le bouton « copier » quand le contenu est fait pour être repris.
TEXT_COPY: list[str] = ["copy"]

# All inference callbacks share one slot. Gradio's default only serializes
# calls to the SAME function, so two model tabs otherwise compete for VRAM.
GPU_QUEUE = {"concurrency_id": "studio-gpu", "concurrency_limit": 1}


# --------------------------------------------------------------------------- #
#  Bouton « Stop »
# --------------------------------------------------------------------------- #
# Tous les boutons d'arrêt appelaient `cancel()`, qui RENVOIE un message
# (« ⏹️ Génération annulée. »), avec `outputs=None`. Le message était donc
# calculé puis jeté : on appuyait sur Stop et rien ne le confirmait à l'écran.
# Gradio 6 le dit maintenant tout haut — « A function returned too many output
# values (needed: 0, returned: 1) » — mais le défaut est plus ancien que
# l'avertissement, et la bonne réponse n'est pas de le taire : c'est d'afficher
# le message.

def stop_into_status(button, cancel_fn, status, cancels) -> None:
    """Arrêt dont la confirmation va dans une zone d'état (elle est remplacée)."""
    # Let the consumer finish after terminating its engine. Cancelling the
    # Gradio generator itself releases the GPU slot while its worker thread
    # can still be running, and skips preview/temporary-file cleanup.
    button.click(lambda: cancel_fn(), outputs=[status],
                 queue=False, show_progress="hidden")


def stop_into_log(button, cancel_fn, log, cancels) -> None:
    """Arrêt dont la confirmation s'AJOUTE au journal.

    Écrire dans un journal, c'est le remplacer : on relit donc son contenu en
    entrée pour poser la ligne à la suite, au lieu d'effacer la trace de ce
    qu'on vient d'interrompre — c'est précisément ce qu'on veut consulter après
    avoir appuyé sur Stop.
    """
    def _append(current):
        msg = cancel_fn()
        return f"{current}\n{msg}" if current else msg

    button.click(_append, inputs=[log], outputs=[log],
                 queue=False, show_progress="hidden")


class ImageHandoff:
    """Wire a source to a destination built later, using one UI transaction.

    Gradio 6 can lose a lazily mounted Image's loading state when a queued
    State.change callback updates it while switching nested tabs. A direct,
    unqueued callback also keeps navigation responsive during inference.
    """

    def __init__(self):
        self._receiver = None
        self._senders = []

    def send(self, trigger, prepare, inputs, outputs):
        sender = (trigger, prepare, inputs, outputs)
        if self._receiver is None:
            self._senders.append(sender)
        else:
            self._bind(sender)

    def receive(self, consume, outputs):
        self._receiver = (consume, outputs)
        for sender in self._senders:
            self._bind(sender)
        self._senders.clear()

    def _bind(self, sender):
        trigger, prepare, inputs, source_outputs = sender
        consume, target_outputs = self._receiver

        def transfer_image(*args):
            payload, *source_values = prepare(*args)
            return source_values + list(consume(payload))

        trigger(transfer_image, inputs=inputs,
                outputs=[*source_outputs, *target_outputs],
                queue=False, show_progress="hidden")
