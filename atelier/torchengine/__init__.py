"""Moteur PyTorch — la branche Test7000.

L'application entière tourne normalement sur stable-diffusion.cpp : un binaire
natif, des poids GGUF, aucune dépendance Python lourde pour générer. Cette
branche répond à une question qu'on ne peut pas trancher en la lisant : à quoi
ressemblerait le MÊME outil, avec les MÊMES fonctions et la même interface, si
tout passait par PyTorch et diffusers ?

Le pari est tenu à une condition : ne rien changer AU-DESSUS. Les onglets, les
préréglages, le catalogue, les formats natifs, les LoRA, la passe HD, l'outpaint
— tout garde ses identifiants et sa forme. Ce qui change est ce qu'il y a
derrière `generate.generate()`, et rien d'autre.

Ce que ça coûte, mesuré et non supposé, est écrit dans README-TORCH.md.
"""
