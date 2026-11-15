# Projet : Question-Adaptive Visual Token Selection (VLM)

## Contexte
- Notebook unique : `Question_Adaptive_VLM.ipynb`. Le plan complet, section par section, est dans `PLAN.md` : c'est la référence.
- Architecture : CLIP ViT-B/16 (gelé) → encodeur de question CLIP text (gelé) → selector (entraîné) → Top-K → projector MLP (entraîné) → Qwen2.5-0.5B-Instruct (gelé + LoRA). Dataset : VQAv2 (sous-ensemble), métrique VQA officielle.
- Les sections 1 à 3 sont terminées et validées. Elles définissent `cfg` (toute la config), `DEVICE`, `DTYPE`, `USE_GRAD_SCALER`, `set_seed`, `seed_worker`, `make_generator`, `load_or_compute`, `show`, `count_params`, `gpu_mem`.

## Méthode de travail (obligatoire)
1. **Une seule cellule à la fois.** Avant d'écrire une cellule : expliquer en 3 à 6 lignes ce qu'elle va faire et le concept en jeu. Ensuite, écrire la cellule, puis **attendre que j'exécute et que je valide** avant la suivante.
2. Chaque sous-étape = une cellule markdown courte (objectif, concept, formule si utile) + une cellule de code.
3. Chaque section se termine par une **cellule de vérification** avec des `assert` (voir les « Vérification » de `PLAN.md`) et un récapitulatif ✅ / ❌.
4. Je suis là pour **comprendre** : expliquer les choix non évidents (formes des tenseurs, axe du softmax, pourquoi tel masque…), en anglais.

## Règles de code
- Aucun hyperparamètre en dur : tout passe par `cfg`. Si un nouveau paramètre est nécessaire, me proposer de l'ajouter à `Config` (section 3.6).
- Tout ce qui coûte cher à calculer passe par `load_or_compute(cfg.path("cache", ...), ...)`, pour que le notebook soit redémarrable.
- Le code doit tourner en `DEBUG = True` (~200 exemples, quelques minutes) **et** en mode complet sans modification.
- Utiliser `show(nom, tenseur)` pour afficher les formes, en particulier sur l'exemple du fil rouge (« What color is the car? »).
- Ne pas modifier une cellule déjà validée sans me le demander. Ne pas réinstaller `torch`.
- Code clair et commenté plutôt que compact. Pas d'abstraction prématurée : le refactoring en `src/` viendra après.

## Honnêteté des résultats
- **Ne jamais inventer un chiffre** (précision, latence, nombre de paramètres attendu…). Tout résultat écrit dans le markdown doit venir d'une cellule exécutée.
- Si un test échoue ou si un résultat est surprenant, le dire et proposer une explication à vérifier, plutôt que d'ajuster le test.
