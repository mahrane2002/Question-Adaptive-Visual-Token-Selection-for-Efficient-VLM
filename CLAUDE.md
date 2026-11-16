# Projet : Question-Adaptive Visual Token Selection (VLM)

## Contexte
- Notebook unique : `Question_Adaptive_VLM.ipynb`. Le plan complet, section par section, est dans `PLAN.md` : c'est la référence.
- Architecture : CLIP ViT-B/16 (gelé) → encodeur de question CLIP text (gelé) → selector (entraîné) → Top-K → projector MLP (entraîné) → Qwen2.5-0.5B-Instruct (gelé + LoRA). Dataset : VQAv2 (sous-ensemble), métrique VQA officielle.
- Les sections 1 à 3 sont terminées et validées. Elles définissent `cfg` (toute la config), `DEVICE`, `DTYPE`, `USE_GRAD_SCALER`, `set_seed`, `seed_worker`, `make_generator`, `load_or_compute`, `show`, `count_params`, `gpu_mem`.
- La section 4 est écrite. Elle définit `vqa_df` (VQAv2 complet), `SPLIT_IDS`, `data_df` (le sous-ensemble avec `split`, `answers`, `target`, `oracle_acc`), `coco_image_path`, `normalize_answer`, `vqa_accuracy`, `vqa_accuracy_batch` et `EXAMPLE` (l'exemple du fil rouge).
- Objectif final : un projet de recherche en IA techniquement correct, reproductible et présentable sur GitHub et dans un CV.

## Méthode de travail (obligatoire)
1. **Une seule cellule à la fois.** Avant d'écrire une cellule : expliquer en 3 à 6 lignes ce qu'elle va faire et le concept en jeu. Ensuite, écrire la cellule, puis **attendre que j'exécute et que je valide** avant la suivante.
2. Chaque sous-étape = une cellule markdown courte (objectif, concept, formule si utile) + une cellule de code.
3. Chaque section se termine par une **cellule de vérification** avec des `assert` (voir les « Vérification » de `PLAN.md`) et un récapitulatif ✅ / ❌.
4. Je suis là pour **comprendre** : expliquer les choix non évidents (formes des tenseurs, axe du softmax, pourquoi tel masque…), en anglais.

## Simplicité (prioritaire)
- **Je dois pouvoir comprendre, expliquer et défendre chaque ligne lors d'un entretien technique.** Ne pas chercher à rendre le code sophistiqué : un code court, clair et professionnel.
- Éviter les abstractions, les classes et les fonctions inutiles. Une fonction ne se justifie que si elle est réutilisée ou si elle rend le code nettement plus lisible. Les modules PyTorch (`nn.Module`) restent la forme normale pour les composants du modèle (selector, projector, budget head…).
- Utiliser PyTorch et Hugging Face (`transformers`, `peft`) autant que possible plutôt que réimplémenter. Exception : les implémentations « à la main » prévues par `PLAN.md` pour comprendre (ex. le cross-attention de la section 10), toujours validées contre l'équivalent PyTorch.
- Pas de dépendance supplémentaire si elle n'est pas nécessaire (les versions sont figées en 3.2).
- Garder une architecture cohérente d'un bout à l'autre du notebook (mêmes noms, mêmes conventions de formes).
- Ne supprimer aucune fonctionnalité importante du projet (tout ce qui est dans `PLAN.md` reste à faire).

## Règles de code
- **Vérifier les dimensions des tenseurs (entrée et sortie) à chaque étape** : `show(...)` pour les afficher, `assert x.shape == (...)` pour les garantir.
- **Tester chaque composant avant de passer au suivant** : un petit test dans la cellule, puis la cellule de vérification de la section.
- Aucun hyperparamètre en dur : tout passe par `cfg`. Si un nouveau paramètre est nécessaire, me proposer de l'ajouter à `Config` (section 3.6).
- Tout ce qui coûte cher à calculer passe par `load_or_compute(cfg.path("cache", ...), ...)`, pour que le notebook soit redémarrable.
- Le code doit tourner en `DEBUG = True` (~200 exemples, quelques minutes) **et** en mode complet sans modification.
- Utiliser `show(nom, tenseur)` pour afficher les formes, en particulier sur l'exemple du fil rouge (« What color is the car? »).
- Ne pas modifier une cellule déjà validée sans me le demander. Ne pas réinstaller `torch`.
- Commentaires uniquement lorsqu'ils sont utiles : le *pourquoi*, les formes des tenseurs, un piège. Ne pas paraphraser le code.
- Pas d'abstraction prématurée : le refactoring en `src/` viendra après.

## Honnêteté des résultats
- **Ne jamais inventer un chiffre** (précision, latence, nombre de paramètres attendu…). Tout résultat écrit dans le markdown doit venir d'une cellule exécutée.
- Si un test échoue ou si un résultat est surprenant, le dire et proposer une explication à vérifier, plutôt que d'ajuster le test.
