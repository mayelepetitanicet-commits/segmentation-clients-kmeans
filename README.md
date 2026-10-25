# Segmentation de clients d'un centre commercial (K-Means)

Projet de data science : regrouper 200 clients en segments homogènes pour proposer des actions marketing adaptées.

**Démonstration en ligne** : https://segmentation-clients-kmeans.streamlit.app/

## Données

Jeu de données *Mall Customers* (200 clients). Variables utilisées pour le clustering : âge, revenu annuel (k$) et score de dépenses (1-100). Le genre, le niveau d'études et la situation familiale servent uniquement au profilage des segments.

## Méthode

1. Chargement et contrôle qualité (aucune valeur manquante, aucun doublon)
2. Exploration visuelle (distributions, corrélations, nuages de points)
3. Standardisation des trois variables (`StandardScaler`)
4. Choix de k par la méthode du coude et le score de silhouette (k = 2 à 10)
5. Entraînement du K-Means avec k = 6 (silhouette = 0,43)
6. Interprétation et profilage des segments
7. Application web de démonstration avec Streamlit

## Résultats

| Segment | Effectif | Profil |
|---|---|---|
| Aisés dépensiers (VIP) | 39 | revenu élevé, dépenses élevées |
| Seniors modérés | 45 | environ 56 ans, revenu et dépenses moyens |
| Jeunes adultes modérés | 39 | environ 27 ans, revenu et dépenses moyens |
| Aisés économes | 33 | revenu élevé, dépenses faibles |
| Jeunes dépensiers à petit budget | 23 | environ 25 ans, petit revenu, dépenses élevées |
| Budget serré, dépenses faibles | 21 | petit revenu, dépenses faibles |

## Limites

- Seulement 200 clients, données pédagogiques.
- Silhouette modeste (0,43) : les groupes se chevauchent en partie, surtout les deux segments « modérés », qui ne diffèrent que par l'âge.
- Le genre, l'éducation et la situation familiale ne différencient pas nettement les segments.
- Les recommandations marketing sont des hypothèses à tester, pas des preuves.

## Lancer le projet en local

```bash
git clone URL_DU_DEPOT
cd segmentation-clients-kmeans
conda create -n segmentation python=3.11 -y
conda activate segmentation
pip install -r requirements.txt
streamlit run app/app.py
```

Pour rejouer l'analyse complète, installez aussi `requirements-dev.txt` et ouvrez `notebooks/01_segmentation.ipynb`.

## Structure

```
app/        application Streamlit
data/       données brutes et clients segmentés
models/     modèle K-Means et scaler sauvegardés
notebooks/  analyse complète
```
