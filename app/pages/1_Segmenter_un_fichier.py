from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

RACINE = Path(__file__).resolve().parent.parent.parent
FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
PLAGES = {"Age": (18, 70), "Annual Income (k$)": (15, 137), "Spending Score (1-100)": (1, 99)}

# Numéros de clusters du modèle sauvegardé (k = 6, random_state = 42)
SEGMENTS = {
    0: ("Seniors modérés", "Services de confort, horaires calmes, offres bien-être."),
    1: ("Jeunes adultes modérés", "Offres mode et loisirs, parrainage, événements en centre."),
    2: ("Aisés économes", "Les attirer vers le haut de gamme : offres ciblées, valeur perçue."),
    3: ("Aisés dépensiers (VIP)", "Programme de fidélité premium, avant-premières, services exclusifs."),
    4: ("Jeunes dépensiers à petit budget", "Promotions, paiement échelonné, offres étudiantes, réseaux sociaux."),
    5: ("Budget serré, dépenses faibles", "Bons plans et soldes, sans investissement marketing lourd."),
}


@st.cache_resource
def charger_modeles():
    kmeans = joblib.load(RACINE / "models" / "kmeans.joblib")
    scaler = joblib.load(RACINE / "models" / "scaler.joblib")
    return kmeans, scaler


@st.cache_data
def charger_clients():
    chemin = RACINE / "data" / "clients_segmentes.csv"
    return pd.read_csv(chemin) if chemin.exists() else None


def seuils_par_segment(clients, kmeans, scaler, quantile=0.95):
    """Distance maximale 'normale' de chaque segment (95 % de ses membres en dessous)."""
    if clients is None or "Cluster" not in clients.columns:
        return None
    z = scaler.transform(clients[FEATURES])
    labels = clients["Cluster"].to_numpy()
    d = np.linalg.norm(z - kmeans.cluster_centers_[labels], axis=1)
    return np.array([np.quantile(d[labels == c], quantile)
                     for c in range(len(kmeans.cluster_centers_))])


def segmenter(donnees, kmeans, scaler, seuils):
    """Ajoute segment, voisin, distance, position et action à chaque client."""
    z = scaler.transform(donnees[FEATURES])
    d = np.linalg.norm(z[:, None, :] - kmeans.cluster_centers_[None, :, :], axis=2)
    ordre = np.argsort(d, axis=1)
    p, s = ordre[:, 0], ordre[:, 1]
    lignes = np.arange(len(d))
    d1, d2 = d[lignes, p], d[lignes, s]
    rapport = d1 / d2

    res = donnees.copy()
    res["Segment"] = [SEGMENTS[i][0] for i in p]
    res["Segment voisin"] = [SEGMENTS[i][0] for i in s]
    res["Distance"] = d1.round(2)
    res["Position"] = np.select([rapport < 0.5, rapport < 0.8],
                                ["clairement", "plutôt"], default="frontière")
    if seuils is not None:
        res["Atypique"] = np.where(d1 > seuils[p], "oui", "non")
    res["Action marketing"] = [SEGMENTS[i][1] for i in p]
    return res


# ---------------------------- Interface ----------------------------
st.set_page_config(page_title="Segmenter un fichier", page_icon="📂", layout="wide")
st.title(" Segmenter un fichier de clients")
st.write("Chargez un fichier CSV : chaque client reçoit son segment, son segment voisin "
         "et l'action marketing associée. Colonnes obligatoires : "
         "`Age`, `Annual Income (k$)`, `Spending Score (1-100)`. "
         "Les autres colonnes sont conservées.")

modele = pd.DataFrame({"Age": [25, 40], "Annual Income (k$)": [30, 80],
                       "Spending Score (1-100)": [75, 20]})
st.download_button("Télécharger un modèle de fichier", modele.to_csv(index=False).encode("utf-8"),
                   "modele_clients.csv", "text/csv")

fichier = st.file_uploader("Fichier CSV de clients", type=["csv"])
if fichier is None:
    st.stop()

kmeans, scaler = charger_modeles()
seuils = seuils_par_segment(charger_clients(), kmeans, scaler)

try:
    brut = pd.read_csv(fichier, sep=None, engine="python")  # détecte , ou ;
except Exception as erreur:
    st.error(f"Impossible de lire le fichier : {erreur}")
    st.stop()

brut.columns = brut.columns.str.strip()
manquantes = [c for c in FEATURES if c not in brut.columns]
if manquantes:
    st.error(f"Colonnes manquantes : {', '.join(manquantes)}")
    st.write("Colonnes trouvées dans le fichier :", list(brut.columns))
    st.stop()

for c in FEATURES:
    brut[c] = pd.to_numeric(brut[c], errors="coerce")
valides = brut.dropna(subset=FEATURES).reset_index(drop=True)
ignorees = len(brut) - len(valides)
if ignorees:
    st.warning(f"{ignorees} ligne(s) ignorée(s) : valeur manquante ou non numérique.")
if valides.empty:
    st.error("Aucun client exploitable dans ce fichier.")
    st.stop()

resultat = segmenter(valides, kmeans, scaler, seuils)

hors = pd.Series(False, index=resultat.index)
for c, (bas, haut) in PLAGES.items():
    hors |= (resultat[c] < bas) | (resultat[c] > haut)
resultat["Hors plage"] = np.where(hors, "oui", "non")
if hors.any():
    st.info(f"{int(hors.sum())} client(s) sortent de la plage des données d'entraînement "
            "(âge 18-70, revenu 15-137 k$, score 1-99) : résultat moins fiable.")

col1, col2, col3 = st.columns(3)
col1.metric("Clients segmentés", len(resultat))
if "Atypique" in resultat.columns:
    col2.metric("Profils atypiques", int((resultat["Atypique"] == "oui").sum()))
col3.metric("À la frontière", int((resultat["Position"] == "frontière").sum()))

st.subheader("Répartition par segment")
st.bar_chart(resultat["Segment"].value_counts())

st.subheader("Résultat détaillé")
st.dataframe(resultat, hide_index=True)

sortie = resultat.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
st.download_button("Télécharger les résultats (CSV)", sortie,
                   "clients_segmentes_resultat.csv", "text/csv")