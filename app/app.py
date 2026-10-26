from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

RACINE = Path(__file__).resolve().parent.parent
FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

# Numéros de clusters du modèle sauvegardé (k = 6, random_state = 42)
SEGMENTS = {
    0: ("Seniors modérés",
        "Clients de 40 à 70 ans, revenu et dépenses moyens.",
        "Services de confort, horaires calmes, offres bien-être."),
    1: ("Jeunes adultes modérés",
        "Clients de 18 à 40 ans, revenu et dépenses moyens.",
        "Offres mode et loisirs, parrainage, événements en centre."),
    2: ("Aisés économes",
        "Revenu élevé, dépenses faibles.",
        "Les attirer vers le haut de gamme : offres ciblées, valeur perçue."),
    3: ("Aisés dépensiers (VIP)",
        "Revenu élevé, dépenses élevées.",
        "Programme de fidélité premium, avant-premières, services exclusifs."),
    4: ("Jeunes dépensiers à petit budget",
        "Jeunes, petit revenu, dépenses élevées.",
        "Promotions, paiement échelonné, offres étudiantes, réseaux sociaux."),
    5: ("Budget serré, dépenses faibles",
        "Petit revenu, dépenses faibles.",
        "Bons plans et soldes, sans investissement marketing lourd."),
}

st.set_page_config(page_title="Segmentation clients", page_icon="🛍️", layout="wide")


@st.cache_resource
def charger_modeles():
    kmeans = joblib.load(RACINE / "models" / "kmeans.joblib")
    scaler = joblib.load(RACINE / "models" / "scaler.joblib")
    return kmeans, scaler


@st.cache_data
def charger_clients():
    chemin = RACINE / "data" / "clients_segmentes.csv"
    return pd.read_csv(chemin) if chemin.exists() else None


def distances_aux_segments(age, revenu, score, kmeans, scaler):
    """Distance du client à chacun des centres (en unités standardisées)."""
    client = pd.DataFrame([[age, revenu, score]], columns=FEATURES)
    z = scaler.transform(client)
    return np.linalg.norm(kmeans.cluster_centers_ - z, axis=1)

def seuil_atypique(clients, kmeans, scaler, cluster, quantile=0.95):
    """Distance au centre en dessous de laquelle se trouvent 95 % des membres du segment."""
    membres = clients[clients["Cluster"] == cluster]
    z = scaler.transform(membres[FEATURES])
    d = np.linalg.norm(kmeans.cluster_centers_[cluster] - z, axis=1)
    return float(np.quantile(d, quantile))

    
kmeans, scaler = charger_modeles()
clients = charger_clients()

st.title("🛍️ Segmentation des clients du centre commercial")
st.write("Réglez le profil d'un client dans le menu de gauche pour connaître son segment "
         "(K-Means, 6 groupes).")

with st.sidebar:
    st.header("Profil du client")
    age = st.slider("Âge", 18, 80, 35)
    revenu = st.slider("Revenu annuel (k$)", 10, 150, 60)
    score = st.slider("Score de dépenses (1-100)", 1, 100, 50)

distances = distances_aux_segments(age, revenu, score, kmeans, scaler)
ordre = np.argsort(distances)
premier, second = int(ordre[0]), int(ordre[1])
rapport = distances[premier] / distances[second]

nom, description, action = SEGMENTS[premier]
col_a, col_b = st.columns([3, 2])

with col_a:
    st.success(f"Segment : **{nom}**")
    st.write(description)
    st.info(f"Action marketing proposée : {action}")

with col_b:
    if rapport < 0.5:
        verdict = "Le client est clairement dans ce segment."
    elif rapport < 0.8:
        verdict = "Le client est plutôt dans ce segment."
    else:
        verdict = "Le client est à la frontière de deux segments."
    st.markdown(f"**Segment voisin :** {SEGMENTS[second][0]}")
    st.write(verdict)
    st.caption(f"Distance au segment : {distances[premier]:.2f} · au voisin : "
               f"{distances[second]:.2f} (unités standardisées). "
               "Seuils indicatifs : moins de 0,5 clair, de 0,5 à 0,8 plutôt, "
               "au-delà frontière.")
    
if clients is not None and "Cluster" in clients.columns:
    seuil = seuil_atypique(clients, kmeans, scaler, premier)
    if distances[premier] > seuil:
        st.warning(
            f"Profil atypique : ce client est plus éloigné de son segment "
            f"({distances[premier]:.2f}) que 95 % des clients du segment "
            f"(seuil : {seuil:.2f}). Le segment proposé est le moins mauvais choix, "
            "pas une description fidèle du client."
        )
        
st.subheader("Distance du client à chaque segment")
tableau = pd.DataFrame({
    "Segment": [SEGMENTS[i][0] for i in range(len(distances))],
    "Distance": np.round(distances, 2),
}).sort_values("Distance")
st.dataframe(tableau, hide_index=True)

if clients is not None and "Segment" in clients.columns:
    st.subheader("Position du client parmi les 200 clients")
    donnees = clients[FEATURES + ["Segment"]].copy()
    donnees["Taille"] = 40
    nouveau = pd.DataFrame([{
        "Age": age,
        "Annual Income (k$)": revenu,
        "Spending Score (1-100)": score,
        "Segment": "Client saisi",
        "Taille": 300,
    }])
    donnees = pd.concat([donnees, nouveau], ignore_index=True)

    onglet1, onglet2 = st.tabs(["Revenu et score de dépenses", "Âge et score de dépenses"])
    with onglet1:
        st.scatter_chart(donnees, x="Annual Income (k$)", y="Spending Score (1-100)",
                         color="Segment", size="Taille")
    with onglet2:
        st.scatter_chart(donnees, x="Age", y="Spending Score (1-100)",
                         color="Segment", size="Taille")

    with st.expander("Voir les 200 clients segmentés"):
        st.dataframe(clients)