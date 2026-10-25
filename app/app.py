from pathlib import Path

import joblib
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

st.set_page_config(page_title="Segmentation clients", page_icon="🛍️")


@st.cache_resource
def charger_modeles():
    kmeans = joblib.load(RACINE / "models" / "kmeans.joblib")
    scaler = joblib.load(RACINE / "models" / "scaler.joblib")
    return kmeans, scaler


kmeans, scaler = charger_modeles()

st.title("🛍️ Segmentation des clients du centre commercial")
st.write("Saisissez le profil d'un client pour connaître son segment (K-Means, 6 groupes).")

col1, col2, col3 = st.columns(3)
age = col1.slider("Âge", 18, 80, 35)
revenu = col2.slider("Revenu annuel (k$)", 10, 150, 60)
score = col3.slider("Score de dépenses (1-100)", 1, 100, 50)

if st.button("Identifier le segment", type="primary"):
    client = pd.DataFrame([[age, revenu, score]], columns=FEATURES)
    cluster = int(kmeans.predict(scaler.transform(client))[0])
    nom, description, action = SEGMENTS[cluster]

    st.success(f"Segment : **{nom}**")
    st.write(description)
    st.info(f"Action marketing proposée : {action}")

chemin_csv = RACINE / "data" / "clients_segmentes.csv"
if chemin_csv.exists():
    with st.expander("Voir les 200 clients segmentés"):
        donnees = pd.read_csv(chemin_csv)
        st.scatter_chart(donnees, x="Annual Income (k$)",
                         y="Spending Score (1-100)", color="Segment")
        st.dataframe(donnees)
