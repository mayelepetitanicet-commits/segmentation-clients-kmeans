from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from scipy.stats import chi2_contingency

RACINE = Path(__file__).resolve().parent.parent.parent

NUM = {"Âge": "Age", "Revenu annuel (k$)": "Revenu", "Score de dépenses": "Score"}
CAT = {"Genre": "Gender", "Niveau d'études": "Education", "Situation familiale": "Marital Status"}

st.set_page_config(page_title="Profils détaillés", page_icon="🔎", layout="wide")


@st.cache_data
def charger_clients():
    chemin = RACINE / "data" / "clients_segmentes.csv"
    return pd.read_csv(chemin) if chemin.exists() else None


def composition(df, colonne):
    """Part (en %) de chaque modalité dans chaque segment."""
    tab = pd.crosstab(df["Segment"], df[colonne], normalize="index") * 100
    return tab.round(1).reset_index().melt(id_vars="Segment", var_name="Modalite", value_name="Part")


def comparer(df, seg_a, seg_b):
    """Moyennes de deux segments, écart brut et écart en écarts-types de l'ensemble."""
    colonnes = list(NUM.values())
    moyennes = df.groupby("Segment")[colonnes].mean()
    sigma = df[colonnes].std()
    res = pd.DataFrame({seg_a: moyennes.loc[seg_a], seg_b: moyennes.loc[seg_b]})
    res["Écart"] = res[seg_a] - res[seg_b]
    res["Écart (en écarts-types)"] = res["Écart"] / sigma
    return res.round(2)


# ---------------------------- Interface ----------------------------
st.title("🔎 Profils détaillés des segments")

clients = charger_clients()
if clients is None or "Segment" not in clients.columns:
    st.error("Le fichier data/clients_segmentes.csv est introuvable ou incomplet.")
    st.stop()

df = clients.copy()
df.columns = df.columns.str.strip()
df = df.rename(columns={"Annual Income (k$)": "Revenu", "Spending Score (1-100)": "Score"})

# 1. Distribution d'une variable numérique
st.subheader("Distribution d'une variable par segment")
nom_var = st.selectbox("Variable", list(NUM))
boites = alt.Chart(df).mark_boxplot(extent="min-max").encode(
    x=alt.X("Segment:N", title=None, axis=alt.Axis(labelAngle=-30)),
    y=alt.Y(f"{NUM[nom_var]}:Q", title=nom_var),
    color=alt.Color("Segment:N", legend=None),
)
st.altair_chart(boites)

# 2. Composition par variable catégorielle
st.subheader("Composition des segments")
nom_cat = st.selectbox("Variable catégorielle", list(CAT))
colonne_cat = CAT[nom_cat]
if colonne_cat not in df.columns:
    st.info(f"La colonne {colonne_cat} n'existe pas dans les données.")
else:
    comp = composition(df, colonne_cat)
    barres = alt.Chart(comp).mark_bar().encode(
        y=alt.Y("Segment:N", title=None),
        x=alt.X("Part:Q", title="Part du segment (%)"),
        color=alt.Color("Modalite:N", title=nom_cat),
        tooltip=["Segment", "Modalite", "Part"],
    )
    st.altair_chart(barres)

    chi2, p, _, _ = chi2_contingency(pd.crosstab(df["Segment"], df[colonne_cat]))
    col_a, col_b = st.columns([1, 3])
    col_a.metric("p-value (khi-deux)", f"{p:.3f}")
    if p < 0.05:
        col_b.write("Écart statistiquement significatif entre les segments (p < 0,05).")
    else:
        col_b.write("Écarts compatibles avec le hasard (p ≥ 0,05) : cette variable "
                    "ne distingue pas nettement les segments.")
    st.caption("Chaque segment compte 21 à 45 clients : de petits écarts ne sont pas "
               "significatifs, et le test est approximatif quand certaines cases ont moins "
               "de 5 clients. « Unknown » désigne une information non renseignée, pas un profil.")

# 3. Comparaison de deux segments
st.subheader("Comparer deux segments")
noms = sorted(df["Segment"].unique())
c1, c2 = st.columns(2)
seg_a = c1.selectbox("Segment A", noms, index=0)
seg_b = c2.selectbox("Segment B", noms, index=min(1, len(noms) - 1))
if seg_a == seg_b:
    st.info("Choisissez deux segments différents.")
else:
    resultat = comparer(df, seg_a, seg_b)
    st.dataframe(resultat)
    st.caption("Écart en écarts-types : 1 signifie que les moyennes diffèrent d'un écart-type "
               "de l'ensemble des clients (grande différence).")
    st.bar_chart(resultat["Écart (en écarts-types)"])