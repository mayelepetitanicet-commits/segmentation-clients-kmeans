from pathlib import Path
import altair as alt
import pandas as pd
import streamlit as st

RACINE = Path(__file__).resolve().parent.parent.parent

st.set_page_config(page_title="Tableau de bord", page_icon="📊", layout="wide")


@st.cache_data
def charger_clients():
    chemin = RACINE / "data" / "clients_segmentes.csv"
    return pd.read_csv(chemin) if chemin.exists() else None


def profil_segments(df):
    """Effectif, part et moyennes de chaque segment."""
    profil = df.groupby("Segment").agg(
        Effectif=("Segment", "size"),
        Age=("Age", "mean"),
        Revenu=("Annual Income (k$)", "mean"),
        Score=("Spending Score (1-100)", "mean"),
    )
    profil["Part (%)"] = 100 * profil["Effectif"] / profil["Effectif"].sum()
    return profil.round(1).sort_values("Effectif", ascending=False)


def filtrer(df, segments, age, revenu, genres):
    masque = (df["Segment"].isin(segments)
              & df["Age"].between(*age)
              & df["Annual Income (k$)"].between(*revenu))
    if genres and "Gender" in df.columns:
        masque &= df["Gender"].isin(genres)
    return df[masque]


# ---------------------------- Interface ----------------------------
st.title("📊 Tableau de bord des segments")

clients = charger_clients()
if clients is None or "Segment" not in clients.columns:
    st.error("Le fichier data/clients_segmentes.csv est introuvable ou incomplet.")
    st.stop()

df = clients.copy()
df.columns = df.columns.str.strip()

with st.sidebar:
    st.header("Filtres")
    tous = sorted(df["Segment"].unique())
    segments = st.multiselect("Segments", tous, default=tous)
    a_min, a_max = int(df["Age"].min()), int(df["Age"].max())
    age = st.slider("Âge", a_min, a_max, (a_min, a_max))
    r_min, r_max = int(df["Annual Income (k$)"].min()), int(df["Annual Income (k$)"].max())
    revenu = st.slider("Revenu annuel (k$)", r_min, r_max, (r_min, r_max))
    genres = []
    if "Gender" in df.columns:
        options = sorted(df["Gender"].unique())
        genres = st.multiselect("Genre", options, default=options)

vue = filtrer(df, segments, age, revenu, genres)
if vue.empty:
    st.warning("Aucun client ne correspond à ces filtres.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Clients", len(vue), f"{len(vue) - len(df):+d} vs total" if len(vue) != len(df) else None)
for colonne, nom, zone in [("Age", "Âge moyen", col2),
                           ("Annual Income (k$)", "Revenu moyen (k$)", col3),
                           ("Spending Score (1-100)", "Score de dépenses moyen", col4)]:
    ecart = vue[colonne].mean() - df[colonne].mean()
    zone.metric(nom, f"{vue[colonne].mean():.1f}",
                f"{ecart:+.1f} vs ensemble" if abs(ecart) >= 0.05 else None)

st.subheader("Effectif par segment")
comptes = vue["Segment"].value_counts().rename_axis("Segment").reset_index(name="Clients")
st.altair_chart(
    alt.Chart(comptes).mark_bar().encode(
        x=alt.X("Clients:Q", title="Nombre de clients"),
        y=alt.Y("Segment:N", sort="-x", title=None),
        tooltip=["Segment", "Clients"],
    )
)

st.subheader("Profil moyen des segments")
st.dataframe(profil_segments(vue))

st.subheader("Nuage de points")
onglet1, onglet2 = st.tabs(["Revenu et score de dépenses", "Âge et score de dépenses"])
with onglet1:
    st.scatter_chart(vue, x="Annual Income (k$)", y="Spending Score (1-100)", color="Segment")
with onglet2:
    st.scatter_chart(vue, x="Age", y="Spending Score (1-100)", color="Segment")

with st.expander("Voir les clients filtrés"):
    st.dataframe(vue, hide_index=True)