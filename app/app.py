"""
MixCraft - Interface Streamlit
3 onglets : Recommandation, Generation, Analyse
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from src.data_loader import load_all_datasets
from src.embeddings import EmbeddingEngine
from src.recommender import CocktailRecommender
from src.rag_pipeline import RAGPipeline

# ---------- Page config ----------
st.set_page_config(
    page_title="MixCraft AI",
    page_icon="🍹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- CSS EFREI branding ----------
st.markdown("""
<style>
  :root {
    --efrei-navy: #163767;
    --efrei-pink: #FF43B8;
    --efrei-dark: #051832;
    --efrei-blue: #0C78B4;
  }
  .stApp { background-color: #f8f9fc; }
  h1, h2, h3 { color: var(--efrei-navy); font-family: 'Helvetica Neue', sans-serif; }
  .metric-card {
    background: white;
    border-left: 4px solid var(--efrei-blue);
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }
  .cocktail-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem;
    margin: 0.5rem 0;
    border: 1px solid #e8edf5;
    box-shadow: 0 2px 6px rgba(22,55,103,0.08);
  }
  .score-badge {
    background: var(--efrei-navy);
    color: white;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.85rem;
  }
  .tag {
    background: #e8f0fe;
    color: var(--efrei-navy);
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.8rem;
    margin: 2px;
    display: inline-block;
  }
</style>
""", unsafe_allow_html=True)


# ---------- Chargement modele (cache Streamlit) ----------
@st.cache_resource(show_spinner="Chargement du corpus et du modele...")
def get_system():
    df = load_all_datasets()
    engine = EmbeddingEngine(cache=True)
    rec = CocktailRecommender(engine=engine)
    rec.fit(df)
    rag = RAGPipeline(recommender=rec, engine=engine)
    return df, rec, rag


# ---------- Radar chart ----------
def radar_chart(flavor_profile: dict, cocktail_name: str) -> go.Figure:
    categories = list(flavor_profile.keys())
    values = list(flavor_profile.values())
    values += values[:1]  # fermer le radar
    categories += categories[:1]

    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(22,55,103,0.15)",
        line=dict(color="#163767", width=2),
        name=cocktail_name,
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=11, color="#163767")),
        ),
        showlegend=False,
        margin=dict(t=20, b=20, l=30, r=30),
        height=300,
    )
    return fig


# ---------- Header ----------
st.markdown("""
<div style="background: linear-gradient(135deg, #163767 0%, #0C78B4 100%);
            padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;">
  <h1 style="color: white; margin: 0; font-size: 2rem;">MixCraft AI</h1>
  <p style="color: rgba(255,255,255,0.85); margin: 0.5rem 0 0 0;">
    Recommandation et creation de cocktails par analyse semantique
  </p>
  <p style="color: rgba(255,255,255,0.6); font-size: 0.85rem; margin: 0.5rem 0 0 0;">
    EFREI Paris - M1 Data Engineering & IA - Adam Beloucif & Emilien Morice
  </p>
</div>
""", unsafe_allow_html=True)

# Chargement
df, rec, rag = get_system()

# ---------- Onglets ----------
tab_reco, tab_gen, tab_analyse = st.tabs(["Recommandation", "Generation", "Analyse"])


# ===== ONGLET 1 : RECOMMANDATION =====
with tab_reco:
    st.markdown("### Decouvrez votre cocktail ideal")
    st.markdown("Decrivez vos envies en langage naturel - le systeme trouve les cocktails les plus proches semantiquement.")

    col_input, col_filters = st.columns([3, 1])

    with col_input:
        query = st.text_input(
            "Votre envie",
            placeholder="ex. quelque chose de frais et fruité avec des agrumes, peu alcoolisé",
            help="Decrivez librement vos preferences de saveur, occasion, ingredients...",
        )

    with col_filters:
        categories = ["Toutes"] + sorted(df["category"].dropna().unique().tolist())
        cat_filter = st.selectbox("Categorie", categories)
        top_k = st.slider("Nombre de resultats", 3, 10, 5)

    if st.button("Rechercher", type="primary"):
        if not query.strip():
            st.warning("Entrez une description pour commencer.")
        else:
            category_arg = None if cat_filter == "Toutes" else cat_filter
            results = rec.recommend_by_query(query, top_k=top_k, category_filter=category_arg)

            if not results:
                st.error("Aucun resultat. Essayez une autre description.")
            else:
                st.markdown(f"**{len(results)} cocktails trouves** pour '{query}'")

                for r in results:
                    with st.container():
                        st.markdown(f"""
                        <div class="cocktail-card">
                          <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h4 style="margin: 0; color: #163767;">{r.name}</h4>
                            <span class="score-badge">Score {r.similarity_score:.2f}</span>
                          </div>
                          <span class="tag">{r.category}</span>
                          <p style="font-size: 0.9rem; color: #555; margin: 0.5rem 0;">
                            <strong>Ingredients :</strong> {r.ingredients[:120]}...
                          </p>
                        </div>
                        """, unsafe_allow_html=True)

                        if r.flavor_profile:
                            with st.expander(f"Profil de saveurs - {r.name}"):
                                fig = radar_chart(r.flavor_profile, r.name)
                                st.plotly_chart(fig, use_container_width=True)
                                st.markdown(f"**Instructions :** {r.instructions}")

    # Recommandation par ingredients
    st.markdown("---")
    st.markdown("### Par ingredients disponibles")
    ing_input = st.text_input(
        "Vos ingredients",
        placeholder="vodka, citron, gingembre, sucre",
    )

    if ing_input and st.button("Trouver des recettes", key="by_ing"):
        ingredients = [i.strip() for i in ing_input.split(",") if i.strip()]
        results = rec.recommend_by_ingredients(ingredients, top_k=5)
        for r in results:
            st.markdown(f"**{r.rank}. {r.name}** (score: {r.similarity_score:.2f})")
            st.markdown(f"*{r.ingredients}*")
            st.markdown("---")


# ===== ONGLET 2 : GENERATION =====
with tab_gen:
    st.markdown("### Generez une recette originale")
    st.markdown("Entrez vos ingredients et le systeme cree une recette personnalisee via le pipeline RAG.")

    gen_query = st.text_area(
        "Description ou ingredients",
        placeholder="ex. rhum blanc, menthe, citron vert, eau gazeuse - cocktail estival pour une soiree d'ete",
        height=100,
    )

    col_params = st.columns(3)
    with col_params[0]:
        temperature = st.slider("Creativite", 0.3, 1.2, 0.7, 0.1)
    with col_params[1]:
        ctx_k = st.slider("Cocktails de reference", 1, 5, 3)
    with col_params[2]:
        generate_text = st.checkbox("Generation de texte", value=True)

    if st.button("Generer", type="primary"):
        if not gen_query.strip():
            st.warning("Entrez une description ou des ingredients.")
        else:
            with st.spinner("Generation en cours..."):
                result = rag.query(
                    gen_query,
                    top_k=ctx_k,
                    temperature=temperature,
                    generate=generate_text,
                )

            if result["status"] == "rejected":
                st.error(
                    f"Requete hors-domaine cocktails (similarite max : {result['max_similarity']:.2f} < seuil {rag.guardrail_threshold}).\n\n"
                    "Le systeme est specialise en cocktails. Essayez une description liee aux boissons."
                )
            else:
                if result["cached"]:
                    st.info("Resultat depuis le cache (meme requete deja traitee).")

                if result.get("generated_recipe"):
                    st.markdown("#### Recette generee")
                    st.markdown(f"""
                    <div style="background: white; border-left: 4px solid #FF43B8;
                                border-radius: 8px; padding: 1.2rem; margin: 1rem 0;
                                box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                    {result['generated_recipe'].replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)

                if result.get("retrieved_cocktails"):
                    st.markdown("#### Cocktails de reference utilises")
                    for c in result["retrieved_cocktails"][:3]:
                        if isinstance(c, dict):
                            st.markdown(f"- **{c.get('name', '?')}** - Score {c.get('similarity_score', 0):.2f}")


# ===== ONGLET 3 : ANALYSE =====
with tab_analyse:
    st.markdown("### Exploration du corpus")
    st.markdown(f"**{len(df)} cocktails** dans le corpus.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Distribution par categorie")
        cat_counts = df["category"].value_counts().head(12)
        import plotly.express as px
        fig_cat = px.bar(
            x=cat_counts.values,
            y=cat_counts.index,
            orientation="h",
            color_discrete_sequence=["#163767"],
            labels={"x": "Nombre", "y": "Categorie"},
        )
        fig_cat.update_layout(margin=dict(t=10, b=10), height=350)
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_b:
        st.markdown("#### Top 15 ingredients")
        all_ings = df["ingredients"].fillna("").str.lower().str.split(r"[,\n]").explode()
        all_ings = all_ings.str.strip().str.replace(r"\d+ml\s*", "", regex=True).str.strip()
        ing_counts = all_ings[all_ings.str.len() > 2].value_counts().head(15)
        fig_ing = px.bar(
            x=ing_counts.values,
            y=ing_counts.index,
            orientation="h",
            color_discrete_sequence=["#0C78B4"],
            labels={"x": "Occurrences", "y": "Ingredient"},
        )
        fig_ing.update_layout(margin=dict(t=10, b=10), height=350)
        st.plotly_chart(fig_ing, use_container_width=True)

    # Profil de saveurs moyen du corpus
    st.markdown("#### Profil de saveurs moyen du corpus")
    flavor_cols = [c for c in ["sweet", "sour", "bitter", "strong", "fresh", "fruity"] if c in df.columns]
    if flavor_cols:
        avg_profile = df[flavor_cols].mean().to_dict()
        fig_radar = radar_chart(avg_profile, "Corpus complet")
        col_rad, _ = st.columns([1, 2])
        with col_rad:
            st.plotly_chart(fig_radar, use_container_width=True)

    # Tableau de donnees
    st.markdown("#### Parcourir le corpus")
    search_term = st.text_input("Rechercher par nom ou ingredient", "")
    filtered = df
    if search_term:
        mask = (
            df["name"].str.lower().str.contains(search_term.lower(), na=False)
            | df["ingredients"].str.lower().str.contains(search_term.lower(), na=False)
        )
        filtered = df[mask]

    st.dataframe(
        filtered[["name", "category", "glass", "ingredients"]].rename(columns={
            "name": "Nom", "category": "Categorie", "glass": "Verre", "ingredients": "Ingredients"
        }),
        use_container_width=True,
        height=350,
    )
