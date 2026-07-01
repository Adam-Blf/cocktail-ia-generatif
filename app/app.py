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

from src import load_all_datasets, EmbeddingEngine, CocktailRecommender, RAGPipeline

# ---------- Page config ----------
st.set_page_config(
    page_title="MixCraft AI",
    page_icon="🍹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- CSS MixCraft premium ----------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  :root {
    --navy:   #163767;
    --pink:   #FF43B8;
    --dark:   #051832;
    --blue:   #0C78B4;
    --glass:  rgba(255,255,255,0.76);
    --shadow: 0 8px 32px rgba(22,55,103,0.11);
    --radius: 16px;
  }

  /* ---- Base ---- */
  .stApp {
    background: linear-gradient(145deg, #f0f4ff 0%, #faf0ff 55%, #f0f8ff 100%) !important;
    font-family: 'Inter', 'Helvetica Neue', sans-serif !important;
  }
  .main .block-container {
    padding-top: 1.5rem !important;
    max-width: 1100px !important;
  }

  /* ---- Typography - force contrast partout ---- */
  h1, h2, h3, h4 {
    color: var(--navy) !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: -0.025em !important;
  }
  h3 { font-weight: 700 !important; font-size: 1.25rem !important; }
  h4 { font-weight: 600 !important; }

  /* Texte Streamlit genere uniquement (pas les divs custom HTML) */
  .stMarkdown p, .stMarkdown li, .stMarkdown strong, .stMarkdown em,
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] li,
  [data-testid="stText"] {
    font-family: 'Inter', sans-serif !important;
    color: #1e2a3a !important;
  }
  .stMarkdown strong { color: var(--navy) !important; font-weight: 700 !important; }

  /* Caption / helper text */
  [data-testid="stCaptionContainer"] p,
  [data-testid="stCaptionContainer"] {
    color: #4a5568 !important;
    font-size: 0.82rem !important;
  }

  /* Placeholder */
  ::placeholder { color: #8fa3bf !important; opacity: 1 !important; }
  ::-webkit-input-placeholder { color: #8fa3bf !important; }
  ::-moz-placeholder { color: #8fa3bf !important; }

  /* ---- Custom scrollbar ---- */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--navy), var(--blue));
    border-radius: 99px;
  }

  /* ---- Sidebar ---- */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--dark) 0%, #0d2545 100%) !important;
  }
  [data-testid="stSidebar"] * { color: rgba(255,255,255,0.87) !important; }

  /* ---- Tabs - pill style ---- */
  [data-testid="stTabs"] [role="tablist"] {
    gap: 6px !important;
    background: rgba(255,255,255,0.82) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(22,55,103,0.14) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    width: fit-content !important;
    box-shadow: 0 2px 12px rgba(22,55,103,0.07) !important;
  }
  [data-testid="stTabs"] [role="tab"] {
    border-radius: 8px !important;
    padding: 7px 24px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: #2d3748 !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
    font-family: 'Inter', sans-serif !important;
    border-bottom: none !important;
  }
  [data-testid="stTabs"] [role="tab"] p {
    color: inherit !important;
    font-weight: inherit !important;
  }
  [data-testid="stTabs"] [role="tab"]:hover {
    background: rgba(22,55,103,0.07) !important;
    color: var(--navy) !important;
  }
  [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, var(--navy) 0%, var(--blue) 100%) !important;
    color: #fff !important;
    box-shadow: 0 4px 14px rgba(22,55,103,0.3) !important;
  }
  [data-testid="stTabs"] [role="tab"][aria-selected="true"] p {
    color: #fff !important;
  }
  /* Supprimer l'indicateur rouge/soulignement Streamlit */
  [data-testid="stTabs"] [role="tab"]::after,
  [data-testid="stTabs"] [role="tab"]::before { display: none !important; }
  [data-baseweb="tab-highlight"],
  [data-baseweb="tab-border"] {
    display: none !important;
    background: transparent !important;
    height: 0 !important;
  }
  /* Conteneur des tabs sans border bottom */
  [data-testid="stTabs"] > div:first-child {
    border-bottom: none !important;
  }

  /* ---- Inputs ---- */
  .stTextInput input, .stTextArea textarea {
    border-radius: 10px !important;
    border: 1.5px solid rgba(22,55,103,0.16) !important;
    background: var(--glass) !important;
    backdrop-filter: blur(8px) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    color: var(--dark) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    padding: 10px 14px !important;
  }
  .stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--navy) !important;
    box-shadow: 0 0 0 3px rgba(22,55,103,0.1) !important;
    outline: none !important;
  }
  .stTextInput label, .stTextArea label,
  .stSelectbox label, .stSlider label,
  .stCheckbox label, .stRadio label {
    font-weight: 600 !important;
    color: var(--navy) !important;
    font-size: 0.83rem !important;
    letter-spacing: 0.02em !important;
    font-family: 'Inter', sans-serif !important;
  }
  /* Help text sous les inputs */
  .stTextInput [data-testid="InputInstructions"],
  .stTextArea [data-testid="InputInstructions"] {
    color: #5a7090 !important;
    font-size: 0.78rem !important;
  }

  /* ---- Selectbox ---- */
  [data-testid="stSelectbox"] > div > div {
    border-radius: 10px !important;
    border: 1.5px solid rgba(22,55,103,0.16) !important;
    background: rgba(255,255,255,0.9) !important;
    color: #1e2a3a !important;
  }
  /* Dropdown options */
  [data-testid="stSelectbox"] li,
  [data-testid="stSelectbox"] [role="option"] {
    color: #1e2a3a !important;
    background: white !important;
  }
  [data-testid="stSelectbox"] [role="option"]:hover {
    background: rgba(22,55,103,0.07) !important;
  }
  /* Valeur selectionnee */
  [data-testid="stSelectbox"] > div > div > div {
    color: #1e2a3a !important;
  }

  /* ---- Slider - override Streamlit rouge -> navy ---- */
  /* Thumb */
  [data-testid="stSlider"] [role="slider"] {
    background: var(--navy) !important;
    border-color: var(--navy) !important;
    box-shadow: 0 0 0 3px rgba(22,55,103,0.2) !important;
  }
  /* Track entier */
  [data-testid="stSlider"] [data-baseweb="slider"] > div {
    background: rgba(22,55,103,0.12) !important;
  }
  /* Portion remplie (filled track) */
  [data-testid="stSlider"] [data-baseweb="slider-inner-track"],
  [data-testid="stSlider"] [data-baseweb="slider"] > div > div,
  [data-testid="stSlider"] [data-baseweb="slider-track-fill"] {
    background: linear-gradient(90deg, var(--navy), var(--blue)) !important;
  }
  /* Forcer override sur toutes les couleurs Streamlit rouge (--primary-color) */
  [data-testid="stSlider"] * {
    --primary-color: #163767 !important;
  }

  /* ---- Checkbox ---- */
  .stCheckbox [data-testid="stCheckbox"] span {
    color: #1e2a3a !important;
  }

  /* ---- Primary button ---- */
  button[kind="primary"] {
    background: linear-gradient(135deg, var(--navy) 0%, var(--blue) 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 16px rgba(22,55,103,0.28) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    font-family: 'Inter', sans-serif !important;
  }
  button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(22,55,103,0.38) !important;
  }
  button[kind="secondary"] {
    border-radius: 10px !important;
    border: 1.5px solid rgba(22,55,103,0.3) !important;
    color: var(--navy) !important;
    font-weight: 500 !important;
    background: var(--glass) !important;
    font-family: 'Inter', sans-serif !important;
  }

  /* ---- Cocktail result cards ---- */
  .cocktail-card {
    background: var(--glass);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin: 0.75rem 0;
    border: 1px solid rgba(22,55,103,0.1);
    box-shadow: var(--shadow);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    position: relative;
    overflow: hidden;
  }
  .cocktail-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--navy), var(--blue), var(--pink));
  }
  .cocktail-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 14px 40px rgba(22,55,103,0.17);
  }

  /* ---- Score badge ---- */
  .score-badge {
    background: linear-gradient(135deg, var(--navy), var(--blue));
    color: #fff;
    border-radius: 99px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    box-shadow: 0 2px 8px rgba(22,55,103,0.25);
    font-family: 'Inter', sans-serif;
  }

  /* ---- Category tags ---- */
  .tag {
    background: linear-gradient(135deg, rgba(22,55,103,0.08), rgba(12,120,180,0.08));
    color: var(--navy);
    border: 1px solid rgba(22,55,103,0.18);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.76rem;
    font-weight: 600;
    margin: 3px 2px;
    display: inline-block;
    letter-spacing: 0.03em;
    font-family: 'Inter', sans-serif;
  }

  /* ---- Metric card ---- */
  .metric-card {
    background: var(--glass);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    margin: 0.5rem 0;
    border: 1px solid rgba(22,55,103,0.1);
    box-shadow: var(--shadow);
    position: relative;
    padding-left: 1.8rem;
    overflow: hidden;
  }
  .metric-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: linear-gradient(180deg, var(--blue), var(--navy));
    border-radius: 4px 0 0 4px;
  }

  /* ---- Recipe card (generation result) ---- */
  .recipe-card {
    background: var(--glass);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-left: 4px solid var(--pink);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin: 1rem 0;
    box-shadow: 0 4px 20px rgba(255,67,184,0.1);
    line-height: 1.75;
    font-size: 0.92rem;
  }

  /* ---- Dividers ---- */
  hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(22,55,103,0.14), transparent) !important;
    margin: 1.5rem 0 !important;
  }

  /* ---- Alerts ---- */
  [data-testid="stAlert"] {
    border-radius: 12px !important;
  }

  /* ---- Dataframe ---- */
  [data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: var(--shadow) !important;
  }

  /* ---- Expander ---- */
  [data-testid="stExpander"] {
    border-radius: 12px !important;
    border: 1px solid rgba(22,55,103,0.1) !important;
    background: rgba(255,255,255,0.85) !important;
    backdrop-filter: blur(8px) !important;
    overflow: hidden !important;
  }
  [data-testid="stExpander"] summary {
    font-weight: 500 !important;
    color: var(--navy) !important;
    font-family: 'Inter', sans-serif !important;
  }
  [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background: rgba(255,255,255,0.9) !important;
  }

  /* ---- Streamlit header toolbar (cacher le badge FREE) ---- */
  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  #MainMenu,
  header[data-testid="stHeader"] { visibility: hidden !important; }

  /* ---- Streamlit Markdown text contrast ---- */
  .stApp .stMarkdown > div > p {
    color: #374151 !important;
    font-size: 0.93rem !important;
    line-height: 1.6 !important;
  }
  .stApp .stMarkdown strong, .stApp .stMarkdown b { color: var(--navy) !important; }

  /* ---- Protection du header custom (mc-header) ---- */
  /* Toutes les regles !important du stMarkdown sont annulees a l'interieur du header */
  .mc-header, .mc-header p, .mc-header span, .mc-header h1,
  .mc-header div, .mc-header strong {
    color: inherit !important;
    font-family: 'Inter', sans-serif !important;
  }
  .mc-header p { color: rgba(255,255,255,0.88) !important; }
  .mc-header h1 { color: #fff !important; }

  /* ---- Fond de page main area (toujours visible) ---- */
  [data-testid="stAppViewContainer"] > section.main {
    background: transparent !important;
  }
  [data-testid="stAppViewContainer"] {
    background: linear-gradient(145deg, #eef2ff 0%, #faf0ff 55%, #eff8ff 100%) !important;
  }

  /* ---- Supprime le spinner "Made with Streamlit" ---- */
  .viewerBadge_container__r5tak,
  [data-testid="stStatusWidget"] { display: none !important; }
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
<div class="mc-header" style="
  background: linear-gradient(135deg, #051832 0%, #163767 45%, #0C78B4 100%);
  padding: 2.5rem 2.4rem;
  border-radius: 20px;
  margin-bottom: 1.8rem;
  position: relative;
  overflow: hidden;
  box-shadow: 0 12px 48px rgba(5,24,50,0.3);
">
  <!-- decorative blobs -->
  <div style="
    position: absolute; top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(255,67,184,0.22) 0%, transparent 70%);
    border-radius: 50%;
  "></div>
  <div style="
    position: absolute; bottom: -30px; left: 30%;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(12,120,180,0.28) 0%, transparent 70%);
    border-radius: 50%;
  "></div>

  <div style="position: relative; z-index: 1;">
    <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 0.6rem;">
      <span style="font-size: 2.2rem; filter: drop-shadow(0 2px 6px rgba(0,0,0,0.3));">&#127379;</span>
      <h1 style="
        color: #fff;
        margin: 0;
        font-size: 2.1rem;
        font-weight: 800;
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.04em;
      ">MixCraft <span style="
        background: linear-gradient(90deg, #FF43B8, #0C78B4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
      ">AI</span></h1>
    </div>
    <p style="
      color: rgba(255,255,255,0.88);
      margin: 0 0 0.3rem 0;
      font-size: 1rem;
      font-family: 'Inter', sans-serif;
      font-weight: 400;
      letter-spacing: 0.01em;
    ">Recommandation et creation de cocktails par analyse semantique</p>
    <div style="display: flex; align-items: center; gap: 8px; margin-top: 0.8rem; flex-wrap: wrap;">
      <span style="
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.2);
        color: rgba(255,255,255,0.75);
        border-radius: 99px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        letter-spacing: 0.03em;
      ">EFREI Paris</span>
      <span style="
        background: rgba(255,67,184,0.18);
        border: 1px solid rgba(255,67,184,0.3);
        color: rgba(255,255,255,0.8);
        border-radius: 99px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
      ">M1 Data Engineering & IA</span>
      <span style="
        color: rgba(255,255,255,0.45);
        font-size: 0.75rem;
        font-family: 'Inter', sans-serif;
      ">Adam Beloucif &amp; Emilien Morice</span>
    </div>
  </div>
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
                    <div class="recipe-card">
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
