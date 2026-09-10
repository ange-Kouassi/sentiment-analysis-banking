import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSentiment — Analyse de Sentiment Bancaire",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── DESIGN SYSTEM ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* Reset & base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background-color: #0D1117;
    color: #E6EDF3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #21262D;
}

[data-testid="stSidebar"] .stRadio label {
    color: #8B949E !important;
    font-size: 0.875rem;
    font-weight: 400;
    padding: 6px 0;
    transition: color 0.15s;
}

[data-testid="stSidebar"] .stRadio label:hover {
    color: #E6EDF3 !important;
}

/* Logo sidebar */
.sidebar-logo {
    font-family: 'DM Mono', monospace;
    font-size: 1.1rem;
    font-weight: 500;
    color: #58A6FF;
    letter-spacing: -0.02em;
    padding: 24px 0 32px 0;
    border-bottom: 1px solid #21262D;
    margin-bottom: 24px;
}

.sidebar-logo span {
    color: #3FB950;
}

/* Page header */
.page-header {
    padding: 32px 0 24px 0;
    border-bottom: 1px solid #21262D;
    margin-bottom: 32px;
}

.page-header h1 {
    font-size: 1.5rem;
    font-weight: 600;
    color: #E6EDF3;
    margin: 0 0 4px 0;
    letter-spacing: -0.03em;
}

.page-header p {
    font-size: 0.875rem;
    color: #8B949E;
    margin: 0;
}

/* KPI cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 32px;
}

.kpi-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 8px;
    padding: 20px 24px;
}

.kpi-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
}

.kpi-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.75rem;
    font-weight: 500;
    color: #E6EDF3;
    line-height: 1;
}

.kpi-delta {
    font-size: 0.75rem;
    color: #3FB950;
    margin-top: 6px;
}

/* Section title */
.section-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 16px;
}

/* Chart container */
.chart-container {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 8px;
    padding: 24px;
    margin-bottom: 16px;
}

/* Sentiment badge */
.badge-positive {
    display: inline-block;
    background: #1A3B2A;
    color: #3FB950;
    border: 1px solid #2D6A3F;
    border-radius: 4px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}

.badge-negative {
    display: inline-block;
    background: #3B1A1A;
    color: #F85149;
    border: 1px solid #6A2D2D;
    border-radius: 4px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}

.badge-neutral {
    display: inline-block;
    background: #1A2A3B;
    color: #58A6FF;
    border: 1px solid #2D4A6A;
    border-radius: 4px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}

/* Result block */
.result-block {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 8px;
    padding: 28px;
    margin-top: 24px;
}

.result-sentiment {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    margin-bottom: 8px;
}

.result-confidence {
    font-family: 'DM Mono', monospace;
    font-size: 0.875rem;
    color: #8B949E;
    margin-bottom: 24px;
}

/* Probability bar */
.prob-row {
    display: flex;
    align-items: center;
    margin-bottom: 12px;
    gap: 12px;
}

.prob-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: #8B949E;
    width: 70px;
    flex-shrink: 0;
}

.prob-bar-bg {
    flex: 1;
    background: #21262D;
    border-radius: 2px;
    height: 6px;
    overflow: hidden;
}

.prob-bar-fill-pos {
    height: 100%;
    background: #3FB950;
    border-radius: 2px;
    transition: width 0.4s ease;
}

.prob-bar-fill-neg {
    height: 100%;
    background: #F85149;
    border-radius: 2px;
}

.prob-bar-fill-neu {
    height: 100%;
    background: #58A6FF;
    border-radius: 2px;
}

.prob-value {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #E6EDF3;
    width: 48px;
    text-align: right;
    flex-shrink: 0;
}

/* Example card */
.example-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-left: 3px solid #58A6FF;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin-bottom: 12px;
    font-size: 0.875rem;
    color: #C9D1D9;
    line-height: 1.5;
}

/* Performance table */
.perf-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
}

.perf-table th {
    text-align: left;
    padding: 10px 16px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-bottom: 1px solid #21262D;
}

.perf-table td {
    padding: 12px 16px;
    border-bottom: 1px solid #161B22;
    color: #C9D1D9;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
}

.perf-table tr:last-child td {
    border-bottom: none;
    color: #E6EDF3;
    font-weight: 500;
}

.perf-table tr:last-child td:first-child {
    color: #3FB950;
}

/* Text area */
.stTextArea textarea {
    background-color: #161B22 !important;
    border: 1px solid #21262D !important;
    border-radius: 8px !important;
    color: #E6EDF3 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}

.stTextArea textarea:focus {
    border-color: #58A6FF !important;
    box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.1) !important;
}

/* Button */
.stButton button {
    background-color: #238636 !important;
    color: #FFFFFF !important;
    border: 1px solid #2EA043 !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
    transition: background-color 0.15s !important;
}

.stButton button:hover {
    background-color: #2EA043 !important;
}

/* Divider */
hr {
    border-color: #21262D !important;
    margin: 24px 0 !important;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── HELPERS ───────────────────────────────────────────────────────────────────
COLORS = {
    'positive': '#3FB950',
    'negative': '#F85149',
    'neutral' : '#58A6FF',
    'bg'      : '#0D1117',
    'surface' : '#161B22',
    'border'  : '#21262D',
    'text'    : '#E6EDF3',
    'muted'   : '#8B949E',
}

def make_fig():
    fig, ax = plt.subplots()
    fig.patch.set_facecolor(COLORS['surface'])
    ax.set_facecolor(COLORS['surface'])
    ax.tick_params(colors=COLORS['muted'], labelsize=10)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color(COLORS['border'])
    return fig, ax

@st.cache_data
def charger_donnees():
    path = 'data/data_cleaned.csv'
    if not os.path.exists(path):
        st.error("Dataset introuvable dans data/data_cleaned.csv")
        st.stop()
    return pd.read_csv(path)

@st.cache_resource
def charger_modele():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    return tokenizer, model

def predire(texte, tokenizer, model):
    inputs = tokenizer(
        texte, return_tensors='pt',
        max_length=512, truncation=True, padding=True
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs  = torch.softmax(logits, dim=1)[0]
    labels = {0: 'positive', 1: 'negative', 2: 'neutral'}
    pred   = torch.argmax(probs).item()
    return {
        'sentiment': labels[pred],
        'positive' : probs[0].item(),
        'negative' : probs[1].item(),
        'neutral'  : probs[2].item(),
    }

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">Fin<span>Sentiment</span></div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "Vue d'ensemble",
        "Analyser un texte",
        "Performance des modeles",
    ], label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem; color:#8B949E; line-height:1.6;">
    Modele : <span style="color:#58A6FF; font-family:'DM Mono'">ProsusAI/finbert</span><br>
    Dataset : <span style="color:#58A6FF; font-family:'DM Mono'">5 841 phrases</span><br>
    Accuracy : <span style="color:#3FB950; font-family:'DM Mono'">75.81%</span>
    </div>
    """, unsafe_allow_html=True)

# ─── LOAD DATA ─────────────────────────────────────────────────────────────────
df = charger_donnees()

counts   = df['Sentiment'].value_counts()
n_pos    = counts.get('positive', 0)
n_neg    = counts.get('negative', 0)
n_neu    = counts.get('neutral',  0)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — VUE D'ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Vue d'ensemble":

    st.markdown("""
    <div class="page-header">
        <h1>Vue d'ensemble</h1>
        <p>Distribution et caracteristiques du corpus financier</p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Total phrases</div>
            <div class="kpi-value">{len(df):,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Positives</div>
            <div class="kpi-value" style="color:#3FB950">{n_pos:,}</div>
            <div class="kpi-delta">{n_pos/len(df)*100:.1f}% du corpus</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Negatives</div>
            <div class="kpi-value" style="color:#F85149">{n_neg:,}</div>
            <div class="kpi-delta" style="color:#F85149">{n_neg/len(df)*100:.1f}% du corpus</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Neutres</div>
            <div class="kpi-value" style="color:#58A6FF">{n_neu:,}</div>
            <div class="kpi-delta" style="color:#58A6FF">{n_neu/len(df)*100:.1f}% du corpus</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-title">Repartition des sentiments</div>', unsafe_allow_html=True)
        fig, ax = make_fig()
        sentiments = ['negative', 'neutral', 'positive']
        vals       = [n_neg, n_neu, n_pos]
        bars       = ax.barh(
            sentiments, vals,
            color=[COLORS['negative'], COLORS['neutral'], COLORS['positive']],
            height=0.5, edgecolor='none'
        )
        ax.set_xlabel('Nombre de phrases', color=COLORS['muted'], fontsize=10)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
                f'{val:,}', va='center', color=COLORS['muted'],
                fontsize=10, fontfamily='monospace'
            )
        ax.set_xlim(0, max(vals) * 1.18)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Longueur moyenne par sentiment</div>', unsafe_allow_html=True)
        df['word_count'] = df['Sentence'].str.split().str.len()
        stats = df.groupby('Sentiment')['word_count'].mean().reindex(sentiments)
        fig, ax = make_fig()
        ax.bar(
            sentiments, stats.values,
            color=[COLORS['negative'], COLORS['neutral'], COLORS['positive']],
            width=0.5, edgecolor='none'
        )
        ax.set_ylabel('Mots en moyenne', color=COLORS['muted'], fontsize=10)
        for i, v in enumerate(stats.values):
            ax.text(i, v + 0.3, f'{v:.1f}', ha='center',
                    color=COLORS['muted'], fontsize=10, fontfamily='monospace')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.markdown('<div class="section-title" style="margin-top:8px">Exemples du corpus</div>', unsafe_allow_html=True)
    sentiment_choisi = st.selectbox(
        "", ['positive', 'negative', 'neutral'],
        label_visibility="collapsed",
        format_func=lambda x: f"Sentiment : {x.upper()}"
    )
    exemples = df[df['Sentiment'] == sentiment_choisi]['Sentence'].sample(3, random_state=42)
    for ex in exemples:
        st.markdown(f'<div class="example-card">{ex}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSER UN TEXTE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Analyser un texte":

    st.markdown("""
    <div class="page-header">
        <h1>Analyser un texte</h1>
        <p>Classification en temps reel avec FinBERT</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Chargement de FinBERT..."):
        tokenizer, model = charger_modele()

    texte = st.text_area(
        "Texte financier a analyser",
        placeholder="Ex: The company reported record profits driven by strong demand across all segments...",
        height=140,
        label_visibility="collapsed"
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        analyser = st.button("Analyser", use_container_width=True)

    if analyser:
        if not texte.strip():
            st.warning("Entre un texte pour commencer l'analyse.")
        else:
            with st.spinner("Analyse en cours..."):
                res = predire(texte, tokenizer, model)

            s   = res['sentiment']
            col = {'positive': COLORS['positive'],
                   'negative': COLORS['negative'],
                   'neutral' : COLORS['neutral']}[s]
            label = {'positive': 'POSITIF', 'negative': 'NEGATIF', 'neutral': 'NEUTRE'}[s]
            conf  = max(res['positive'], res['negative'], res['neutral'])

            st.markdown(f"""
            <div class="result-block">
                <div class="result-sentiment" style="color:{col}">{label}</div>
                <div class="result-confidence">Confiance : {conf:.1%}</div>

                <div class="prob-row">
                    <div class="prob-label">Positif</div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill-pos" style="width:{res['positive']*100:.1f}%"></div>
                    </div>
                    <div class="prob-value">{res['positive']:.1%}</div>
                </div>
                <div class="prob-row">
                    <div class="prob-label">Negatif</div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill-neg" style="width:{res['negative']*100:.1f}%"></div>
                    </div>
                    <div class="prob-value">{res['negative']:.1%}</div>
                </div>
                <div class="prob-row">
                    <div class="prob-label">Neutre</div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill-neu" style="width:{res['neutral']*100:.1f}%"></div>
                    </div>
                    <div class="prob-value">{res['neutral']:.1%}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Exemples a tester</div>', unsafe_allow_html=True)

    exemples_test = [
        "The bank reported a 23% increase in net profit, driven by strong retail lending growth.",
        "Operating losses widened significantly as credit defaults surged across the portfolio.",
        "The board approved the annual financial statements for the fiscal year ended December 31.",
    ]
    for ex in exemples_test:
        st.markdown(f'<div class="example-card">{ex}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PERFORMANCE DES MODELES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Performance des modeles":

    st.markdown("""
    <div class="page-header">
        <h1>Performance des modeles</h1>
        <p>Comparaison TF-IDF baseline vs FinBERT</p>
    </div>
    """, unsafe_allow_html=True)

    # Table
    st.markdown('<div class="section-title">Resultats par modele</div>', unsafe_allow_html=True)
    st.markdown("""
    <table class="perf-table">
        <thead>
            <tr>
                <th>Modele</th>
                <th>Accuracy</th>
                <th>F1 Negatif</th>
                <th>F1 Neutre</th>
                <th>F1 Positif</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>TF-IDF + LR</td><td>0.698</td><td>0.21</td><td>0.78</td><td>0.69</td></tr>
            <tr><td>TF-IDF + RF</td><td>0.643</td><td>0.16</td><td>0.73</td><td>0.69</td></tr>
            <tr><td>TF-IDF + SVM</td><td>0.699</td><td>0.18</td><td>0.78</td><td>0.71</td></tr>
            <tr><td>FinBERT</td><td>0.758</td><td>0.63</td><td>0.78</td><td>0.80</td></tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="section-title">F1-Score par classe</div>', unsafe_allow_html=True)
        fig, ax = make_fig()
        modeles = ['LR', 'RF', 'SVM', 'FinBERT']
        x       = np.arange(len(modeles))
        w       = 0.25

        ax.bar(x - w,   [0.21, 0.16, 0.18, 0.63], w, color=COLORS['negative'], label='Negatif', edgecolor='none')
        ax.bar(x,       [0.78, 0.73, 0.78, 0.78], w, color=COLORS['neutral'],  label='Neutre',  edgecolor='none')
        ax.bar(x + w,   [0.69, 0.69, 0.71, 0.80], w, color=COLORS['positive'], label='Positif', edgecolor='none')

        ax.set_xticks(x)
        ax.set_xticklabels(modeles, color=COLORS['muted'])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('F1-Score', color=COLORS['muted'], fontsize=10)
        ax.legend(
            facecolor=COLORS['surface'], edgecolor=COLORS['border'],
            labelcolor=COLORS['muted'], fontsize=9
        )
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Accuracy globale</div>', unsafe_allow_html=True)
        fig, ax = make_fig()
        accs    = [0.698, 0.643, 0.699, 0.758]
        colors  = [COLORS['muted']] * 3 + [COLORS['positive']]
        bars    = ax.bar(modeles, accs, color=colors, width=0.5, edgecolor='none')
        ax.set_ylim(0.55, 0.82)
        ax.set_ylabel('Accuracy', color=COLORS['muted'], fontsize=10)
        ax.axhline(y=0.758, color=COLORS['positive'], linestyle='--', linewidth=1, alpha=0.4)
        for bar, val in zip(bars, accs):
            ax.text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f'{val:.3f}', ha='center', color=COLORS['muted'],
                fontsize=10, fontfamily='monospace'
            )
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # Insight
    st.markdown("""
    <div style="background:#1A3B2A; border:1px solid #2D6A3F; border-radius:8px; padding:20px 24px; margin-top:8px;">
        <div style="font-size:0.75rem; font-weight:600; color:#3FB950; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;">
            Conclusion
        </div>
        <div style="font-size:0.875rem; color:#C9D1D9; line-height:1.6;">
            FinBERT depasse les modeles baseline sur toutes les classes.
            L'amelioration la plus significative concerne la classe <strong style="color:#F85149">negative</strong>
            : F1 de 0.18 (SVM) a 0.63 (+250%), grace a la comprehension
            contextuelle du modele Transformer.
        </div>
    </div>
    """, unsafe_allow_html=True)
