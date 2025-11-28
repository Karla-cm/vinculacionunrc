import streamlit as st
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import hashlib

# --- CARGAR DATOS ---
@st.cache_data
def load_data():
    try:
        vacantes = json.load(open('vacantes.json', 'r', encoding='utf-8'))
        cursos = json.load(open('cursos.json', 'r', encoding='utf-8'))
        egresados = pd.read_csv('egresados_data.csv')
    except FileNotFoundError:
        st.error("Archivos no encontrados.")
        vacantes = []
        cursos = []
        egresados = pd.DataFrame()
    return vacantes, cursos, egresados

VACANTES, CURSOS, EGRESADOS = load_data()

# Inicializar página actual
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'inicio'

# Función para hashear contraseña
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Contraseñas individuales por usuario (ID de registro: NombreApellido)
USER_PASSWORDS = {
    "SofíaCasas": hash_password("Sofia2024sc"),
    "DanielaEspinosa": hash_password("Daniela2023de"),
    "AndrésLópez": hash_password("Andres2022al"),
    "MarianaRojas": hash_password("Mariana2025mr"),
    "JavierSoto": hash_password("Javier2023js")
}

# Inicializar sesión
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'user_data' not in st.session_state:
    st.session_state.user_data = None

# --- FUNCIONES DE NLP ---
def normalizar_habilidad(habilidad):
    """Limpia la habilidad y maneja sinónimos básicos."""
    habilidad = habilidad.lower().strip()
    
    if 'estadistica' in habilidad:
        return 'estadística'
    if 'trabajo en equipo' in habilidad or 'equipo' in habilidad:
        return 'trabajo en equipo'
    if 'resolución' in habilidad and 'problemas' in habilidad:
        return 'resolución de problemas'
    
    terminos_clave = ['python', 'sql', 'excel', 'javascript', 'node.js', 'google ads', 'seo', 'docker', 'liderazgo']
    for termino in terminos_clave:
        if termino in habilidad:
            return termino
            
    return habilidad

def extraer_habilidades(cv_texto, lista_habilidades_conocidas):
    """Procesa el texto del CV y busca coincidencias."""
    habilidades_encontradas = set()
    habilidades_normalizadas = [normalizar_habilidad(h) for h in lista_habilidades_conocidas]
    cv_texto_limpio = normalizar_habilidad(cv_texto)
    
    for habilidad in habilidades_normalizadas:
        if habilidad in cv_texto_limpio:
            habilidades_encontradas.add(habilidad)
            
    return habilidades_encontradas

def calcular_similitud_tfidf(cv_texto, vacantes):
    """Calcula la similitud coseno."""
    if not vacantes:
        return {}
    
    documentos = [cv_texto] + [v.get('descripcion', '') for v in vacantes]
    
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(documentos)
    
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])
    scores = cosine_sim[0]
    
    tfidf_scores = {}
    for i, score in enumerate(scores):
        vacante_id = vacantes[i].get('id', i)
        tfidf_scores[vacante_id] = float(score)
        
    return tfidf_scores

def perform_matching(cv_texto):
    """Realiza el matching entre CV y vacantes."""
    if not cv_texto or not VACANTES:
        return []

    resultados = []
    
    # Extrae todas las habilidades
    todas_habilidades = set()
    for v in VACANTES:
        todas_habilidades.update(v.get('requisitos_tecnicos', []))
        todas_habilidades.update(v.get('requisitos_blandos', []))

    habilidades_cv = extraer_habilidades(cv_texto, todas_habilidades)
    tfidf_scores = calcular_similitud_tfidf(cv_texto, VACANTES)

    for vacante in VACANTES:
        req_tec = set(normalizar_habilidad(h) for h in vacante.get('requisitos_tecnicos', []))
        req_blando = set(normalizar_habilidad(h) for h in vacante.get('requisitos_blandos', []))
        req_totales = req_tec.union(req_blando)
        
        habilidades_cumplidas = habilidades_cv.intersection(req_totales)
        habilidades_faltantes = req_totales - habilidades_cv

        # Score final
        total_req = len(req_totales)
        score_cumplimiento = len(habilidades_cumplidas) / total_req if total_req else 0
        score_relevancia = tfidf_scores.get(vacante.get('id', 0), 0)
        puntaje_final = (score_cumplimiento * 0.6) + (score_relevancia * 0.4)
        
        # Cursos recomendados
        cursos_recomendados = [
            curso for curso in CURSOS 
            if normalizar_habilidad(curso.get('habilidad', '')) in habilidades_faltantes
        ]

        resultados.append({
            "vacante": vacante,
            "puntaje_match": round(puntaje_final * 100, 2),
            "habilidades_cumplidas": sorted(list(habilidades_cumplidas)),
            "habilidades_faltantes": sorted(list(habilidades_faltantes)),
            "cursos_recomendados": cursos_recomendados
        })

    return sorted(resultados, key=lambda x: x['puntaje_match'], reverse=True)

# --- UI DE STREAMLIT ---
st.set_page_config(page_title="CogniLink UNRC", page_icon="💼", layout="wide")

st.markdown("""
<style>
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #ffffff;
}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
}

/* Navbar */
.navbar {
    background: rgba(255, 255, 255, 0.98);
    backdrop-filter: blur(10px);
    padding: 1rem 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 20px rgba(15, 23, 42, 0.06);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.navbar-brand {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1e40af;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Hero Section */
.hero-section {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    color: white;
    padding: 6rem 3rem;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 3rem;
    box-shadow: 0 25px 50px rgba(15, 23, 42, 0.25);
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.1);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.hero-section::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.1) 0%, transparent 70%);
    animation: pulse 4s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 0.3; }
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    margin-bottom: 1.5rem;
    text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    position: relative;
    z-index: 1;
    letter-spacing: -1px;
    text-align: center;
    width: 100%;
}

.hero-subtitle {
    font-size: 1.3rem;
    opacity: 1;
    margin-bottom: 0;
    position: relative;
    z-index: 1;
    max-width: 700px;
    margin-left: auto;
    margin-right: auto;
    color: #e2e8f0;
    text-align: center;
    line-height: 1.6;
}

.hero-stats {
    display: flex;
    justify-content: center;
    gap: 3rem;
    margin-top: 2rem;
    position: relative;
    z-index: 1;
}

.hero-stat {
    text-align: center;
}

.hero-stat-number {
    font-size: 2.5rem;
    font-weight: 700;
    display: block;
    color: #60a5fa;
}

.hero-stat-label {
    font-size: 0.95rem;
    opacity: 1;
    color: #e2e8f0;
}

/* Features Section */
.features-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 2rem;
    margin: 3rem 0;
}

.feature-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
    text-align: center;
    border: 1px solid #475569;
}

.feature-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    border-color: #60a5fa;
}

.feature-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}

.feature-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.8rem;
}

.feature-desc {
    color: #e2e8f0;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Section Titles */
.section-header {
    text-align: center;
    margin: 3rem 0 2rem 0;
}

.section-header h2 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.5rem;
}

.section-header p {
    color: #cbd5e1;
    font-size: 1.1rem;
}

/* Vacancy Cards */
.vacancy-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    padding: 1.8rem;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    margin-bottom: 1.5rem;
    border-left: 4px solid #60a5fa;
    transition: all 0.3s ease;
    border: 1px solid #475569;
}

.vacancy-card:hover {
    transform: translateX(4px);
    box-shadow: 0 8px 30px rgba(15, 23, 42, 0.1);
}

.vacancy-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.3rem;
}

.vacancy-company {
    color: #60a5fa;
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 0.8rem;
}

.vacancy-desc {
    color: #e2e8f0;
    font-size: 0.95rem;
    line-height: 1.5;
    margin-bottom: 1rem;
}

.vacancy-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.vacancy-tag {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: white;
    padding: 0.4rem 0.8rem;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 500;
}

/* Course Cards */
.course-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    text-align: center;
    transition: all 0.3s ease;
    border: 1px solid #475569;
}

.course-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
    border-color: #34d399;
}

.course-icon {
    font-size: 2.5rem;
    margin-bottom: 0.8rem;
}

.course-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.3rem;
}

.course-provider {
    color: #34d399;
    font-weight: 700;
    font-size: 0.9rem;
}

.course-skill {
    background: #065f46;
    color: #ffffff;
    padding: 0.3rem 0.8rem;
    border-radius: 6px;
    font-size: 0.8rem;
    margin-top: 0.8rem;
    display: inline-block;
    font-weight: 600;
}

/* CTA Section */
.cta-section {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: white;
    padding: 3rem;
    border-radius: 16px;
    text-align: center;
    margin: 3rem 0;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.cta-title {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 1rem;
}

.cta-subtitle {
    font-size: 1.1rem;
    opacity: 1;
    margin-bottom: 2rem;
    color: #cbd5e1;
}

/* Testimonials */
.testimonial-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    position: relative;
    border: 1px solid #475569;
}

.testimonial-quote {
    font-size: 3rem;
    color: #3b82f6;
    position: absolute;
    top: 1rem;
    left: 1.5rem;
    opacity: 0.2;
}

.testimonial-text {
    font-style: italic;
    color: #e2e8f0;
    line-height: 1.7;
    margin-bottom: 1.5rem;
    padding-top: 1rem;
}

.testimonial-author {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.testimonial-avatar {
    width: 50px;
    height: 50px;
    background: linear-gradient(135deg, #0f172a 0%, #334155 100%);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 700;
    font-size: 1.2rem;
}

.testimonial-info h4 {
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.2rem;
}

.testimonial-info p {
    color: #60a5fa;
    font-size: 0.9rem;
    font-weight: 600;
}

/* Footer */
.footer {
    background: #0f172a;
    color: white;
    padding: 2rem;
    border-radius: 12px;
    margin-top: 3rem;
    text-align: center;
}

.footer-links {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-bottom: 1.5rem;
}

.footer-link {
    color: rgba(255, 255, 255, 0.7);
    text-decoration: none;
    transition: color 0.3s;
}

.footer-link:hover {
    color: #3b82f6;
}

.footer-copy {
    color: rgba(255, 255, 255, 0.5);
    font-size: 0.9rem;
}

/* Header Principal */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: white;
    padding: 3rem 2rem;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px rgba(15, 23, 42, 0.2);
    animation: slideDown 0.5s ease-out;
}

.main-header h1 {
    font-size: 2.8rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    letter-spacing: -0.5px;
    color: #ffffff;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.main-header p {
    font-size: 1.1rem;
    opacity: 1;
    font-weight: 400;
    color: #e2e8f0;
}

/* Login Container */
.login-container {
    max-width: 420px;
    margin: 4rem auto;
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    padding: 2.5rem;
    border-radius: 12px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
    animation: fadeIn 0.6s ease-out;
    border: 1px solid #475569;
}

.login-container h2 {
    color: #ffffff;
    font-size: 1.8rem;
    margin-bottom: 0.5rem;
    text-align: center;
    font-weight: 700;
}

.login-container p {
    color: #e2e8f0;
    text-align: center;
    margin-bottom: 2rem;
    font-size: 0.95rem;
    font-weight: 600;
}

.login-form {
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
}

/* Profile Card */
.profile-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: white;
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px rgba(15, 23, 42, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.profile-card h2 {
    font-size: 2rem;
    margin-bottom: 1rem;
    font-weight: 700;
    color: #ffffff;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
}

.profile-info {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    font-size: 0.95rem;
    color: rgba(255, 255, 255, 0.9);
}

.profile-info-item {
    background: rgba(255, 255, 255, 0.1);
    padding: 0.6rem 1.2rem;
    border-radius: 8px;
    backdrop-filter: blur(10px);
    font-weight: 500;
    border: 1px solid rgba(255, 255, 255, 0.15);
    color: #ffffff;
}

/* Result Card */
.result-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border-radius: 12px;
    padding: 1.8rem;
    margin: 1.5rem 0;
    border-left: 4px solid #60a5fa;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
    border: 1px solid #475569;
}

.result-card:hover {
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
    transform: translateY(-2px);
}

.result-card h3 {
    color: #ffffff;
    margin-bottom: 0.5rem;
    font-size: 1.3rem;
    font-weight: 700;
}

.result-card .empresa {
    color: #60a5fa;
    font-weight: 700;
    margin-bottom: 0.8rem;
    font-size: 1rem;
}

/* Skills */
.skill-match {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    color: white;
    border-radius: 6px;
    padding: 0.5rem 1rem;
    margin: 0.3rem;
    display: inline-block;
    font-size: 0.85rem;
    font-weight: 500;
    box-shadow: 0 2px 8px rgba(16, 185, 129, 0.2);
}

.skill-missing {
    background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
    color: white;
    border-radius: 6px;
    padding: 0.5rem 1rem;
    margin: 0.3rem;
    display: inline-block;
    font-size: 0.85rem;
    font-weight: 500;
    box-shadow: 0 2px 8px rgba(239, 68, 68, 0.2);
}

.skill-neutral {
    background: #475569;
    color: #ffffff;
    border-radius: 6px;
    padding: 0.5rem 1rem;
    margin: 0.3rem;
    display: inline-block;
    font-size: 0.85rem;
    font-weight: 600;
}

/* Match Score */
.match-score {
    font-size: 2.5rem;
    font-weight: 700;
    text-align: center;
}

.match-score.excellent { color: #059669; }
.match-score.good { color: #d97706; }
.match-score.low { color: #dc2626; }

/* Sections */
.section-title {
    color: #ffffff;
    font-size: 1.4rem;
    font-weight: 700;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.8rem;
    border-bottom: 3px solid #60a5fa;
    display: inline-block;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: #ffffff !important;
    border: none;
    border-radius: 8px;
    padding: 0.8rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    cursor: pointer;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.5);
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
}

/* Input Fields */
.stSelectbox label, .stTextInput label, .stPasswordInput label {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
}

.stSelectbox > div > div > select,
.stTextInput input,
.stPasswordInput input {
    border-radius: 8px !important;
    border: 2px solid #475569 !important;
    padding: 0.8rem !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    background: #0f172a !important;
    color: #ffffff !important;
    font-weight: 500 !important;
}

.stSelectbox > div > div > select:focus,
.stTextInput input:focus,
.stPasswordInput input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
}

/* Info Boxes */
.stInfo {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    color: #1e40af !important;
}

/* Animations */
@keyframes slideDown {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

/* Divider */
hr {
    margin: 2rem 0;
    border: none;
    border-top: 1px solid #e2e8f0;
}

/* Text Colors */
p, span, div, h4, h5, h6 {
    color: #ffffff !important;
}

h1, h2, h3 {
    color: #ffffff !important;
}

/* Metrics */
.stMetric {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    padding: 1rem;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
}

.stMetric label, .stMetric div {
    color: #ffffff !important;
}

/* How it works */
.step-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    text-align: center;
    position: relative;
    border: 1px solid #475569;
}

.step-number {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: white;
    width: 50px;
    height: 50px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0 auto 1rem auto;
}

.step-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.5rem;
}

.step-desc {
    color: #e2e8f0;
    font-size: 0.95rem;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

def show_header():
    # Navbar con navegación - diferente si está logueado o no
    if st.session_state.logged_in:
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([2, 1, 1, 1, 1, 1, 1, 1])
    else:
        col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 1])
    
    with col1:
        st.markdown("<div style='font-size: 1.5rem; font-weight: 700; color: #60a5fa;'>CogniLink UNRC</div>", unsafe_allow_html=True)
    
    with col2:
        if st.button("Inicio", use_container_width=True):
            st.session_state.current_page = 'inicio'
            st.rerun()
    
    if st.session_state.logged_in:
        with col3:
            if st.button("Nuestra Historia", use_container_width=True):
                st.session_state.current_page = 'nosotros'
                st.rerun()
        
        with col4:
            if st.button("Vacantes", use_container_width=True):
                st.session_state.current_page = 'vacantes'
                st.rerun()
        
        with col5:
            if st.button("Cursos", use_container_width=True):
                st.session_state.current_page = 'cursos'
                st.rerun()
        
        with col6:
            if st.button("Testimonios", use_container_width=True):
                st.session_state.current_page = 'testimonios'
                st.rerun()
        
        with col7:
            if st.button("Aviso de Privacidad", use_container_width=True):
                st.session_state.current_page = 'privacidad'
                st.rerun()
        
        with col8:
            if st.button("Salir", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.user_id = None
                st.session_state.user_data = None
                st.session_state.current_page = 'inicio'
                st.rerun()
    else:
        with col3:
            if st.button("Nuestra Historia", use_container_width=True):
                st.session_state.current_page = 'nosotros'
                st.rerun()
        
        with col4:
            if st.button("Acceder", use_container_width=True):
                st.session_state.current_page = 'login'
                st.rerun()
        
        with col5:
            if st.button("Testimonios", use_container_width=True):
                st.session_state.current_page = 'testimonios'
                st.rerun()
        
        with col6:
            if st.button("Aviso de Privacidad", use_container_width=True):
                st.session_state.current_page = 'privacidad'
                st.rerun()
    
    st.markdown("---")

# --- PÁGINA DE INICIO ---
def show_home():
    # Hero Section
    st.markdown(f"""
    <div class='hero-section'>
        <h1 class='hero-title'>🎓 CogniLink UNRC</h1>
        <p class='hero-subtitle'>
            Conectamos el talento de egresados de la Universidad Nacional Rosario Castellanos 
            con las mejores oportunidades laborales del mercado
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Section
    st.markdown("""
    <div class='section-header'>
        <h2>¿Qué ofrecemos?</h2>
        <p>Herramientas inteligentes para impulsar tu carrera profesional</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>🤖</div>
            <div class='feature-title'>Matching Inteligente</div>
            <div class='feature-desc'>
                Algoritmos de NLP y Machine Learning que analizan tu perfil 
                y encuentran las vacantes más compatibles con tus habilidades.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>📊</div>
            <div class='feature-title'>Análisis de Habilidades</div>
            <div class='feature-desc'>
                Evaluación detallada de tus competencias técnicas y blandas 
                comparadas con los requisitos del mercado laboral.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>🎯</div>
            <div class='feature-title'>Recomendaciones Personalizadas</div>
            <div class='feature-desc'>
                Cursos y capacitaciones sugeridos para cerrar las brechas 
                de habilidades y aumentar tu empleabilidad.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>📚</div>
            <div class='feature-title'>Cursos Recomendados</div>
            <div class='feature-desc'>
                Accede a una amplia biblioteca de cursos especializados para 
                desarrollar las habilidades más demandadas del mercado laboral actual.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # How it works
    st.markdown("""
    <div class='section-header'>
        <h2>¿Cómo funciona?</h2>
        <p>Tres simples pasos para encontrar tu trabajo ideal</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='step-card'>
            <div class='step-number'>1</div>
            <div class='step-title'>Regístrate</div>
            <div class='step-desc'>
                Crea tu perfil con tu información académica, 
                experiencia laboral y habilidades.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='step-card'>
            <div class='step-number'>2</div>
            <div class='step-title'>Analiza</div>
            <div class='step-desc'>
                Nuestro sistema evalúa tu perfil y lo compara 
                con las vacantes disponibles.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='step-card'>
            <div class='step-number'>3</div>
            <div class='step-title'>Conecta</div>
            <div class='step-desc'>
                Recibe las mejores oportunidades y cursos 
                recomendados para tu perfil.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # CTA Section
    st.markdown("""
    <div class='cta-section'>
        <div class='cta-title'>🌟 ¿Listo para dar el siguiente paso?</div>
        <div class='cta-subtitle'>
            Únete a la comunidad de egresados de UNRC y descubre oportunidades 
            que se adaptan perfectamente a tu perfil profesional.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class='footer'>
        <div class='footer-links'>
            <span class='footer-link'>📧 contacto@cognilink.unrc.edu.ar</span>
            <span class='footer-link'>📍 UNRC, Calle Soledad San Bernabé</span>
            <span class='footer-link'>📱 +54 358 467-6200</span>
        </div>
        <div class='footer-copy'>
            © 2025 CogniLink UNRC - Sistema de Vinculación Laboral para Egresados<br>
            Universidad Nacional Rosario Castellanos
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- PÁGINA DE VACANTES ---
def show_vacantes_page():
    st.markdown("""
    <div class='section-header'>
        <h2>💼 Vacantes Disponibles</h2>
        <p>Explora todas las oportunidades laborales activas</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filtros
    col1, col2 = st.columns(2)
    with col1:
        empresas = list(set([v.get('empresa', '') for v in VACANTES]))
        empresa_filter = st.selectbox("🏢 Filtrar por empresa", ["Todas"] + empresas)
    
    with col2:
        todas_skills = set()
        for v in VACANTES:
            todas_skills.update(v.get('requisitos_tecnicos', []))
        skill_filter = st.selectbox("🛠️ Filtrar por habilidad", ["Todas"] + sorted(list(todas_skills)))
    
    st.markdown("---")
    
    for vacante in VACANTES:
        # Aplicar filtros
        if empresa_filter != "Todas" and vacante.get('empresa') != empresa_filter:
            continue
        if skill_filter != "Todas" and skill_filter not in vacante.get('requisitos_tecnicos', []):
            continue
        
        tags_tec = " ".join([f"<span class='vacancy-tag'>{tag}</span>" for tag in vacante.get('requisitos_tecnicos', [])])
        tags_soft = " ".join([f"<span class='skill-match'>{tag}</span>" for tag in vacante.get('requisitos_blandos', [])])
        
        st.markdown(f"""
        <div class='vacancy-card'>
            <div class='vacancy-title'>{vacante.get('titulo', 'Vacante')}</div>
            <div class='vacancy-company'>🏢 {vacante.get('empresa', 'Empresa')}</div>
            <div class='vacancy-desc'>{vacante.get('descripcion', '')}</div>
            <div style='margin-top: 1rem;'>
                <strong style='color: #1f2937;'>Requisitos Técnicos:</strong><br>
                {tags_tec}
            </div>
            <div style='margin-top: 0.8rem;'>
                <strong style='color: #1f2937;'>Habilidades Blandas:</strong><br>
                {tags_soft}
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- PÁGINA DE CURSOS ---
def show_cursos_page():
    st.markdown("""
    <div class='section-header'>
        <h2>📚 Cursos Recomendados</h2>
        <p>Desarrolla las habilidades más demandadas del mercado</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mostrar cursos en grid
    cols = st.columns(3)
    
    for idx, curso in enumerate(CURSOS):
        with cols[idx % 3]:
            icon = "💻" if curso.get('habilidad') in ['Python', 'SQL', 'Excel'] else "📖"
            st.markdown(f"""
            <div class='course-card'>
                <div class='course-icon'>{icon}</div>
                <div class='course-title'>{curso.get('titulo_curso', 'Curso')}</div>
                <div class='course-provider'>🎓 {curso.get('proveedor', 'Proveedor')}</div>
                <div class='course-skill'>{curso.get('habilidad', 'Habilidad')}</div>
            </div>
            """, unsafe_allow_html=True)

# --- PÁGINA DE AVISO DE PRIVACIDAD ---
def show_privacidad_page():
    st.markdown("""
    <div style='max-width: 900px; margin: 0 auto;'>
        <!-- Contenedor único estilo corporativo -->
        <div style='background: #ffffff; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                    overflow: hidden; margin-top: 1rem;'>
            
            <!-- Header -->
            <div style='background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); 
                        padding: 1.5rem 2rem; text-align: center;'>
                <h1 style='color: #ffffff; font-size: 1.5rem; font-weight: 600; margin: 0;'>
                    AVISO DE PRIVACIDAD
                </h1>
                <p style='color: rgba(255,255,255,0.8); font-size: 0.85rem; margin: 0.5rem 0 0 0;'>
                    CogniLink UNRC | Última actualización: 28 de noviembre de 2025
                </p>
            </div>
            
            <!-- Contenido -->
            <div style='padding: 2rem; color: #333333; font-size: 0.9rem; line-height: 1.7;'>
                
                <p style='margin-bottom: 1.5rem;'>
                    <strong>CogniLink UNRC</strong>, con domicilio en Calle Soledad San Bernabé, UNRC, 
                    Universidad Nacional Rosario Castellanos, es responsable del tratamiento de sus datos 
                    personales de conformidad con la normativa vigente en materia de protección de datos 
                    y los estándares ISO/IEC 27001.
                </p>
                
                <p style='margin-bottom: 1rem;'><strong>Datos que recopilamos:</strong></p>
                <p style='margin-bottom: 1.5rem; padding-left: 1rem;'>
                    Nombre, identificador de egresado, año de egreso, competencias técnicas y blandas, 
                    experiencia laboral, resumen curricular y credenciales de acceso protegidas mediante 
                    cifrado SHA-256.
                </p>
                
                <p style='margin-bottom: 1rem;'><strong>Finalidades del tratamiento:</strong></p>
                <p style='margin-bottom: 1.5rem; padding-left: 1rem;'>
                    Sus datos serán utilizados para: (i) realizar el matching inteligente entre su perfil 
                    y vacantes laborales mediante algoritmos de NLP; (ii) recomendar cursos de capacitación 
                    personalizados; (iii) facilitar la vinculación con empresas; (iv) enviar comunicaciones 
                    sobre oportunidades relevantes; y (v) mejorar nuestros servicios.
                </p>
                
                <p style='margin-bottom: 1rem;'><strong>Transferencia de datos:</strong></p>
                <p style='margin-bottom: 1.5rem; padding-left: 1rem;'>
                    Sus datos podrán compartirse con empresas registradas en la plataforma para fines de 
                    reclutamiento, proveedores de capacitación asociados, y autoridades cuando sea legalmente 
                    requerido. No vendemos ni comercializamos sus datos personales.
                </p>
                
                <p style='margin-bottom: 1rem;'><strong>Derechos ARCO:</strong></p>
                <p style='margin-bottom: 1.5rem; padding-left: 1rem;'>
                    Usted tiene derecho a Acceder, Rectificar, Cancelar u Oponerse al tratamiento de sus 
                    datos personales. Para ejercer estos derechos, envíe su solicitud a 
                    <strong>contacto@cognilink.unrc.edu.ar</strong> indicando su nombre completo y el 
                    derecho que desea ejercer.
                </p>
                
                <p style='margin-bottom: 1rem;'><strong>Medidas de seguridad:</strong></p>
                <p style='margin-bottom: 1.5rem; padding-left: 1rem;'>
                    Implementamos medidas técnicas y organizativas alineadas con ISO/IEC 27001, incluyendo 
                    cifrado de contraseñas, control de acceso, protección contra código malicioso y 
                    transmisión segura mediante TLS.
                </p>
                
                <p style='margin-bottom: 1rem;'><strong>Cookies:</strong></p>
                <p style='margin-bottom: 1.5rem; padding-left: 1rem;'>
                    Utilizamos cookies de sesión para mantener su acceso seguro. Estas se eliminan 
                    automáticamente al cerrar el navegador.
                </p>
                
                <p style='margin-bottom: 1rem;'><strong>Modificaciones:</strong></p>
                <p style='margin-bottom: 1.5rem; padding-left: 1rem;'>
                    Nos reservamos el derecho de modificar este aviso. Los cambios serán notificados 
                    a través de la plataforma.
                </p>
                
                <!-- Línea divisoria -->
                <hr style='border: none; border-top: 1px solid #e0e0e0; margin: 1.5rem 0;'>
                
                <p style='font-size: 0.85rem; color: #666666; text-align: center; margin-bottom: 0;'>
                    Al utilizar CogniLink UNRC, usted acepta los términos de este Aviso de Privacidad.<br>
                    <strong>Contacto:</strong> contacto@cognilink.unrc.edu.ar | Tel: +54 358 467-6200
                </p>
                
            </div>
            
            <!-- Footer del documento -->
            <div style='background: #f5f5f5; padding: 1rem 2rem; text-align: center; 
                        border-top: 1px solid #e0e0e0;'>
                <p style='color: #888888; font-size: 0.75rem; margin: 0;'>
                    © 2025 CogniLink UNRC - Universidad Nacional Rosario Castellanos | 
                    Documento conforme a ISO/IEC 27001:2022
                </p>
            </div>
            
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Botón para volver
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("← Volver al Inicio", use_container_width=True):
            st.session_state.current_page = 'inicio'
            st.rerun()

# --- PÁGINA TESTIMONIOS ---
def show_testimonios_page():
    st.markdown("""
    <div class='hero-section' style='padding: 3rem;'>
        <h1 class='hero-title' style='font-size: 2.5rem;'>💬 Testimonios</h1>
        <p class='hero-subtitle'>
            Historias de éxito de la comunidad UNRC
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='section-header'>
        <h2>🌟 Lo que dicen nuestros egresados</h2>
        <p>Experiencias reales de quienes ya encontraron su camino profesional</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='testimonial-card'>
            <div class='testimonial-quote'>"</div>
            <div class='testimonial-text'>
                Gracias a CogniLink encontré una posición que se ajustaba perfectamente 
                a mis habilidades en Python y Machine Learning. El sistema de matching 
                es increíblemente preciso.
            </div>
            <div class='testimonial-author'>
                <div class='testimonial-avatar'>AL</div>
                <div class='testimonial-info'>
                    <h4>Andrés López</h4>
                    <p>Ingeniero de ML - Egresado 2022</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='testimonial-card'>
            <div class='testimonial-quote'>"</div>
            <div class='testimonial-text'>
                Las recomendaciones de cursos me ayudaron a desarrollar las habilidades 
                que me faltaban. En menos de un mes conseguí mi primer trabajo como 
                consultora de datos.
            </div>
            <div class='testimonial-author'>
                <div class='testimonial-avatar'>DE</div>
                <div class='testimonial-info'>
                    <h4>Daniela Espinosa</h4>
                    <p>Analista de Datos - Egresada 2023</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='testimonial-card'>
            <div class='testimonial-quote'>"</div>
            <div class='testimonial-text'>
                El análisis de habilidades me mostró exactamente qué necesitaba aprender. 
                Los cursos recomendados fueron muy útiles para mi desarrollo profesional.
            </div>
            <div class='testimonial-author'>
                <div class='testimonial-avatar'>SC</div>
                <div class='testimonial-info'>
                    <h4>Sofía Casas</h4>
                    <p>Científica de Datos Jr - Egresada 2024</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='testimonial-card'>
            <div class='testimonial-quote'>"</div>
            <div class='testimonial-text'>
                CogniLink me ayudó a conectar con empresas que valoraban mi experiencia 
                en consultoría de datos. El proceso fue muy sencillo y efectivo.
            </div>
            <div class='testimonial-author'>
                <div class='testimonial-avatar'>JS</div>
                <div class='testimonial-info'>
                    <h4>Javier Soto</h4>
                    <p>Consultor de Datos - Egresado 2023</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- PÁGINA NOSOTROS ---
def show_about_page():
    # Historia de Origen y Nuestra Identidad unificados
    st.markdown("""
    <div class='hero-section' style='padding: 3rem;'>
        <h1 class='hero-title' style='font-size: 2.2rem;'>CogniLink: Nuestra Historia e Identidad</h1>
        <p class='hero-subtitle'>
            Conectando Talento con Inteligencia (UNRC - Universidad Nacional Rosario Castellanos)
        </p>
        <div style='text-align: left; max-width: 900px; margin: 2rem auto 0 auto; padding: 0 1rem;'>
            <p style='color: #e2e8f0; line-height: 1.8; margin-bottom: 1.5rem;'>
                La Universidad Nacional Rosario Castellanos (UNRC), como institucion innovadora y orientada al futuro, 
                rapidamente se posiciono como un centro vital para la formacion de profesionales especializados, 
                en particular en areas como la Ciencia de Datos para Negocios. Sin embargo, con un rapido crecimiento 
                y un enfoque vanguardista, surgio un desafio: la necesidad de un sistema de vinculacion laboral 
                que estuviera a la altura de su modelo educativo.
            </p>
            <p style='color: #e2e8f0; line-height: 1.8; margin-bottom: 1.5rem;'>
                Los metodos tradicionales de networking y ferias de empleo eran insuficientes para un cuerpo estudiantil 
                y de egresados que domina la analitica avanzada. Se requeria una solucion que hablara el mismo idioma: 
                el lenguaje de los datos. El proceso de matching entre el talento especializado de la UNRC y las vacantes 
                complejas del sector productivo era lento, manual y carecia de la precision que solo la IA puede ofrecer.
            </p>
            <h3 style='color: #60a5fa; margin-bottom: 1rem;'>El Nacimiento de la Solucion Inteligente</h3>
            <p style='color: #e2e8f0; line-height: 1.8;'>
                En 2024, en el octavo semestre de la carrera de Ciencia de Datos para Negocios, 
                <strong>Daniela Espinosa</strong> y <strong>Sofia Casas</strong> vieron la oportunidad de aplicar 
                sus conocimientos para transformar esta deficiencia en una ventaja competitiva para su universidad 
                y sus compañeros.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Misión, Visión y Tecnología en 3 columnas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>🎯</div>
            <div class='feature-title'>Nuestra Misión</div>
            <div class='feature-desc'>
                Facilitar la inserción laboral de los egresados de la UNRC mediante 
                tecnologías de inteligencia artificial que conectan talento con 
                oportunidades de manera eficiente y personalizada.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>👁️</div>
            <div class='feature-title'>Nuestra Visión</div>
            <div class='feature-desc'>
                Ser el puente principal entre la academia y el mundo laboral, 
                reduciendo la brecha entre la formación universitaria y las 
                necesidades del mercado.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>🔬</div>
            <div class='feature-title'>Tecnología</div>
            <div class='feature-desc'>
                Utilizamos algoritmos de NLP (Procesamiento de Lenguaje Natural) 
                y técnicas de Machine Learning como TF-IDF y similitud coseno 
                para analizar perfiles y vacantes.
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Header del login
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <div style='font-size: 4rem; margin-bottom: 1rem;'>🔐</div>
            <h1 style='color: #ffffff; font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;'>
                Bienvenido a CogniLink
            </h1>
            <p style='color: #94a3b8; font-size: 1rem;'>
                Accede a tu cuenta para explorar oportunidades personalizadas
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form", clear_on_submit=False):
            # Campo de texto para ingresar usuario manualmente
            user_input = st.text_input(
                "Usuario",
                placeholder="Ingresa tu usuario"
            )
            
            password = st.text_input(
                "Contraseña",
                type="password",
                placeholder="Ingresa tu contraseña"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            submitted = st.form_submit_button(
                "Iniciar Sesión",
                use_container_width=True,
                type="primary"
            )
            
            if submitted:
                if not user_input or not password:
                    st.error("Por favor completa todos los campos")
                else:
                    user_id_registro = user_input.strip()
                    
                    # Verificar contraseña individual
                    if user_id_registro in USER_PASSWORDS:
                        if hash_password(password) != USER_PASSWORDS[user_id_registro]:
                            st.error("Contraseña incorrecta")
                        else:
                            # Buscar usuario por nombre (reconstruir desde ID de registro)
                            user_found = None
                            for _, row in EGRESADOS.iterrows():
                                nombre_sin_espacios = row['Nombre'].replace(" ", "")
                                if nombre_sin_espacios == user_id_registro:
                                    user_found = row
                                    break
                            
                            if user_found is not None:
                                st.session_state.logged_in = True
                                st.session_state.user_id = user_id_registro
                                st.session_state.user_data = user_found.to_dict()
                                st.success("Sesión iniciada correctamente")
                                st.rerun()
                            else:
                                st.error("Usuario no encontrado en la base de datos")
                    else:
                        st.error("Usuario no registrado")
        
        # Botón para volver
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Volver al Inicio", use_container_width=True):
            st.session_state.current_page = 'inicio'
            st.rerun()

def show_profile():
    user = st.session_state.user_data
    
    st.markdown(f"""
    <div class='profile-card'>
        <h2>👤 {user['Nombre']}</h2>
        <div class='profile-info'>
            <div class='profile-info-item'><b>ID:</b> {user['ID_Egresado']}</div>
            <div class='profile-info-item'><b>Egreso:</b> {user['Anio_Egreso']}</div>
            <div class='profile-info-item'><b>Experiencia:</b> {user['Experiencia_Anios']} años</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='section-title'>🎯 Rol Deseado</div>", unsafe_allow_html=True)
        st.markdown(f"**{user['Rol_Deseado']}**")
        
        st.markdown("<div class='section-title'>🛠️ Hard Skills</div>", unsafe_allow_html=True)
        hard_skills = [s.strip() for s in str(user['Hard_Skills']).split(',')]
        skills_html = " ".join([f"<span class='skill-match'>{s}</span>" for s in hard_skills])
        st.markdown(skills_html, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='section-title'>📄 Resumen Profesional</div>", unsafe_allow_html=True)
        st.write(user['Resumen_CV'])
        
        st.markdown("<div class='section-title'>💬 Soft Skills</div>", unsafe_allow_html=True)
        soft_skills = [s.strip() for s in str(user['Soft_Skills']).split(',')]
        skills_html = " ".join([f"<span class='skill-match'>{s}</span>" for s in soft_skills])
        st.markdown(skills_html, unsafe_allow_html=True)

def show_analysis():
    st.markdown("---")
    st.markdown("<div class='section-title'>🔍 Análisis Inteligente de Vacantes</div>", unsafe_allow_html=True)
    
    user = st.session_state.user_data
    cv_text = str(user['Resumen_CV']) + " " + str(user['Hard_Skills']) + " " + str(user['Soft_Skills'])
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("✨ Analizar Compatibilidad", use_container_width=True, type="primary"):
            with st.spinner("🔍 Analizando compatibilidad con vacantes..."):
                resultados = perform_matching(cv_text)
            
            if resultados:
                st.session_state.analysis_results = resultados
    
    if 'analysis_results' in st.session_state:
        resultados = st.session_state.analysis_results
        st.markdown("<div class='section-title'>🎯 Resultados del Análisis</div>", unsafe_allow_html=True)
        
        # Mostrar resumen
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total de Vacantes", len(resultados), label_visibility="collapsed")
        with col2:
            mejor_match = max([r['puntaje_match'] for r in resultados]) if resultados else 0
            st.metric("🏆 Mejor Compatibilidad", f"{mejor_match}%", label_visibility="collapsed")
        with col3:
            muy_compatibles = len([r for r in resultados if r['puntaje_match'] > 70])
            st.metric("✨ Muy Compatibles", muy_compatibles, label_visibility="collapsed")
        
        st.markdown("---")
        
        for i, result in enumerate(resultados, 1):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"<div style='color: #1f2937; font-weight: 700; font-size: 1.15rem;'>#### {i}. {result['vacante'].get('titulo', 'Vacante')}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='color: #667eea; font-weight: 600; margin: 0.5rem 0;'>🏢 {result['vacante'].get('empresa', 'Empresa')}</div>", unsafe_allow_html=True)
            
            with col2:
                match_pct = result['puntaje_match']
                if match_pct > 70:
                    st.markdown(f"<div class='match-score excellent'>{match_pct}%</div>", unsafe_allow_html=True)
                elif match_pct > 50:
                    st.markdown(f"<div class='match-score good'>{match_pct}%</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='match-score low'>{match_pct}%</div>", unsafe_allow_html=True)
            
            st.markdown(f"<div style='color: #4b5563; font-style: italic; margin: 0.8rem 0;'>{result['vacante'].get('descripcion', '')}</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div style='color: #1f2937; font-weight: 600; margin-bottom: 0.5rem;'>✅ Habilidades que tienes:</div>", unsafe_allow_html=True)
                if result['habilidades_cumplidas']:
                    skills_html = " ".join([f"<span class='skill-match'>{s}</span>" for s in result['habilidades_cumplidas']])
                    st.markdown(skills_html, unsafe_allow_html=True)
                else:
                    st.markdown("<span class='skill-neutral'>Ninguna de las habilidades requeridas</span>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div style='color: #1f2937; font-weight: 600; margin-bottom: 0.5rem;'>📚 Habilidades a desarrollar:</div>", unsafe_allow_html=True)
                if result['habilidades_faltantes']:
                    skills_html = " ".join([f"<span class='skill-missing'>{s}</span>" for s in result['habilidades_faltantes']])
                    st.markdown(skills_html, unsafe_allow_html=True)
                else:
                    st.markdown("<span class='skill-match'>✓ ¡Cumples todos los requisitos!</span>", unsafe_allow_html=True)
            
            if result['cursos_recomendados']:
                st.markdown("<div style='color: #1f2937; font-weight: 600; margin: 1rem 0 0.5rem 0;'>🎓 Cursos recomendados:</div>", unsafe_allow_html=True)
                cols = st.columns(len(result['cursos_recomendados'][:3]))
                for idx, curso in enumerate(result['cursos_recomendados'][:3]):
                    with cols[idx]:
                        st.info(f"📖 **{curso.get('titulo_curso', 'Curso')}**\n\n{curso.get('proveedor', 'Proveedor')}")
            
            st.markdown("---")

# --- FLUJO PRINCIPAL ---
show_header()

# Navegación por páginas
if st.session_state.current_page == 'inicio':
    show_home()
elif st.session_state.current_page == 'vacantes':
    show_vacantes_page()
elif st.session_state.current_page == 'cursos':
    show_cursos_page()
elif st.session_state.current_page == 'testimonios':
    show_testimonios_page()
elif st.session_state.current_page == 'nosotros':
    show_about_page()
elif st.session_state.current_page == 'privacidad':
    show_privacidad_page()
elif st.session_state.current_page == 'login':
    if not st.session_state.logged_in:
        show_login()
    else:
        show_profile()
        show_analysis()
elif st.session_state.logged_in:
    show_profile()
    show_analysis()
else:
    show_home()
