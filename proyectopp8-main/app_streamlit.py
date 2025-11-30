import streamlit as st
import json
import re
import base64
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import hashlib

# Función para cargar imagen como base64
def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# --- CARGAR DATOS ---
@st.cache_data(ttl=60)  # Recargar datos cada 60 segundos
def load_data():
    try:
        vacantes = json.load(open('vacantes.json', 'r', encoding='utf-8'))
        cursos = json.load(open('cursos.json', 'r', encoding='utf-8'))
        egresados = pd.read_csv('egresados_data.csv', encoding='utf-8')
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
import unicodedata

def normalizar_texto(texto):
    """Normaliza texto removiendo acentos y convirtiendo a minúsculas."""
    texto = texto.lower().strip()
    # Remover acentos
    texto = unicodedata.normalize('NFD', texto)
    texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
    return texto

def normalizar_habilidad(habilidad):
    """Limpia la habilidad para comparación."""
    return normalizar_texto(habilidad)

def extraer_habilidades(cv_texto, lista_habilidades_conocidas):
    """Procesa el texto del CV y busca coincidencias con palabras clave."""
    habilidades_encontradas = set()
    cv_texto_limpio = normalizar_texto(cv_texto)
    
    # Diccionario de sinónimos - la clave es la habilidad normalizada (sin acentos)
    sinonimos = {
        'python': ['python', 'pandas', 'numpy', 'sklearn', 'scikit-learn', 'scikit', 'matplotlib', 
                   'flask', 'django', 'pytorch', 'tensorflow', 'jupyter', 'streamlit'],
        'sql': ['sql', 'postgresql', 'mysql', 'sqlite', 'consultas sql', 'oracle', 'sql server'],
        'excel': ['excel', 'tablas dinamicas', 'power query', 'spreadsheet', 'pivot'],
        'estadistica': ['estadistica', 'estadistico', 'estadisticos', 'analisis estadistico', 
                        'regresion', 'probabilidad', 'inferencia', 'correlacion', 'varianza', 'r avanzado',
                        'descriptiva', 'inferencial', 'hipotesis', 'anova', 'scipy'],
        'docker': ['docker', 'contenedor', 'contenedores', 'container', 'kubernetes', 'k8s'],
        'fastapi': ['fastapi', 'api rest', 'apis rest', 'rest api', 'endpoints', 'microservicios'],
        'trabajo en equipo': ['trabajo en equipo', 'equipo', 'colaborativo', 'colaboracion', 
                              'team', 'equipos', 'multidisciplinario', 'interdisciplinario', 'scrum', 'agil'],
        'resolucion de problemas': ['resolucion de problemas', 'resolver problemas', 'solucion de problemas', 
                                    'problem solving', 'troubleshooting', 'debugging', 'depuracion',
                                    'resolucion problemas'],
        'liderazgo': ['liderazgo', 'liderar', 'lidere', 'lider', 'gestion de equipos', 
                      'mentoria', 'mentor', 'coordinador', 'mentore', 'capacite', 'capacitador'],
        'proactividad': ['proactividad', 'proactivo', 'proactiva', 'iniciativa', 'autonomia', 
                         'autonomo', 'autodidacta', 'propositivo'],
        'comunicacion': ['comunicacion', 'comunicar', 'presentaciones', 'comunicacion efectiva', 
                         'expositor', 'redaccion', 'documentacion', 'reportes', 'c-level'],
        'seo': ['seo', 'optimizacion motores', 'search engine', 'posicionamiento web'],
        'google ads': ['google ads', 'adwords', 'publicidad google', 'sem', 'ppc'],
        'marketing digital': ['marketing digital', 'marketing online', 'estrategias digitales',
                              'redes sociales', 'social media', 'community manager'],
        'creatividad': ['creatividad', 'creativo', 'creativa', 'innovacion', 'innovador', 'diseno']
    }
    
    # Buscar cada habilidad requerida por la vacante
    for habilidad in lista_habilidades_conocidas:
        hab_normalizada = normalizar_texto(habilidad)  # Sin acentos, minúsculas
        encontrada = False
        
        # 1. Buscar coincidencia directa en el CV
        if hab_normalizada in cv_texto_limpio:
            habilidades_encontradas.add(hab_normalizada)
            encontrada = True
            continue
        
        # 2. Buscar si la habilidad tiene sinónimos definidos
        if hab_normalizada in sinonimos:
            for sinonimo in sinonimos[hab_normalizada]:
                if sinonimo in cv_texto_limpio:
                    habilidades_encontradas.add(hab_normalizada)
                    encontrada = True
                    break
    
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
        # Crear mapeo de nombre normalizado a nombre original
        req_originales = vacante.get('requisitos_tecnicos', []) + vacante.get('requisitos_blandos', [])
        mapeo_nombres = {normalizar_texto(h): h for h in req_originales}
        
        req_tec = set(normalizar_texto(h) for h in vacante.get('requisitos_tecnicos', []))
        req_blando = set(normalizar_texto(h) for h in vacante.get('requisitos_blandos', []))
        req_totales = req_tec.union(req_blando)
        
        habilidades_cumplidas = habilidades_cv.intersection(req_totales)
        habilidades_faltantes = req_totales - habilidades_cv

        # Score final basado solo en requisitos cumplidos (sin TF-IDF para simplificar)
        total_req = len(req_totales)
        score_cumplimiento = len(habilidades_cumplidas) / total_req if total_req else 0
        puntaje_final = score_cumplimiento * 100
        
        # Convertir nombres normalizados a nombres originales para mostrar
        habilidades_cumplidas_display = [mapeo_nombres.get(h, h.title()) for h in habilidades_cumplidas]
        habilidades_faltantes_display = [mapeo_nombres.get(h, h.title()) for h in habilidades_faltantes]
        
        # Cursos recomendados
        cursos_recomendados = [
            curso for curso in CURSOS 
            if normalizar_texto(curso.get('habilidad', '')) in habilidades_faltantes
        ]

        resultados.append({
            "vacante": vacante,
            "puntaje_match": round(puntaje_final, 1),
            "habilidades_cumplidas": sorted(habilidades_cumplidas_display),
            "habilidades_faltantes": sorted(habilidades_faltantes_display),
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
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
    else:
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
    
    with col1:
        # Intentar cargar logo, si falla usar texto
        try:
            from PIL import Image
            import os
            logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
            if os.path.exists(logo_path):
                img = Image.open(logo_path)
                col_a, col_b = st.columns([0.15, 0.85])
                with col_a:
                    st.image(img, width=45)
                with col_b:
                    st.markdown("<div style='font-size: 1.3rem; font-weight: 700; color: #60a5fa; padding-top: 5px;'>CogniLink UNRC</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size: 1.5rem; font-weight: 700; color: #60a5fa;'>🔗 CogniLink UNRC</div>", unsafe_allow_html=True)
        except:
            st.markdown("<div style='font-size: 1.5rem; font-weight: 700; color: #60a5fa;'>🔗 CogniLink UNRC</div>", unsafe_allow_html=True)
    
    if st.session_state.logged_in:
        with col2:
            if st.button("Mi Perfil", use_container_width=True):
                st.session_state.current_page = 'perfil'
                st.rerun()
        
        with col3:
            if st.button("Vacantes", use_container_width=True):
                st.session_state.current_page = 'vacantes'
                st.rerun()
        
        with col4:
            if st.button("Cursos", use_container_width=True):
                st.session_state.current_page = 'cursos'
                st.rerun()
        
        with col5:
            if st.button("Salir", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.user_id = None
                st.session_state.user_data = None
                st.session_state.current_page = 'inicio'
                st.rerun()
    else:
        with col2:
            if st.button("Inicio", use_container_width=True):
                st.session_state.current_page = 'inicio'
                st.rerun()
        
        with col3:
            if st.button("Nuestra Historia", use_container_width=True):
                st.session_state.current_page = 'nosotros'
                st.rerun()
        
        with col4:
            if st.button("Acceder", use_container_width=True):
                st.session_state.current_page = 'login'
                st.rerun()
        
        with col5:
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
    # La página de vacantes ahora solo muestra el análisis inteligente
    pass

# --- PÁGINA DE CURSOS ---
def show_cursos_page():
    st.markdown("""
    <div class='section-header'>
        <h2>📚 Cursos Recomendados</h2>
        <p>Desarrolla las habilidades más demandadas del mercado</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mostrar cursos en grid con expanders
    cols = st.columns(3)
    
    for idx, curso in enumerate(CURSOS):
        with cols[idx % 3]:
            icon = "💻" if curso.get('habilidad') in ['Python', 'SQL', 'Excel'] else "📖"
            nivel = curso.get('nivel', 'N/A')
            nivel_color = "#10b981" if nivel == "Principiante" else "#f59e0b" if nivel == "Intermedio" else "#ef4444"
            
            with st.expander(f"{icon} {curso.get('titulo_curso', 'Curso')}"):
                st.markdown(f"""
                <div style='padding: 0.5rem 0;'>
                    <p style='color: #60a5fa; font-weight: 600; margin-bottom: 0.5rem;'>🎓 {curso.get('proveedor', 'Proveedor')}</p>
                    <p style='color: #e2e8f0; line-height: 1.6; margin-bottom: 1rem;'>{curso.get('descripcion', 'Sin descripción disponible.')}</p>
                    <div style='display: flex; gap: 1rem; flex-wrap: wrap;'>
                        <span style='background: #1e293b; color: #cbd5e1; padding: 0.3rem 0.8rem; border-radius: 6px; font-size: 0.85rem;'>⏱️ {curso.get('duracion', 'N/A')}</span>
                        <span style='background: {nivel_color}; color: white; padding: 0.3rem 0.8rem; border-radius: 6px; font-size: 0.85rem;'>📊 {nivel}</span>
                        <span style='background: #065f46; color: white; padding: 0.3rem 0.8rem; border-radius: 6px; font-size: 0.85rem;'>🎯 {curso.get('habilidad', 'Habilidad')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# --- PÁGINA DE AVISO DE PRIVACIDAD ---
def show_privacidad_page():
    # Header
    st.markdown("""
    <div style='max-width: 850px; margin: 2rem auto; padding: 0 1rem;'>
        <div style='background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%); border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.4); overflow: hidden; border: 1px solid #334155;'>
            <div style='background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); padding: 2rem; text-align: center;'>
                <h1 style='color: #ffffff; font-size: 1.8rem; font-weight: 700; margin: 0; text-transform: uppercase; letter-spacing: 1px;'>Aviso de Privacidad</h1>
                <p style='color: rgba(255,255,255,0.9); font-size: 0.9rem; margin: 0.75rem 0 0 0;'>CogniLink UNRC | Última actualización: 28 de noviembre de 2025</p>
            </div>
            <div style='padding: 2rem; color: #e2e8f0; font-size: 0.95rem; line-height: 1.8;'>
                <p style='margin-bottom: 0.75rem; color: #60a5fa; font-weight: 600;'>Datos que recopilamos:</p>
                <p style='margin-bottom: 1.5rem; padding-left: 1rem; color: #cbd5e1; border-left: 2px solid #3b82f6;'>Nombre, identificador de egresado, año de egreso, competencias técnicas y blandas, experiencia laboral, resumen curricular y credenciales de acceso protegidas mediante cifrado SHA-256.</p>
                <p style='margin-bottom: 0.75rem; color: #60a5fa; font-weight: 600;'>Finalidades del tratamiento:</p>
                <p style='margin-bottom: 1.5rem; padding-left: 1rem; color: #cbd5e1; border-left: 2px solid #3b82f6;'>Sus datos serán utilizados para: (i) realizar el matching inteligente entre su perfil y vacantes laborales mediante algoritmos de NLP; (ii) recomendar cursos de capacitación personalizados; (iii) facilitar la vinculación con empresas; (iv) enviar comunicaciones sobre oportunidades relevantes; y (v) mejorar nuestros servicios.</p>
                <p style='margin-bottom: 0.75rem; color: #60a5fa; font-weight: 600;'>Transferencia de datos:</p>
                <p style='margin-bottom: 1.5rem; padding-left: 1rem; color: #cbd5e1; border-left: 2px solid #3b82f6;'>Sus datos podrán compartirse con empresas registradas en la plataforma para fines de reclutamiento, proveedores de capacitación asociados, y autoridades cuando sea legalmente requerido. No vendemos ni comercializamos sus datos personales.</p>
                <p style='margin-bottom: 0.75rem; color: #60a5fa; font-weight: 600;'>Derechos ARCO:</p>
                <p style='margin-bottom: 1.5rem; padding-left: 1rem; color: #cbd5e1; border-left: 2px solid #3b82f6;'>Usted tiene derecho a Acceder, Rectificar, Cancelar u Oponerse al tratamiento de sus datos personales. Para ejercer estos derechos, envíe su solicitud a <strong style='color: #60a5fa;'>contacto@cognilink.unrc.edu.ar</strong> indicando su nombre completo y el derecho que desea ejercer.</p>
                <p style='margin-bottom: 0.75rem; color: #60a5fa; font-weight: 600;'>Medidas de seguridad:</p>
                <p style='margin-bottom: 1.5rem; padding-left: 1rem; color: #cbd5e1; border-left: 2px solid #3b82f6;'>Implementamos medidas técnicas y organizativas alineadas con ISO/IEC 27001, incluyendo cifrado de contraseñas, control de acceso, protección contra código malicioso y transmisión segura mediante TLS.</p>
                <p style='margin-bottom: 0.75rem; color: #60a5fa; font-weight: 600;'>Cookies:</p>
                <p style='margin-bottom: 1.5rem; padding-left: 1rem; color: #cbd5e1; border-left: 2px solid #3b82f6;'>Utilizamos cookies de sesión para mantener su acceso seguro. Estas se eliminan automáticamente al cerrar el navegador.</p>
                <p style='margin-bottom: 0.75rem; color: #60a5fa; font-weight: 600;'>Modificaciones:</p>
                <p style='margin-bottom: 1.5rem; padding-left: 1rem; color: #cbd5e1; border-left: 2px solid #3b82f6;'>Nos reservamos el derecho de modificar este aviso. Los cambios serán notificados a través de la plataforma.</p>
                <hr style='border: none; border-top: 1px solid #334155; margin: 1.5rem 0;'>
                <p style='font-size: 0.9rem; color: #94a3b8; text-align: center; margin-bottom: 0;'>Al utilizar CogniLink UNRC, usted acepta los términos de este Aviso de Privacidad.<br><strong style='color: #e2e8f0;'>Contacto:</strong> contacto@cognilink.unrc.edu.ar | Tel: +54 358 467-6200</p>
            </div>
            <div style='background: #0f172a; padding: 1.25rem 2rem; text-align: center; border-top: 1px solid #334155;'>
                <p style='color: #64748b; font-size: 0.8rem; margin: 0;'>© 2025 CogniLink UNRC - Universidad Nacional Rosario Castellanos<br><span style='color: #22c55e;'>✓</span> Documento conforme a ISO/IEC 27001:2022</p>
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
    
    # Banner superior con gradiente
    st.markdown("""
    <div style='height: 100px; background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%); 
                border-radius: 20px 20px 0 0; margin-bottom: -50px;'></div>
    """, unsafe_allow_html=True)
    
    # Avatar y nombre
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); 
                border-radius: 0 0 20px 20px; padding: 1rem 2rem 2rem 2rem;
                box-shadow: 0 25px 50px rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.1); margin-bottom: 2rem;'>
        <div style='display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap;'>
            <div style='width: 100px; height: 100px; background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); 
                        border-radius: 50%; display: flex; align-items: center; justify-content: center;
                        font-size: 2.5rem; color: white; font-weight: 700; 
                        box-shadow: 0 10px 30px rgba(59,130,246,0.5); border: 4px solid #0f172a;'>
                {user['Nombre'][0]}
            </div>
            <div style='flex: 1;'>
                <h1 style='color: #ffffff; font-size: 1.8rem; font-weight: 700; margin: 0 0 0.3rem 0;'>{user['Nombre']}</h1>
                <p style='color: #60a5fa; font-size: 1.1rem; font-weight: 600; margin: 0 0 1rem 0;'>{user['Rol_Deseado']}</p>
                <div style='display: flex; gap: 2rem; flex-wrap: wrap;'>
                    <div style='text-align: center;'>
                        <p style='color: #60a5fa; font-size: 1.3rem; margin: 0;'>🎓</p>
                        <p style='color: #94a3b8; font-size: 0.7rem; margin: 0; text-transform: uppercase;'>Egreso</p>
                        <p style='color: #ffffff; font-size: 1rem; font-weight: 600; margin: 0;'>{user['Anio_Egreso']}</p>
                    </div>
                    <div style='text-align: center;'>
                        <p style='color: #10b981; font-size: 1.3rem; margin: 0;'>💼</p>
                        <p style='color: #94a3b8; font-size: 0.7rem; margin: 0; text-transform: uppercase;'>Experiencia</p>
                        <p style='color: #ffffff; font-size: 1rem; font-weight: 600; margin: 0;'>{user['Experiencia_Anios']} años</p>
                    </div>
                    <div style='text-align: center;'>
                        <p style='color: #8b5cf6; font-size: 1.3rem; margin: 0;'>🆔</p>
                        <p style='color: #94a3b8; font-size: 0.7rem; margin: 0; text-transform: uppercase;'>ID</p>
                        <p style='color: #ffffff; font-size: 1rem; font-weight: 600; margin: 0;'>{user['ID_Egresado']}</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sección de Habilidades en dos columnas
    st.markdown("<h2 style='color: #ffffff; font-size: 1.4rem; font-weight: 700; margin: 2rem 0 1rem 0; display: flex; align-items: center; gap: 0.5rem;'>🎯 Mis Competencias</h2>", unsafe_allow_html=True)
    
    col_skills1, col_skills2 = st.columns(2)
    
    with col_skills1:
        # Hard Skills
        hard_skills = [s.strip() for s in str(user['Hard_Skills']).split(',')]
        skills_html = "".join([f"<span style='background: linear-gradient(135deg, #059669 0%, #10b981 100%); color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.85rem; font-weight: 500; display: inline-block; margin: 0.25rem; box-shadow: 0 2px 8px rgba(16,185,129,0.3);'>{s}</span>" for s in hard_skills])
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1e293b 0%, #334155 100%); border-radius: 16px; 
                    padding: 1.5rem; border: 1px solid #475569; height: 100%;'>
            <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem; padding-bottom: 0.8rem; border-bottom: 2px solid #10b981;'>
                <span style='font-size: 1.3rem;'>🛠️</span>
                <h3 style='color: #10b981; font-size: 1.1rem; font-weight: 600; margin: 0;'>Habilidades Técnicas</h3>
                <span style='background: #10b981; color: white; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; margin-left: auto;'>{len(hard_skills)}</span>
            </div>
            <div style='display: flex; flex-wrap: wrap; gap: 0.5rem;'>
                {skills_html}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_skills2:
        # Soft Skills
        soft_skills = [s.strip() for s in str(user['Soft_Skills']).split(',')]
        soft_skills_html = "".join([f"<span style='background: linear-gradient(135deg, #7c3aed 0%, #8b5cf6 100%); color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.85rem; font-weight: 500; display: inline-block; margin: 0.25rem; box-shadow: 0 2px 8px rgba(139,92,246,0.3);'>{s}</span>" for s in soft_skills])
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1e293b 0%, #334155 100%); border-radius: 16px; 
                    padding: 1.5rem; border: 1px solid #475569; height: 100%;'>
            <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem; padding-bottom: 0.8rem; border-bottom: 2px solid #8b5cf6;'>
                <span style='font-size: 1.3rem;'>💬</span>
                <h3 style='color: #8b5cf6; font-size: 1.1rem; font-weight: 600; margin: 0;'>Habilidades Blandas</h3>
                <span style='background: #8b5cf6; color: white; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; margin-left: auto;'>{len(soft_skills)}</span>
            </div>
            <div style='display: flex; flex-wrap: wrap; gap: 0.5rem;'>
                {soft_skills_html}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Sección del CV completo
    st.markdown("<h2 style='color: #ffffff; font-size: 1.4rem; font-weight: 700; margin: 2rem 0 1rem 0; display: flex; align-items: center; gap: 0.5rem;'>📄 Mi Trayectoria Profesional</h2>", unsafe_allow_html=True)
    
    cv_completo = str(user['Resumen_CV'])
    
    # Formatear el CV para mejor visualización profesional
    cv_formateado = cv_completo.replace("CURRÍCULUM VITAE -", "<div style='text-align: center; margin-bottom: 1.5rem;'><h3 style='color: #60a5fa; margin: 0; font-size: 1.2rem; font-weight: 700;'>")
    cv_formateado = cv_formateado.replace("| DATOS PERSONALES:", "</h3></div><div style='display: none;'>")
    cv_formateado = cv_formateado.replace("| PERFIL PROFESIONAL:", "</div><div style='background: rgba(16,185,129,0.1); border-left: 4px solid #10b981; padding: 1rem 1.5rem; border-radius: 0 12px 12px 0; margin-bottom: 1.5rem;'><strong style='color: #10b981; font-size: 1rem; display: block; margin-bottom: 0.5rem;'>👤 PERFIL PROFESIONAL</strong><span style='color: #e2e8f0; line-height: 1.7;'>")
    cv_formateado = cv_formateado.replace("| EXPERIENCIA LABORAL:", "</span></div><div style='background: rgba(245,158,11,0.1); border-left: 4px solid #f59e0b; padding: 1rem 1.5rem; border-radius: 0 12px 12px 0; margin-bottom: 1.5rem;'><strong style='color: #f59e0b; font-size: 1rem; display: block; margin-bottom: 0.5rem;'>💼 EXPERIENCIA LABORAL</strong><span style='color: #e2e8f0; line-height: 1.7;'>")
    cv_formateado = cv_formateado.replace("| PROYECTOS:", "</span></div><div style='background: rgba(139,92,246,0.1); border-left: 4px solid #8b5cf6; padding: 1rem 1.5rem; border-radius: 0 12px 12px 0; margin-bottom: 1.5rem;'><strong style='color: #8b5cf6; font-size: 1rem; display: block; margin-bottom: 0.5rem;'>🚀 PROYECTOS DESTACADOS</strong><span style='color: #e2e8f0; line-height: 1.7;'>")
    cv_formateado = cv_formateado.replace("| FORMACIÓN:", "</span></div><div style='background: rgba(236,72,153,0.1); border-left: 4px solid #ec4899; padding: 1rem 1.5rem; border-radius: 0 12px 12px 0; margin-bottom: 1.5rem;'><strong style='color: #ec4899; font-size: 1rem; display: block; margin-bottom: 0.5rem;'>🎓 FORMACIÓN ACADÉMICA</strong><span style='color: #e2e8f0; line-height: 1.7;'>")
    cv_formateado = cv_formateado.replace("| HABILIDADES TÉCNICAS:", "</span></div><div style='display: none;'>")
    cv_formateado = cv_formateado.replace("| HABILIDADES BLANDAS:", "</div><div style='display: none;'>")
    cv_formateado = cv_formateado.replace("| PUBLICACIONES:", "</span></div><div style='background: rgba(59,130,246,0.1); border-left: 4px solid #3b82f6; padding: 1rem 1.5rem; border-radius: 0 12px 12px 0; margin-bottom: 1.5rem;'><strong style='color: #3b82f6; font-size: 1rem; display: block; margin-bottom: 0.5rem;'>📚 PUBLICACIONES</strong><span style='color: #e2e8f0; line-height: 1.7;'>")
    cv_formateado = cv_formateado + "</span></div>"
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1e293b 0%, #334155 100%); border-radius: 16px; 
                padding: 2rem; border: 1px solid #475569;'>
        <div style='color: #e2e8f0; font-size: 0.95rem;'>{cv_formateado}</div>
    </div>
    """, unsafe_allow_html=True)

def show_analysis():
    # Header de la sección de análisis
    st.markdown("""
    <div style='background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 16px; 
                padding: 2rem; margin: 2rem 0; border: 1px solid rgba(255,255,255,0.1);
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);'>
        <div style='text-align: center;'>
            <h2 style='color: #ffffff; font-size: 1.8rem; font-weight: 700; margin: 0;'>
                🔍 Análisis Inteligente de Vacantes
            </h2>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Campo de texto para pegar CV
    cv_text = st.text_area(
        "📄 Pega aquí tu CV o descripción profesional:",
        height=200,
        placeholder="Pega aquí el contenido de tu CV, incluyendo tu experiencia laboral, habilidades técnicas, soft skills, certificaciones, proyectos, etc.\n\nEntre más detallada sea la información, mejor será el análisis de compatibilidad con las vacantes disponibles.",
        help="El sistema analizará tu perfil y lo comparará con las vacantes disponibles para mostrarte el porcentaje de compatibilidad."
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analizar_btn = st.button("✨ Analizar Compatibilidad", use_container_width=True, type="primary")
    
    if analizar_btn:
        if cv_text.strip():
            with st.spinner("🔍 Analizando compatibilidad con vacantes..."):
                resultados = perform_matching(cv_text)
            
            if resultados:
                st.session_state.analysis_results = resultados
                st.session_state.cv_analizado = True
        else:
            st.warning("⚠️ Por favor, pega el contenido de tu CV para realizar el análisis.")
    
    if 'analysis_results' in st.session_state and st.session_state.get('cv_analizado'):
        resultados = st.session_state.analysis_results
        
        # Resumen de resultados en tarjetas
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        mejor_match = max([r['puntaje_match'] for r in resultados]) if resultados else 0
        muy_compatibles = len([r for r in resultados if r['puntaje_match'] > 70])
        
        with col1:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1e293b 0%, #334155 100%); border-radius: 12px; 
                        padding: 1.5rem; text-align: center; border: 1px solid #475569;'>
                <p style='color: #60a5fa; font-size: 2.5rem; font-weight: 700; margin: 0;'>{len(resultados)}</p>
                <p style='color: #94a3b8; font-size: 0.9rem; margin: 0.5rem 0 0 0;'>Vacantes Analizadas</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1e293b 0%, #334155 100%); border-radius: 12px; 
                        padding: 1.5rem; text-align: center; border: 1px solid #475569;'>
                <p style='color: #10b981; font-size: 2.5rem; font-weight: 700; margin: 0;'>{mejor_match}%</p>
                <p style='color: #94a3b8; font-size: 0.9rem; margin: 0.5rem 0 0 0;'>Mejor Compatibilidad</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1e293b 0%, #334155 100%); border-radius: 12px; 
                        padding: 1.5rem; text-align: center; border: 1px solid #475569;'>
                <p style='color: #f59e0b; font-size: 2.5rem; font-weight: 700; margin: 0;'>{muy_compatibles}</p>
                <p style='color: #94a3b8; font-size: 0.9rem; margin: 0.5rem 0 0 0;'>Alta Compatibilidad (+70%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Listado de vacantes
        for i, result in enumerate(resultados, 1):
            # Calcular porcentaje sobre 100 basado en requisitos
            total_requisitos = len(result['habilidades_cumplidas']) + len(result['habilidades_faltantes'])
            cumplidos = len(result['habilidades_cumplidas'])
            if total_requisitos > 0:
                match_pct = round((cumplidos / total_requisitos) * 100, 1)
            else:
                match_pct = 0
            
            # Color según compatibilidad
            if match_pct >= 70:
                match_color = "#10b981"
                match_label = "Alta"
            elif match_pct >= 50:
                match_color = "#f59e0b"
                match_label = "Media"
            else:
                match_color = "#ef4444"
                match_label = "Baja"
            
            # Tarjeta de vacante usando componentes de Streamlit
            with st.container():
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #1e293b 0%, #334155 100%); border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem; border: 1px solid #475569; border-left: 4px solid {match_color};'>
                    <div style='display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;'>
                        <div style='flex: 1;'>
                            <h3 style='color: #ffffff; font-size: 1.25rem; font-weight: 700; margin: 0 0 0.3rem 0;'>{result['vacante'].get('titulo', 'Vacante')}</h3>
                            <p style='color: #60a5fa; font-weight: 600; font-size: 1rem; margin: 0;'>🏢 {result['vacante'].get('empresa', 'Empresa')}</p>
                        </div>
                        <div style='background: rgba(0,0,0,0.3); border: 2px solid {match_color}; border-radius: 12px; padding: 0.8rem 1.2rem; text-align: center; min-width: 130px;'>
                            <p style='color: {match_color}; font-size: 2rem; font-weight: 700; margin: 0; line-height: 1;'>{match_pct}%</p>
                            <p style='color: {match_color}; font-size: 0.75rem; margin: 0.2rem 0 0 0; text-transform: uppercase;'>Compatibilidad {match_label}</p>
                            <div style='width: 100%; background: #1e293b; border-radius: 10px; height: 6px; margin: 0.5rem 0;'>
                                <div style='width: {match_pct}%; background: {match_color}; height: 100%; border-radius: 10px;'></div>
                            </div>
                            <p style='color: #94a3b8; font-size: 0.75rem; margin: 0;'>{cumplidos} de {total_requisitos} requisitos</p>
                        </div>
                    </div>
                    <p style='color: #cbd5e1; font-size: 0.95rem; line-height: 1.6; margin: 1rem 0; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px;'>{result['vacante'].get('descripcion', '')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Habilidades en columnas de Streamlit
                col_hab1, col_hab2 = st.columns(2)
                
                with col_hab1:
                    st.markdown(f"<p style='color: #10b981; font-weight: 600; font-size: 0.9rem; margin: 0 0 0.5rem 0;'>✅ Requisitos que cumples ({cumplidos})</p>", unsafe_allow_html=True)
                    if result['habilidades_cumplidas']:
                        hab_html = ""
                        for s in result['habilidades_cumplidas']:
                            hab_html += f"<span style='background: linear-gradient(135deg, #059669 0%, #10b981 100%); color: white; padding: 0.3rem 0.6rem; border-radius: 6px; font-size: 0.8rem; font-weight: 500; display: inline-block; margin: 0.15rem;'>{s}</span>"
                        st.markdown(hab_html, unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color: #64748b; font-style: italic;'>Ninguna coincidente</span>", unsafe_allow_html=True)
                
                with col_hab2:
                    st.markdown(f"<p style='color: #ef4444; font-weight: 600; font-size: 0.9rem; margin: 0 0 0.5rem 0;'>📚 Requisitos faltantes ({len(result['habilidades_faltantes'])})</p>", unsafe_allow_html=True)
                    if result['habilidades_faltantes']:
                        hab_html = ""
                        for s in result['habilidades_faltantes']:
                            hab_html += f"<span style='background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); color: white; padding: 0.3rem 0.6rem; border-radius: 6px; font-size: 0.8rem; font-weight: 500; display: inline-block; margin: 0.15rem;'>{s}</span>"
                        st.markdown(hab_html, unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color: #10b981; font-weight: 600;'>✓ Cumples todos los requisitos</span>", unsafe_allow_html=True)
                
                # Cursos recomendados
                if result['cursos_recomendados']:
                    with st.expander(f"🎓 Ver {len(result['cursos_recomendados'])} curso(s) recomendado(s)"):
                        cols_cursos = st.columns(min(len(result['cursos_recomendados']), 3))
                        for idx, curso in enumerate(result['cursos_recomendados'][:3]):
                            with cols_cursos[idx]:
                                st.markdown(f"""
                                <div style='background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 10px; padding: 1rem; border: 1px solid #475569;'>
                                    <p style='color: #60a5fa; font-weight: 600; font-size: 0.95rem; margin: 0 0 0.5rem 0;'>📖 {curso.get('titulo_curso', 'Curso')}</p>
                                    <p style='color: #94a3b8; font-size: 0.85rem; margin: 0;'>🎓 {curso.get('proveedor', 'Proveedor')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                
                st.markdown("<hr style='border: none; border-top: 1px solid #334155; margin: 1.5rem 0;'>", unsafe_allow_html=True)

# --- FLUJO PRINCIPAL ---
show_header()

# Navegación por páginas
if st.session_state.current_page == 'inicio':
    if st.session_state.logged_in:
        show_profile()
    else:
        show_home()
elif st.session_state.current_page == 'perfil':
    show_profile()
elif st.session_state.current_page == 'vacantes':
    show_vacantes_page()
    if st.session_state.logged_in:
        show_analysis()
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
elif st.session_state.logged_in:
    show_profile()
else:
    show_home()
