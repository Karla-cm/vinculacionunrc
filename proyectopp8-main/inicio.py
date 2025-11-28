from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Tuple
from textwrap import dedent
import streamlit as st
import pandas as pd
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# ===============================
# 📌 CONFIGURACIÓN Y CONSTANTES
# ===============================

API_URL = "http://localhost:5000"
SESSION_KEYS = {
    'user': 'current_user',
    'profile': 'user_profile',
    'results': 'analysis_results',
    'cv_text': 'cv_text_input'
}

# ===============================
# 📌 MODELOS DE DATOS
# ===============================

@dataclass
class ExperienciaLaboral:
    """Representa una experiencia laboral en el perfil de un candidato."""
    puesto: str
    empresa: str
    descripcion: str
    fecha_inicio: date
    fecha_fin: Optional[date] = None

    def __str__(self) -> str:
        fecha_fin_str = self.fecha_fin.strftime("%B %Y") if self.fecha_fin else "Actualidad"
        return (
            f"- Puesto: {self.puesto} en {self.empresa} "
            f"({self.fecha_inicio.strftime('%B %Y')} - {fecha_fin_str})\n"
            f"  Descripción: {self.descripcion}"
        )


@dataclass
class OfertaDeTrabajo:
    """Modela una oferta de trabajo con sus requisitos."""
    puesto: str
    empresa: str
    habilidades_requeridas: List[str] = field(default_factory=list)
    experiencia_minima_meses: int = 0


@dataclass
class PerfilCandidato:
    """Modela el perfil de un candidato en la plataforma."""
    nombre: str
    email: str
    telefono: Optional[str] = None
    resumen_profesional: str = ""
    habilidades: List[str] = field(default_factory=list)
    experiencias: List[ExperienciaLaboral] = field(default_factory=list)

    def get_experiencia_total_meses(self) -> int:
        """Calcula los meses totales de experiencia del candidato."""
        hoy = date.today()
        total_meses = 0
        for exp in self.experiencias:
            fecha_fin = exp.fecha_fin or hoy
            total_meses += (fecha_fin.year - exp.fecha_inicio.year) * 12 + (fecha_fin.month - exp.fecha_inicio.month)
        return total_meses

    def get_habilidades_normalizadas(self) -> List[str]:
        """Devuelve las habilidades en minúsculas para comparación."""
        return [h.lower() for h in self.habilidades]

    def agregar_habilidad(self, habilidad: str) -> None:
        """Agrega una habilidad si no existe (ignorando mayúsculas/minúsculas)."""
        if habilidad.lower() not in [h.lower() for h in self.habilidades]:
            self.habilidades.append(habilidad)

    def agregar_experiencia(self, experiencia: ExperienciaLaboral) -> None:
        """Agrega una experiencia laboral al perfil, ordenada por fecha."""
        self.experiencias.append(experiencia)
        self.experiencias.sort(key=lambda exp: exp.fecha_inicio, reverse=True)


# ===============================
# 🎨 COMPONENTES DE UI
# ===============================

class UIComponents:
    """Componentes reutilizables para la interfaz de usuario."""
    
    @staticmethod
    def apply_custom_styles():
        """Aplica estilos CSS personalizados a la aplicación."""
        st.markdown("""
        <style>
        /* Estilos generales */
        .main {
            background-color: #f8fafc;
        }
        
        /* Tarjetas principales */
        .main-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            padding: 2.5rem;
            margin: 2rem auto;
            color: white;
            max-width: 700px;
            text-align: center;
        }
        
        /* Tarjetas de resultados */
        .result-card {
            background: white;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            padding: 2rem;
            margin: 1.5rem 0;
            border-left: 5px solid #667eea;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .result-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        }
        
        /* Botones */
        .stButton > button {
            border-radius: 12px;
            padding: 0.7rem 2rem;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        
        .primary-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .secondary-button {
            background: #f1f5f9;
            color: #475569;
            border: 2px solid #e2e8f0 !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        }
        
        /* Habilidades */
        .skill-match {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            border-radius: 20px;
            padding: 0.4rem 1rem;
            margin: 0.3rem;
            display: inline-block;
            font-weight: 500;
            font-size: 0.9rem;
        }
        
        .skill-missing {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
            border-radius: 20px;
            padding: 0.4rem 1rem;
            margin: 0.3rem;
            display: inline-block;
            font-weight: 500;
            font-size: 0.9rem;
        }
        
        .skill-neutral {
            background: #f1f5f9;
            color: #475569;
            border-radius: 20px;
            padding: 0.4rem 1rem;
            margin: 0.3rem;
            display: inline-block;
            font-weight: 500;
            font-size: 0.9rem;
        }
        
        /* Cursos */
        .course-card {
            background: linear-gradient(135deg, #dbeafe 0%, #eff6ff 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            border-left: 4px solid #3b82f6;
            transition: all 0.3s ease;
        }
        
        .course-card:hover {
            transform: translateX(5px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        /* Barras de progreso */
        .match-score {
            font-size: 2rem;
            font-weight: bold;
            color: #1e293b;
            margin: 1rem 0;
        }
        
        /* Iconos */
        .icon {
            font-size: 1.2rem;
            margin-right: 0.5rem;
        }
        
        /* Inputs */
        .stTextInput > div > div > input {
            border-radius: 12px;
            border: 2px solid #e2e8f0;
            padding: 0.8rem 1rem;
            font-size: 1rem;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        /* Text areas */
        .stTextArea > div > div > textarea {
            border-radius: 12px;
            border: 2px solid #e2e8f0;
            padding: 1rem;
            font-size: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_main_header():
        """Crea el encabezado principal de la aplicación."""
        st.markdown("""
        <div class='main-card'>
            <div style='display: flex; align-items: center; justify-content: center; margin-bottom: 1.5rem;'>
                <div style='background: white; padding: 1rem; border-radius: 50%; margin-right: 1rem;'>
                    <span style='font-size: 2rem; color: #667eea;'>💼</span>
                </div>
                <h1 style='margin: 0; color: white; font-size: 2.5rem;'>CogniLink UNRC</h1>
            </div>
            <p style='font-size: 1.2rem; margin-bottom: 0; opacity: 0.9;'>
                Sistema inteligente de vinculación laboral para egresados UNRC
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_login_form():
        """Crea el formulario de login."""
        with st.form("login_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            
            with col1:
                id_input = st.text_input(
                    "🔑 ID de egresado",
                    placeholder="Ej: UNRC12345",
                    max_chars=10,
                    help="Ingresa tu ID de egresado UNRC"
                )
            
            with col2:
                password_input = st.text_input(
                    "🔒 Contraseña",
                    type="password",
                    placeholder="Tu contraseña",
                    help="Ingresa tu contraseña personal"
                )
            
            login_btn = st.form_submit_button(
                "🚀 Ingresar al Sistema",
                use_container_width=True
            )
            
            return id_input, password_input, login_btn
    
    @staticmethod
    def create_user_profile_card(user_data):
        """Crea una tarjeta con el perfil del usuario."""
        st.markdown(f"""
        <div class='result-card'>
            <div style='display: flex; align-items: center; margin-bottom: 1.5rem;'>
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 50%; margin-right: 1rem;'>
                    <span style='font-size: 1.5rem; color: white;'>👤</span>
                </div>
                <div>
                    <h2 style='color: #1e293b; margin: 0;'>Perfil Profesional</h2>
                    <p style='color: #64748b; margin: 0;'>Bienvenido/a al sistema</p>
                </div>
            </div>
            
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem;'>
                <div>
                    <h4 style='color: #667eea; margin-bottom: 0.5rem;'>📋 Información Personal</h4>
                    <p><b>Nombre:</b> {user_data['Nombre']}</p>
                    <p><b>ID:</b> {user_data['ID_Egresado']}</p>
                    <p><b>Año de Egreso:</b> {user_data['Anio_Egreso']}</p>
                </div>
                <div>
                    <h4 style='color: #667eea; margin-bottom: 0.5rem;'>🎯 Objetivos Profesionales</h4>
                    <p><b>Rol Deseado:</b> {user_data['Rol_Deseado']}</p>
                    <p><b>Experiencia:</b> {user_data['Experiencia_Anios']} años</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_skills_section(hard_skills, soft_skills):
        """Crea la sección de habilidades del usuario."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🛠️ Hard Skills")
            if hard_skills:
                skills_html = " ".join([f"<span class='skill-match'>{skill}</span>" for skill in hard_skills])
                st.markdown(skills_html, unsafe_allow_html=True)
            else:
                st.info("No hay hard skills registradas")
        
        with col2:
            st.markdown("### 💬 Soft Skills")
            if soft_skills:
                skills_html = " ".join([f"<span class='skill-neutral'>{skill}</span>" for skill in soft_skills])
                st.markdown(skills_html, unsafe_allow_html=True)
            else:
                st.info("No hay soft skills registradas")
    
    @staticmethod
    def create_analysis_section():
        """Crea la sección de análisis de CV."""
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; margin: 2rem 0;'>
            <h2 style='color: #1e293b;'>🔍 Análisis Inteligente de CV</h2>
            <p style='color: #64748b;'>
                Obtén recomendaciones personalizadas de vacantes y cursos basadas en tu perfil
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        cv_text = st.text_area(
            "📄 Pega aquí el texto de tu CV:",
            placeholder="Copia y pega el contenido completo de tu CV aquí...",
            height=200,
            help="Incluye tu experiencia laboral, educación, habilidades y proyectos relevantes"
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            analyze_btn = st.button(
                "✨ Analizar y Recomendar",
                use_container_width=True,
                type="primary"
            )
        
        return cv_text, analyze_btn
    
    @staticmethod
    def create_results_section(results):
        """Crea la sección de resultados del análisis."""
        if not results:
            return
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; margin: 2rem 0;'>
            <h2 style='color: #1e293b;'>🎯 Resultados del Análisis</h2>
            <p style='color: #64748b;'>Vacantes ordenadas por compatibilidad con tu perfil</p>
        </div>
        """, unsafe_allow_html=True)
        
        for i, result in enumerate(results, 1):
            with st.container():
                st.markdown(f"""
                <div class='result-card'>
                    <div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;'>
                        <div>
                            <h3 style='color: #1e293b; margin: 0;'>{result['vacante']['titulo']}</h3>
                            <p style='color: #667eea; font-weight: 600; margin: 0;'>{result['vacante']['empresa']}</p>
                        </div>
                        <div class='match-score' style='color: {'#10b981' if result['puntaje_match'] > 70 else '#f59e0b' if result['puntaje_match'] > 50 else '#ef4444'};'>
                            {result['puntaje_match']}%
                        </div>
                    </div>
                    
                    <div style='margin: 1.5rem 0;'>
                        <h4 style='color: #475569; margin-bottom: 0.5rem;'>✅ Habilidades Coincidentes</h4>
                        {' '.join([f"<span class='skill-match'>{skill}</span>" for skill in result['habilidades_cumplidas']])}
                    </div>
                    
                    <div style='margin: 1.5rem 0;'>
                        <h4 style='color: #475569; margin-bottom: 0.5rem;'>📚 Habilidades a Desarrollar</h4>
                        {' '.join([f"<span class='skill-missing'>{skill}</span>" for skill in result['habilidades_faltantes']]) if result['habilidades_faltantes'] else '<p style=\"color: #64748b;\">¡Excelente! Cumples con todos los requisitos</p>'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Cursos recomendados
                if result['cursos_recomendados']:
                    st.markdown("#### 🎓 Cursos Recomendados")
                    for curso in result['cursos_recomendados']:
                        st.markdown(f"""
                        <div class='course-card'>
                            <h5 style='color: #1e293b; margin: 0 0 0.5rem 0;'>{curso['titulo_curso']}</h5>
                            <p style='color: #64748b; margin: 0 0 0.5rem 0;'>
                                <b>Proveedor:</b> {curso['proveedor']} | 
                                <b>Habilidad:</b> {curso['habilidad']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Botones de acción
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.button(
                        f"📨 Aplicar a Vacante {i}",
                        key=f"apply_{i}",
                        use_container_width=True
                    )
                with col2:
                    st.button(
                        f"💾 Guardar Análisis {i}",
                        key=f"save_{i}",
                        use_container_width=True
                    )
                with col3:
                    st.button(
                        f"📊 Ver Detalles {i}",
                        key=f"details_{i}",
                        use_container_width=True
                    )
                
                if i < len(results):
                    st.markdown("---")


# ===============================
# 🔧 UTILIDADES Y SERVICIOS
# ===============================

class DataService:
    """Servicio para manejar datos y operaciones relacionadas."""
    
    @staticmethod
    def load_sample_data():
        """Carga datos de ejemplo para demostración."""
        try:
            from db_utils import cargar_tabla
            df_egresados = cargar_tabla('egresados')
            df_ofertas = cargar_tabla('ofertas')
            df_habilidades = cargar_tabla('habilidades')
            return df_egresados, df_ofertas, df_habilidades
        except ImportError:
            # Datos de ejemplo
            return (
                pd.DataFrame([{
                    'ID_Egresado': 'UNRC123',
                    'Nombre': 'María González',
                    'Anio_Egreso': 2020,
                    'Rol_Deseado': 'Desarrollador Full Stack',
                    'Experiencia_Anios': 3,
                    'Hard_Skills': 'Python, JavaScript, React, SQL, Django',
                    'Soft_Skills': 'Trabajo en equipo, Comunicación, Liderazgo',
                    'Resumen_CV': 'Desarrolladora full stack con experiencia en aplicaciones web modernas'
                }]),
                pd.DataFrame([{
                    'Puesto': 'Desarrollador Python Senior',
                    'Empresa': 'Tech Innovations',
                    'Min_Exp_Anios': 4,
                    'Req_Hard_Skills': 'Python, Django, PostgreSQL, AWS',
                    'Req_Soft_Skills': 'Liderazgo, Comunicación',
                    'Descripcion_Puesto': 'Desarrollo de aplicaciones empresariales escalables'
                }]),
                pd.DataFrame()
            )
    
    @staticmethod
    def analyze_cv_with_api(cv_text):
        """Analiza el CV utilizando la API Flask."""
        try:
            # Simular llamada a API (en producción sería una llamada real)
            time.sleep(2)  # Simular procesamiento
            
            # Datos de ejemplo para demostración
            return [
                {
                    "vacante": {
                        "id": 1,
                        "titulo": "Desarrollador Python Senior",
                        "empresa": "Tech Solutions",
                        "descripcion": "Desarrollo de aplicaciones web con Python y Django"
                    },
                    "puntaje_match": 85.5,
                    "habilidades_cumplidas": ["python", "django", "sql"],
                    "habilidades_faltantes": ["aws", "docker"],
                    "cursos_recomendados": [
                        {
                            "habilidad": "AWS",
                            "titulo_curso": "AWS Certified Solutions Architect",
                            "proveedor": "Amazon Training"
                        },
                        {
                            "habilidad": "Docker",
                            "titulo_curso": "Docker para Desarrolladores",
                            "proveedor": "Platzi"
                        }
                    ]
                },
                {
                    "vacante": {
                        "id": 2,
                        "titulo": "Data Scientist",
                        "empresa": "Data Analytics Corp",
                        "descripcion": "Análisis de datos y machine learning"
                    },
                    "puntaje_match": 72.3,
                    "habilidades_cumplidas": ["python", "sql"],
                    "habilidades_faltantes": ["machine learning", "estadística"],
                    "cursos_recomendados": [
                        {
                            "habilidad": "Machine Learning",
                            "titulo_curso": "Machine Learning Specialization",
                            "proveedor": "Coursera"
                        }
                    ]
                }
            ]
        except Exception as e:
            st.error(f"Error al analizar el CV: {str(e)}")
            return []


# ===============================
# 🧠 FUNCIONES NLP (mantenidas del código original)
# ===============================

def normalizar_habilidad(habilidad):
    """Limpia la habilidad y maneja sinónimos básicos y versiones."""
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
    """Procesa el texto del CV y busca coincidencias con las habilidades conocidas."""
    habilidades_encontradas = set()
    habilidades_normalizadas = [normalizar_habilidad(h) for h in lista_habilidades_conocidas]
    cv_texto_limpio = normalizar_habilidad(cv_texto)
    for habilidad in habilidades_normalizadas:
        if habilidad in cv_texto_limpio:
            habilidades_encontradas.add(habilidad)
    return habilidades_encontradas

def calcular_similitud_tfidf(cv_texto, vacantes):
    """Calcula la similitud coseno entre el texto del CV y la descripción de cada vacante."""
    documentos = [cv_texto] + [v['descripcion'] for v in vacantes]
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(documentos)
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])
    scores = cosine_sim[0]
    tfidf_scores = {}
    for i, score in enumerate(scores):
        vacante_id = vacantes[i]['id']
        tfidf_scores[vacante_id] = score
    return tfidf_scores


# ===============================
# 🚀 APLICACIÓN PRINCIPAL
# ===============================

class CogniLinkApp:
    """Clase principal que maneja la aplicación Streamlit."""
    
    def __init__(self):
        self.ui = UIComponents()
        self.data_service = DataService()
        self.df_egresados, self.df_ofertas, self.df_habilidades = self.data_service.load_sample_data()
    
    def run(self):
        """Ejecuta la aplicación principal."""
        # Configuración de página
        st.set_page_config(
            page_title="CogniLink UNRC",
            page_icon="💼",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # Aplicar estilos
        self.ui.apply_custom_styles()
        
        # Header principal
        self.ui.create_main_header()
        
        # Estado de la sesión
        if SESSION_KEYS['user'] not in st.session_state:
            st.session_state[SESSION_KEYS['user']] = None
        
        # Mostrar login o dashboard según el estado
        if st.session_state[SESSION_KEYS['user']] is None:
            self.show_login_section()
        else:
            self.show_dashboard()
    
    def show_login_section(self):
        """Muestra la sección de login."""
        id_input, password_input, login_btn = self.ui.create_login_form()
        
        if login_btn:
            self.handle_login(id_input, password_input)
    
    def handle_login(self, id_input, password_input):
        """Maneja el proceso de login."""
        if not id_input or not password_input:
            st.error("❌ Por favor, completa todos los campos")
            return
        
        # Buscar usuario
        egresado = self.df_egresados[
            self.df_egresados['ID_Egresado'].astype(str) == id_input
        ]
        
        if not egresado.empty:
            # En una aplicación real, aquí verificarías la contraseña correctamente
            if egresado.iloc[0]['Nombre'].strip().lower() == password_input.strip().lower():
                st.session_state[SESSION_KEYS['user']] = egresado.iloc[0].to_dict()
                st.success(f"🎉 ¡Bienvenido/a, {egresado.iloc[0]['Nombre']}!")
                st.rerun()
            else:
                st.error("❌ Contraseña incorrecta")
        else:
            st.error("❌ ID de egresado no encontrado")
    
    def show_dashboard(self):
        """Muestra el dashboard principal después del login."""
        user_data = st.session_state[SESSION_KEYS['user']]
        
        # Header con opción de logout
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### 👋 Hola, {user_data['Nombre']}")
        with col2:
            if st.button("🚪 Cerrar Sesión", use_container_width=True):
                st.session_state.clear()
                st.rerun()
        
        # Perfil del usuario
        self.ui.create_user_profile_card(user_data)
        
        # Habilidades
        hard_skills = [h.strip() for h in user_data['Hard_Skills'].split(',')]
        soft_skills = [s.strip() for s in user_data['Soft_Skills'].split(',')]
        self.ui.create_skills_section(hard_skills, soft_skills)
        
        # Sección de análisis
        cv_text, analyze_btn = self.ui.create_analysis_section()
        
        # Manejar análisis
        if analyze_btn:
            if not cv_text.strip():
                st.error("❌ Por favor, ingresa el texto de tu CV")
            else:
                with st.spinner("🔍 Analizando tu CV y buscando oportunidades..."):
                    results = self.data_service.analyze_cv_with_api(cv_text)
                    st.session_state[SESSION_KEYS['results']] = results
                    st.session_state[SESSION_KEYS['cv_text']] = cv_text
        
        # Mostrar resultados si existen
        if SESSION_KEYS['results'] in st.session_state:
            self.ui.create_results_section(st.session_state[SESSION_KEYS['results']])


# ===============================
# 🚀 PUNTO DE ENTRADA PRINCIPAL
# ===============================

if __name__ == "__main__":
    print("🎯 Iniciando aplicación Streamlit")
    app_instance = CogniLinkApp()
    app_instance.run()