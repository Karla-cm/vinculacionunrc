from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Tuple
import pandas as pd
import streamlit as st
import json
import re

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
        """Agrega una habilidad si no existe."""
        if habilidad.lower() not in [h.lower() for h in self.habilidades]:
            self.habilidades.append(habilidad)

# ===============================
# 📊 FUNCIONES DE ANÁLISIS Y NLP
# ===============================

def normalizar_habilidad(habilidad):
    """Limpia la habilidad y maneja sinónimos básicos y versiones."""
    habilidad = habilidad.lower().strip()
    
    # Normalizar sinónimos clave y términos compuestos
    if 'estadistica' in habilidad:
        return 'estadística'
    if 'trabajo en equipo' in habilidad or 'equipo' in habilidad:
        return 'trabajo en equipo'
    if 'resolución' in habilidad and 'problemas' in habilidad:
        return 'resolución de problemas'
    
    # Manejar versiones o términos compuestos 
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
    
    # Vectoriza los documentos (TF-IDF)
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(documentos)
    
    # Calcula la similitud coseno 
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])
    scores = cosine_sim[0]
    
    # Crea un diccionario {id_vacante: score_tfidf}
    tfidf_scores = {}
    for i, score in enumerate(scores):
        vacante_id = vacantes[i]['id']
        tfidf_scores[vacante_id] = score
        
    return tfidf_scores

def aplicar_vacante_streamlit(cv_texto, VACANTES, CURSOS):
    """Versión Streamlit del endpoint /aplicar - procesa CV y devuelve recomendaciones."""
    if not cv_texto:
        return []
    
    resultados = []
    
    # MODELO 1: Extracción de Habilidades (Base para Brechas)
    todas_habilidades = set()
    for v in VACANTES:
        todas_habilidades.update(v['requisitos_tecnicos'])
        todas_habilidades.update(v['requisitos_blandos'])

    habilidades_cv = extraer_habilidades(cv_texto, todas_habilidades)
    
    # MODELO 2: Similitud Coseno con TF-IDF (Score de Relevancia Semántica)
    tfidf_scores = calcular_similitud_tfidf(cv_texto, VACANTES)

    for vacante in VACANTES:
        req_tec = set(normalizar_habilidad(h) for h in vacante['requisitos_tecnicos'])
        req_blando = set(normalizar_habilidad(h) for h in vacante['requisitos_blandos'])
        req_totales = req_tec.union(req_blando)
        
        habilidades_cumplidas = habilidades_cv.intersection(req_totales)
        habilidades_faltantes = req_totales - habilidades_cv

        # Cálculo del Score FINAL (Combinación de los dos modelos)
        total_req = len(req_totales)
        score_cumplimiento = len(habilidades_cumplidas) / total_req if total_req else 0
        
        # Score de Relevancia Semántica (TF-IDF - Peso 40%)
        score_relevancia = tfidf_scores.get(vacante['id'], 0)
        
        # Fusión de scores para robustez
        puntaje_final = (score_cumplimiento * 0.6) + (score_relevancia * 0.4)
        
        # Recomendación de Cursos
        cursos_recomendados = [
            curso for curso in CURSOS 
            if normalizar_habilidad(curso['habilidad']) in habilidades_faltantes
        ]

        resultados.append({
            "vacante": vacante,
            "puntaje_match": round(puntaje_final * 100, 2),
            "habilidades_cumplidas": list(habilidades_cumplidas),
            "habilidades_faltantes": list(habilidades_faltantes),
            "cursos_recomendados": cursos_recomendados
        })

    # Ordenar resultados por mejor match
    resultados_ordenados = sorted(resultados, key=lambda x: x['puntaje_match'], reverse=True)
    
    return resultados_ordenados

# ===============================
# 🚀 APLICACIÓN STREAMLIT
# ===============================

def main():
    # Configuración de la página
    st.set_page_config(page_title="CogniLink UNRC", layout="wide")
    
    # Cargar datos (mock data o desde archivos)
    try:
        with open('vacantes.json', 'r', encoding='utf-8') as f:
            VACANTES = json.load(f)
        with open('cursos.json', 'r', encoding='utf-8') as f:
            CURSOS = json.load(f)
    except FileNotFoundError:
        # Datos de ejemplo si no hay archivos
        VACANTES = [
            {
                "id": 1,
                "titulo": "Data Scientist",
                "empresa": "Tech Solutions",
                "descripcion": "Buscamos científico de datos con experiencia en Python y machine learning",
                "requisitos_tecnicos": ["Python", "SQL", "Machine Learning", "Estadística"],
                "requisitos_blandos": ["Trabajo en equipo", "Comunicación"]
            },
            {
                "id": 2,
                "titulo": "Desarrollador Full Stack",
                "empresa": "Digital Labs",
                "descripcion": "Desarrollador con conocimientos en JavaScript, Node.js y React",
                "requisitos_tecnicos": ["JavaScript", "Node.js", "React", "SQL"],
                "requisitos_blandos": ["Creatividad", "Resolución de problemas"]
            }
        ]
        CURSOS = [
            {"habilidad": "Python", "titulo_curso": "Curso intensivo de Python", "proveedor": "Coursera"},
            {"habilidad": "SQL", "titulo_curso": "Bases de Datos SQL", "proveedor": "edX"},
            {"habilidad": "Machine Learning", "titulo_curso": "ML Avanzado", "proveedor": "Udemy"}
        ]

    # Estilos CSS
    st.markdown(r'''
    <style>
    body { background-color: #f5faff; }
    .main-card {
        background: linear-gradient(90deg, #0a1f2e 60%, #00e6e6 100%);
        border-radius: 20px;
        box-shadow: 0 4px 24px #0a1f2e22;
        padding: 2.5rem;
        margin-top: 2rem;
        color: #fff;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    .main-card h1 {
        color: #00e6e6;
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
    }
    .vacante-card {
        background: #fff;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px #0a1f2e22;
        border-left: 4px solid #00e6e6;
    }
    .habilidad-cumplida {
        background: #00e6e6;
        color: #0a1f2e;
        border-radius: 6px;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        display: inline-block;
        font-weight: 500;
    }
    .habilidad-faltante {
        background: #ff6b6b;
        color: white;
        border-radius: 6px;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        display: inline-block;
        font-weight: 500;
    }
    .curso-card {
        background: #e6f7ff;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #00e6e6;
    }
    </style>
    ''', unsafe_allow_html=True)

    # Header principal con menú interactivo
    st.markdown(r'''
    <div class='main-card'>
        <h1>CogniLink UNRC</h1>
        <p style='font-size:1.2rem;'>Sistema inteligente de vinculación laboral para egresados UNRC.<br>Analiza tu CV y encuentra las mejores oportunidades.</p>
    </div>
    ''', unsafe_allow_html=True)

    # Menú de navegación
    menu = st.sidebar.radio("Navegación", ["Inicio", "Analizar CV", "Vacantes", "Cursos recomendados", "Acerca de"])

    if menu == "Inicio":
        st.markdown("""
        ### Bienvenido a CogniLink UNRC
        - Sube tu CV y obtén recomendaciones personalizadas de vacantes.
        - Descubre cursos para mejorar tus oportunidades.
        - Visualiza el match entre tu perfil y las ofertas laborales.
        """)

    elif menu == "Analizar CV":
        st.subheader("🔎 Analiza tu CV")
        st.write("Sube tu CV en texto o pégalo en el cuadro de abajo para obtener recomendaciones de vacantes y cursos.")
        cv_texto = st.text_area("Pega aquí el texto de tu CV", height=200)
        analizar = st.button("Analizar CV", type="primary")
        if analizar and cv_texto:
            resultados = aplicar_vacante_streamlit(cv_texto, VACANTES, CURSOS)
            if resultados:
                st.success(f"Se encontraron {len(resultados)} vacantes recomendadas.")
                for res in resultados:
                    vac = res["vacante"]
                    with st.container():
                        st.markdown(f"<div class='vacante-card'><b>{vac['titulo']}</b> en <i>{vac['empresa']}</i><br>\n<p style='color:#0a1f2e;'>{vac['descripcion']}</p>", unsafe_allow_html=True)
                        st.markdown(f"<b>Puntaje de match:</b> <span style='color:#00e6e6;font-size:1.2rem;'>{res['puntaje_match']}%</span>", unsafe_allow_html=True)
                        st.markdown("<b>Habilidades cumplidas:</b> " + ''.join([f"<span class='habilidad-cumplida'>{h}</span>" for h in res['habilidades_cumplidas']]), unsafe_allow_html=True)
                        st.markdown("<b>Habilidades faltantes:</b> " + ''.join([f"<span class='habilidad-faltante'>{h}</span>" for h in res['habilidades_faltantes']]), unsafe_allow_html=True)
                        if res['cursos_recomendados']:
                            st.markdown("<b>Cursos recomendados:</b>")
                            for curso in res['cursos_recomendados']:
                                st.markdown(f"<div class='curso-card'><b>{curso['titulo_curso']}</b> - {curso['proveedor']}</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.button(f"Postularme a {vac['titulo']}", key=f"btn_{vac['id']}")
            else:
                st.warning("No se encontraron vacantes que coincidan con tu perfil.")
        elif analizar:
            st.error("Por favor, ingresa el texto de tu CV.")

    elif menu == "Vacantes":
        st.subheader("💼 Vacantes disponibles")
        for vac in VACANTES:
            with st.expander(f"{vac['titulo']} en {vac['empresa']}"):
                st.write(vac['descripcion'])
                st.markdown("<b>Requisitos técnicos:</b> " + ', '.join(vac['requisitos_tecnicos']), unsafe_allow_html=True)
                st.markdown("<b>Requisitos blandos:</b> " + ', '.join(vac['requisitos_blandos']), unsafe_allow_html=True)
                st.button(f"Postularme a {vac['titulo']}", key=f"post_{vac['id']}")

    elif menu == "Cursos recomendados":
        st.subheader("📚 Cursos recomendados")
        for curso in CURSOS:
            st.markdown(f"<div class='curso-card'><b>{curso['titulo_curso']}</b> - {curso['proveedor']}<br><i>Habilidad: {curso['habilidad']}</i></div>", unsafe_allow_html=True)

    elif menu == "Acerca de":
        st.subheader("ℹ️ Acerca de CogniLink UNRC")
        st.write("""
        CogniLink UNRC es una plataforma desarrollada para facilitar la vinculación laboral de egresados de la Universidad Nacional de Río Cuarto. Utiliza inteligencia artificial para analizar perfiles y recomendar oportunidades laborales y de formación.
        """)