import streamlit as st
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd # Para mostrar los resultados de manera más amigable

# --- CARGAR MOCK DATA ---
# Cargar datos una sola vez para que no se recarguen cada vez que Streamlit re-ejecuta el script
@st.cache_data
def load_data():
    try:
        with open('vacantes.json', 'r', encoding='utf-8') as f:
            vacantes = json.load(f)
        with open('cursos.json', 'r', encoding='utf-8') as f:
            cursos = json.load(f)
    except FileNotFoundError:
        st.error("Archivos 'vacantes.json' o 'cursos.json' no encontrados. Asegúrate de que estén en el mismo directorio.")
        vacantes = []
        cursos = []
    return vacantes, cursos

VACANTES, CURSOS = load_data()

# --- FUNCIONES DE NLP SIMPLIFICADO MEJORADO ---

def normalizar_habilidad(habilidad):
    """Limpia la habilidad y maneja sinónimos básicos y versiones."""
    habilidad = habilidad.lower().strip()
    
    # 1. Normalizar sinónimos clave y términos compuestos
    if 'estadistica' in habilidad:
        return 'estadística'
    if 'trabajo en equipo' in habilidad or 'equipo' in habilidad:
        return 'trabajo en equipo'
    if 'resolución' in habilidad and 'problemas' in habilidad:
        return 'resolución de problemas'
    
    # 2. Manejar versiones o términos compuestos 
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

# --- NUEVO MODELO AVANZADO DE NLP (TF-IDF) ---

def calcular_similitud_tfidf(cv_texto, vacantes):
    """Calcula la similitud coseno entre el texto del CV y la descripción de cada vacante."""
    
    documentos = [cv_texto] + [v['descripcion'] for v in vacantes]
    
    # 1. Vectoriza los documentos (TF-IDF)
    # CORRECCIÓN: 'english' se usa como placeholder válido, ya que 'spanish' falló.
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True) 
    tfidf_matrix = vectorizer.fit_transform(documentos)
    
    # 2. Calcula la similitud coseno 
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])
    
    scores = cosine_sim[0]
    
    # Crea un diccionario {id_vacante: score_tfidf}
    tfidf_scores = {}
    for i, score in enumerate(scores):
        vacante_id = vacantes[i]['id']
        tfidf_scores[vacante_id] = score
        
    return tfidf_scores

# --- FUNCIÓN PRINCIPAL DE MATCHING ---

def perform_matching(cv_texto):
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
        
        # Score de Cumplimiento de Requisitos (Peso 60%)
        total_req = len(req_totales)
        score_cumplimiento = len(habilidades_cumplidas) / total_req if total_req else 0
        
        # Score de Relevancia Semántica (TF-IDF - Peso 40%)
        score_relevancia = tfidf_scores.get(vacante['id'], 0)
        
        # Fusión de scores para robustez
        puntaje_final = (score_cumplimiento * 0.6) + (score_relevancia * 0.4)
        
        # 3. Recomendación de Cursos
        cursos_recomendados_para_vacante = [
            curso for curso in CURSOS 
            if normalizar_habilidad(curso['habilidad']) in habilidades_faltantes
        ]

        resultados.append({
            "vacante": vacante,
            "puntaje_match": round(puntaje_final * 100, 2), # Ahora es más robusto
            "habilidades_cumplidas": list(habilidades_cumplidas),
            "habilidades_faltantes": list(habilidades_faltantes),
            "cursos_recomendados": cursos_recomendados_para_vacante
        })

    # 4. Ordenar resultados por mejor match
    resultados_ordenados = sorted(resultados, key=lambda x: x['puntaje_match'], reverse=True)
    
    return resultados_ordenados

# --- INTERFAZ DE STREAMLIT ---

st.set_page_config(page_title="CV Matcher y Recomendador de Cursos", layout="wide")

st.title("Buscador de Vacantes y Recomendador de Cursos")
st.markdown("Copia y pega el texto de tu CV para encontrar las vacantes más adecuadas y cursos que te ayudarán a cerrar brechas de habilidades.")

cv_texto_input = st.text_area("Pega aquí el texto completo de tu CV:", height=300)

if st.button("Buscar Vacantes"):
    if cv_texto_input:
        with st.spinner("Analizando CV y buscando vacantes..."):
            resultados = perform_matching(cv_texto_input)
        
        if resultados:
            st.subheader("Resultados de Match:")
            for i, res in enumerate(resultados):
                st.markdown(f"---")
                st.markdown(f"### {i+1}. Vacante: {res['vacante']['titulo']} (Match: {res['puntaje_match']}%)")
                st.write(f"**Empresa:** {res['vacante']['empresa']}")
                st.write(f"**Descripción:** {res['vacante']['descripcion']}")
                
                st.markdown(f"**Habilidades en tu CV que coinciden:**")
                if res['habilidades_cumplidas']:
                    st.success(", ".join(res['habilidades_cumplidas']))
                else:
                    st.info("No se encontraron habilidades que coincidan directamente.")
                
                st.markdown(f"**Habilidades requeridas que te faltan:**")
                if res['habilidades_faltantes']:
                    st.warning(", ".join(res['habilidades_faltantes']))
                else:
                    st.success("¡Parece que tienes todas las habilidades requeridas!")
                
                st.markdown(f"**Cursos recomendados para cerrar brechas:**")
                if res['cursos_recomendados']:
                    cursos_df = pd.DataFrame(res['cursos_recomendados'])
                    cursos_df.columns = ['ID', 'Título del Curso', 'Habilidad'] # Renombrar columnas para mejor visualización
                    st.dataframe(cursos_df[['Título del Curso', 'Habilidad']], hide_index=True)
                else:
                    st.info("No se encontraron cursos recomendados (¡ya tienes todas las habilidades!).")
        else:
            st.info("No se encontraron vacantes o no hay datos para procesar.")
    else:
        st.warning("Por favor, pega el texto de tu CV para empezar.")