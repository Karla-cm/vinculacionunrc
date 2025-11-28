from flask import Flask, request, jsonify
from flask_cors import CORS 
import json
import re
# Importaciones para el nuevo modelo de NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 

# --- CONFIGURACIÓN DE FLASK ---
app = Flask(__name__)
CORS(app) 

# --- CARGAR MOCK DATA ---
try:
    with open('vacantes.json', 'r', encoding='utf-8') as f:
        VACANTES = json.load(f)
    
    with open('cursos.json', 'r', encoding='utf-8') as f:
        CURSOS = json.load(f)
except FileNotFoundError:
    VACANTES = []
    CURSOS = []

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


# --- ENDPOINTS DE LA API (ACTUALIZADO CON DOS MODELOS) ---

@app.route('/aplicar', methods=['POST'])
def aplicar_vacante():
    """Recibe el texto del CV, hace el match con dos modelos y devuelve recomendaciones."""
    data = request.json
    cv_texto = data.get('cv_texto', '')
    
    if not cv_texto:
        return jsonify({"error": "Debe enviar 'cv_texto' en la solicitud."}), 400

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
        cursos_recomendados = [
            curso for curso in CURSOS 
            if normalizar_habilidad(curso['habilidad']) in habilidades_faltantes
        ]

        resultados.append({
            "vacante": vacante,
            "puntaje_match": round(puntaje_final * 100, 2), # Ahora es más robusto
            "habilidades_cumplidas": list(habilidades_cumplidas),
            "habilidades_faltantes": list(habilidades_faltantes),
            "cursos_recomendados": cursos_recomendados
        })

    # 4. Ordenar resultados por mejor match
    resultados_ordenados = sorted(resultados, key=lambda x: x['puntaje_match'], reverse=True)
    
    return jsonify(resultados_ordenados)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)