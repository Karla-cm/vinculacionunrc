import pandas as pd
from sqlalchemy import create_engine

# =======================================================
# 🔗 CONEXIÓN A TU BASE DE DATOS EN RENDER
# =======================================================
DATABASE_URL = "postgresql://base_fjwm_user:herHQfSBfoUjEITVn33ePllUToGTsMVm@dpg-d46achshg0os73eesftg-a.oregon-postgres.render.com/base_fjwm"

# Crear motor de conexión
engine = create_engine(DATABASE_URL)

# =======================================================
# 1️⃣ BASE DE DATOS DE EGRESADOS
# =======================================================
data_egresados = {
    'ID_Egresado': [1001, 1002, 1003, 1004, 1005],
    'Nombre': ['Sofía Casas', 'Daniela Espinosa', 'Andrés López', 'Mariana Rojas', 'Javier Soto'],
    'Anio_Egreso': [2024, 2023, 2022, 2025, 2023],
    'Rol_Deseado': [
        'Científico de Datos Junior',
        'Analista de Datos Senior',
        'Ingeniero de Machine Learning',
        'Investigadora en IA',
        'Consultor de Datos'
    ],
    'Experiencia_Anios': [0.5, 2.0, 3.5, 0.0, 1.5],
    'Hard_Skills': [
        'Python, Pandas, Sklearn, SQL Básico, Visualización de Datos (Matplotlib), Git',
        'R, Bases de Datos NoSQL (MongoDB), ETL, Tableau, Estadística Avanzada',
        'Python, TensorFlow, PyTorch, Docker, Kubernetes, CI/CD, AWS/Azure',
        'R, Estadística, Procesamiento de Señales, Simulación (MATLAB), LaTeX',
        'SQL, PowerBI, Excel Avanzado, Análisis Financiero, Presentaciones Ejecutivas'
    ],
    'Soft_Skills': [
        'Curiosidad, Adaptabilidad, Aprendizaje Rápido, Proactividad',
        'Liderazgo, Resolución de Problemas, Gestión de Proyectos, Comunicación',
        'Pensamiento Crítico, Autonomía, Detalle, Innovación',
        'Trabajo en Equipo, Ética Profesional, Disciplina, Organización',
        'Negociación, Orientación al Cliente, Comunicación, Persuasión'
    ],
    'Resumen_CV': [
        "Recién egresada con proyecto de tesis enfocado en regresión logística y clasificación. Busco una posición inicial que me permita crecer en un entorno de Big Data y aplicar mis conocimientos teóricos en Machine Learning.",
        "Dos años de experiencia liderando la migración de datos y optimización de bases de datos. Fuerte dominio en modelos estadísticos avanzados y reporting ejecutivo. Interés en roles de gestión de equipos de analítica.",
        "Ingeniero con amplia experiencia en la puesta en producción (MLOps) de modelos de Deep Learning. Experto en optimización de rendimiento y escalabilidad en la nube. Busco desafíos en sistemas distribuidos.",
        "Egresada de excelencia con enfoque en la investigación académica y modelos de inferencia. Interés en la aplicación de IA en el sector salud o medio ambiente. Dominio de métodos de validación robustos.",
        "Consultor con experiencia en el sector financiero, enfocado en traducir análisis complejos en estrategias de negocio accionables. Fuerte habilidad en la comunicación de insights a audiencias no técnicas."
    ]
}
df_egresados = pd.DataFrame(data_egresados)

# =======================================================
# 2️⃣ BASE DE DATOS DE OFERTAS
# =======================================================
data_ofertas = {
    'ID_Oferta': [501, 502, 503, 504],
    'Empresa': ['TechCorp Analytics', 'Data Innova Solutions', 'Gobierno Digital MX', 'FinTech Global'],
    'Puesto': ['Científico de Datos Jr.', 'Ingeniero de MLOps', 'Analista de Datos Público', 'Consultor Estratégico de Datos'],
    'Min_Exp_Anios': [1.0, 2.5, 0.0, 2.0],
    'Req_Hard_Skills': [
        'Python, SQL Avanzado, Modelos de Series de Tiempo',
        'TensorFlow, PyTorch, Docker, Kubernetes, AWS',
        'R, Estadística, PowerBI, Excel',
        'SQL, Análisis Financiero, Presentaciones Ejecutivas, PowerBI'
    ],
    'Req_Soft_Skills': [
        'Trabajo en Equipo, Resolución de Problemas, Comunicación',
        'Liderazgo, Autonomía, Detalle, Pensamiento Crítico',
        'Ética Profesional, Organización, Comunicación',
        'Negociación, Orientación al Cliente, Persuasión'
    ],
    'Descripcion_Puesto': [
        "Buscamos un Científico de Datos Junior con al menos 1 año de experiencia en el manejo de grandes volúmenes de datos. Se requiere dominio de Python y SQL para la extracción, limpieza y modelado predictivo. Valoramos fuertemente la capacidad de comunicar resultados de manera clara y trabajar en equipo.",
        "Se requiere un Ingeniero de Machine Learning con experiencia en despliegue de modelos en la nube (AWS o Azure). El candidato ideal debe ser autónomo y tener un gran detalle en la implementación de pipelines de CI/CD para modelos de Deep Learning.",
        "Vacante para recién egresados sin experiencia requerida. Se valorará el dominio de R y la estadística para el análisis de indicadores sociales. Es fundamental la ética profesional y la comunicación efectiva de resultados.",
        "Rol de consultoría en el sector financiero. El candidato debe ser experto en SQL y PowerBI para generar reportes y tener sólidas habilidades de negociación y persuasión para presentar recomendaciones a nivel ejecutivo."
    ]
}
df_ofertas = pd.DataFrame(data_ofertas)

# =======================================================
# 3️⃣ BASE DE DATOS DE HABILIDADES
# =======================================================
data_habilidades = {
    'Tipo': ['Hard', 'Hard', 'Hard', 'Hard', 'Soft', 'Soft', 'Soft', 'Soft'],
    'Habilidad': ['Python', 'SQL', 'Machine Learning', 'TensorFlow', 'Comunicación', 'Liderazgo', 'Trabajo en Equipo', 'Autonomía'],
    'Sinonimos': [
        'Piton, Python 3.x, programación en Python',
        'Base de datos SQL, MySQL, PostgreSQL, Transact-SQL',
        'ML, Aprendizaje Automático, modelos predictivos',
        'TF, Keras, TFlow',
        'Comunicación efectiva, Habilidades de presentación, Reporting',
        'Liderar equipos, Gestión de personas',
        'Colaboración, Espíritu de equipo, trabajo colaborativo',
        'Independencia, Iniciativa propia'
    ]
}
df_habilidades = pd.DataFrame(data_habilidades)

# =======================================================
# 💾 4️⃣ GUARDAR CSV Y SUBIR A POSTGRESQL
# =======================================================
try:
    # Guardar CSV
    df_egresados.to_csv('egresados_data.csv', index=False, encoding='utf-8')
    df_ofertas.to_csv('ofertas_data.csv', index=False, encoding='utf-8')
    df_habilidades.to_csv('habilidades_referencia.csv', index=False, encoding='utf-8')
    print("✅ Archivos CSV generados correctamente.")

    # Subir a PostgreSQL
    df_egresados.to_sql('egresados', engine, if_exists='replace', index=False)
    df_ofertas.to_sql('ofertas', engine, if_exists='replace', index=False)
    df_habilidades.to_sql('habilidades', engine, if_exists='replace', index=False)
    print("✅ Tablas subidas correctamente a Render PostgreSQL.")

    print("\n🎉 Proceso completado con éxito.")
    print("Tablas disponibles: egresados, ofertas, habilidades")

except Exception as e:
    print("❌ Error durante el proceso:", e)
