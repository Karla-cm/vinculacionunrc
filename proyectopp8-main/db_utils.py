import pandas as pd
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://base_fjwm_user:herHQfSBfoUjEITVn33ePllUToGTsMVm@dpg-d46achshg0os73eesftg-a.oregon-postgres.render.com/base_fjwm"
engine = create_engine(DATABASE_URL)

def cargar_tabla(nombre_tabla):
    query = f"SELECT * FROM {nombre_tabla}"
    return pd.read_sql_query(query, engine)
