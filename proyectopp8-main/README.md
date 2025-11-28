API para vincular egresados de la Universidad Nacional Rosario Castellanos (Lic. en Ciencias en Datos para Negocios) con empresas.

Rápido inicio (local):

1. Crear un entorno virtual e instalar dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements-api.txt
```

2. Iniciar la API:

```powershell
uvicorn backend.main:app --reload
```

Endpoints principales:
- POST /egresados/  -> crear egresado
- GET  /egresados/  -> listar egresados (filtro por `q` para habilidades)
- GET  /egresados/{id} -> obtener egresado
- POST /empresas/ -> crear empresa
- GET  /empresas/ -> listar empresas

Autenticación (JWT) - flujo rápido:

1. Registrar empresa (POST /empresas/) con campo `password`.
2. Solicitar token (POST /token) usando `username` (email) y `password`.
3. Llamar endpoints protegidos con Header: `Authorization: Bearer <token>`.

Ejemplo con curl:

```bash
curl -X POST "http://127.0.0.1:8000/token" -d "username=empresa@ejemplo.com&password=tu_password"
```

Notas:
- Este es un MVP inicial. Falta añadir autenticación JWT, gestión de archivos (CVs), consentimiento y pruebas.
