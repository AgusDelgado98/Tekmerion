# Tekmérion

**Evidence-first analysis of the Data, BI & AI job market.**

Tekmérion convierte vacantes laborales imperfectas en evidencia estructurada, reproducible y útil.

> Datos → procesamiento → evidencia → interpretación → decisión

No es un job board.  
No es un chatbot genérico.  
Es una pieza de portfolio y un sistema de análisis diseñado para demostrar competencias reales en datos, pipelines, testing y (próximamente) IA fundamentada.

---

## Estado actual (V0.3)

- Pipeline de procesamiento determinista y no-mutante
- Clasificación de **role family** y **seniority** por reglas explícitas
- Extracción y normalización de **skills** como dato estructurado
- Detección de duplicados por fingerprint de contenido
- **Capa de evidencia** reproducible: frecuencias, co-ocurrencias, distribuciones y comparación entre roles
- **Interfaz Flask** para explorar la evidencia (overview, skills, roles, comparación, co-ocurrencia)
- Suite de tests automatizados
- Muestra sintética de 17 registros para desarrollo y validación
- Documentación de metodología + Architecture Decision Records

Flujo actual:

```text
raw data → pipeline → structured evidence → Flask UI
```

Todavía **no** incluye IA generativa ni fuentes reales (Kaggle / Adzuna).

---

## Familias de roles soportadas

| Role Family       | Descripción breve                          |
|-------------------|--------------------------------------------|
| `data_analyst`    | Análisis de datos, KPIs, reporting         |
| `bi_analyst`      | Business Intelligence, dashboards, DAX     |
| `data_scientist`  | Modelado, estadística, ML experimental     |
| `ml_engineer`     | Producción de modelos, MLOps               |
| `ai_analyst`      | Evaluación de IA, LLMs, prompt engineering |
| `data_engineer`   | Pipelines, ETL, warehouses, infraestructura|
| `business_analyst`| Requisitos, procesos, análisis funcional   |

---

## Quick start

```bash
# Clonar (cuando esté en GitHub) o trabajar localmente
cd tekmerion

# Entorno
python -m venv .venv
source .venv/bin/activate   # o el equivalente en tu OS

pip install -e ".[dev]"

# Ejecutar tests
pytest -v

# Procesar la muestra sintética + generar evidencia
python scripts/run_pipeline.py
python scripts/run_evidence.py

# Iniciar la interfaz web
python -m app
# → http://127.0.0.1:5000
```

---

## Estructura del repositorio

```
tekmerion/
├── analysis/
│   ├── pipeline.py      # Orquestación del pipeline
│   ├── evidence.py      # Capa de evidencia y métricas
│   ├── models.py        # ProcessedJob, PipelineResult, enums
│   ├── classifiers.py   # Role family & seniority (reglas)
│   └── skills.py        # Extracción y normalización de skills
├── app/                 # Interfaz Flask (solo presentación)
│   ├── __init__.py
│   ├── routes.py
│   ├── templates/
│   └── static/
├── data/
│   ├── raw/             # Datos de entrada (muestra sintética)
│   └── processed/       # Salida del pipeline + evidence.json
├── docs/
│   ├── methodology.md   # Cómo funciona y por qué
│   └── adr/             # Architecture Decision Records
├── tests/
│   └── test_pipeline.py
├── pyproject.toml
└── README.md
```

---

## Principios de diseño

1. **Evidence first** — Las conclusiones salen de datos calculados, no de generación libre.
2. **Reproducibility** — Mismo input → mismo output.
3. **No mutation** — El pipeline nunca modifica los registros de entrada.
4. **Explicit rules** — Las clasificaciones son auditables en código.
5. **Small steps** — Cada capacidad se valida de forma aislada.
6. **Public quality** — El código debe poder mostrarse a recruiters y técnicos sin avergonzarse.

---

## Roadmap (alto nivel)

- [x] Pipeline core + tests + documentación base
- [x] Capa de evidencia reproducible (frecuencias, co-ocurrencias, diferencias entre roles)
- [x] Interfaz Flask sencilla para explorar evidencia
- [ ] Capa de IA generativa **grounded** (explica evidencia, no inventa)
- [ ] Integración con Adzuna API + datasets de Kaggle
- [ ] Informe / case study en PDF
- [ ] Publicación del repositorio

---

## Filosofía de IA (futura)

La IA es una capa de **interpretación**, no la fuente de verdad.

```
Datos procesados → métricas/evidencia → contexto estructurado → modelo generativo → explicación
```

Toda conclusión generada debe poder rastrearse hasta la evidencia que la sustenta.

---

## Licencia

MIT (provisional)

---

Construido como evidencia profesional pública.
