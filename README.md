# Agente de Investigación Bibliográfica para Psicología Clínica

> Un agente de IA que busca, filtra y selecciona los papers más relevantes para casos clínicos de psicología, buscando simultáneamente en múltiples bases de datos académicas en inglés y español.

---

## El problema

Los psicólogos clínicos latinoamericanos dedican horas a buscar bibliografía relevante para sus casos. Las bases de datos científicas están fragmentadas, la mayoría del contenido está en inglés, y encontrar literatura contextualizada para la realidad latinoamericana requiere buscar en múltiples fuentes manualmente.

## La solución

Un agente de IA que toma una bitácora clínica redactada en español, traduce los conceptos clave al inglés técnico, y ejecuta búsquedas paralelas en **5 fuentes académicas** simultáneamente. Devuelve los **3 papers más relevantes** con justificación clínica, cita APA y descarga en PDF/CSV.

---

## Arquitectura

```
┌──────────────────────────────────────────────────────────────┐
│                    Bitácora clínica (ES)                      │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              Sub-agente extractor de queries                  │
│    (Gemini) → extrae términos en INGLÉS + ESPAÑOL            │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│               Búsqueda paralela en 5 APIs                    │
│                                                              │
│   ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌───────────┐      │
│   │ PubMed  │ │ Semantic │ │ OpenAlex  │ │ OpenAlex  │      │
│   │ (EN)    │ │ Scholar  │ │ (EN)      │ │ (ES)      │      │
│   │         │ │ (EN)     │ │           │ │           │      │
│   └─────────┘ └──────────┘ └───────────┘ └───────────┘      │
│                                                              │
│                      ┌───────────┐                           │
│                      │ CrossRef  │                           │
│                      │ (ES)      │                           │
│                      └───────────┘                           │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  Deduplicación por DOI + ranking algorítmico (top 50%)      │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│         Agente principal (Gemini) → selecciona 3 papers      │
│    con justificación clínica en español                       │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│   Interfaz web (NiceGUI)  ·  CLI  ·  PDF  ·  CSV            │
└──────────────────────────────────────────────────────────────┘
```

---

## Características principales

- **Búsqueda bilingüe** — Extrae términos del caso clínico en español y genera automáticamente búsquedas en inglés técnico y en español original, maximizando la cobertura bibliográfica.
- **5 fuentes en paralelo** — PubMed, Semantic Scholar, OpenAlex (inglés y español) y CrossRef. Tolerante a fallos: si una fuente no responde, continúa con las demás.
- **CrossRef para Latinoamérica** — CrossRef indexa revistas de SciELO y otras editoriales iberoamericanas, proporcionando acceso a bibliografía contextualizada para la región.
- **Ranking inteligente** — Dos fases: primero un ranking algorítmico por coincidencia de términos (ahorra tokens), luego el LLM selecciona los 3 más relevantes con justificación clínica.
- **Deduplicación** — Papers encontrados en múltiples fuentes se fusionan automáticamente, combinando las fuentes de origen.
- **Justificación clínica** — Cada paper incluye una explicación en español de por qué es relevante para el caso presentado.
- **Descarga PDF y CSV** — Los resultados se pueden exportar con un clic desde la interfaz web.
- **Modelos configurables** — Compatible con cualquier modelo soportado por pydantic-ai (Gemini, OpenAI, Anthropic, etc.), configurable por variables de entorno.

---

## Interfaz web

La aplicación incluye una interfaz web construida con [NiceGUI](https://nicegui.io/), que ofrece:

- Área de texto para pegar la bitácora clínica
- Búsqueda asíncrona nativa (sin recargas de página)
- Cards visuales para cada paper con badges de color por fuente
- Títulos y DOIs clickeables
- Abstract original en panel expandible
- Botones de descarga PDF y CSV

---

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| Framework de agente | [pydantic-ai](https://github.com/pydantic/pydantic-ai) |
| LLM | Google Gemini (configurable) |
| Interfaz web | [NiceGUI](https://nicegui.io/) (FastAPI + Vue.js) |
| HTTP asíncrono | [httpx](https://github.com/encode/httpx) |
| Modelos de datos | [Pydantic](https://docs.pydantic.dev/) |
| Generación PDF | [fpdf2](https://github.com/reingart/pyfpdf) |
| Gestión de entorno | [uv](https://github.com/astral-sh/uv) |

---

## Instalación

### Requisitos

- Python 3.14+
- Una API key de Google Gemini (gratuita en [AI Studio](https://aistudio.google.com/app/apikey))

### Pasos

```bash
# Clonar el repositorio
git clone https://github.com/RubenYuberth/Agente-Investigacion-Psicologia-Clinica.git
cd Agente-Investigacion-Psicologia-Clinica

# Instalar dependencias
uv sync

# Configurar variables de entorno
cp .env.example .env
# Editar .env y agregar tu GOOGLE_API_KEY
```

### Variables de entorno

| Variable | Requerida | Descripción |
|---|---|---|
| `GOOGLE_API_KEY` | Sí | API key de Google Gemini |
| `SEMANTIC_SCHOLAR_API_KEY` | No | Aumenta rate limits en Semantic Scholar |
| `CROSSREF_MAILTO` | No | Email de contacto para CrossRef (evita rate limiting) |
| `MODEL_EXTRACTOR` | No | Modelo para el extractor de queries (default: `google:gemini-3.1-flash-lite-preview`) |
| `MODEL_INVESTIGACION` | No | Modelo para el agente principal (default: `google:gemini-3.1-flash-lite-preview`) |

---

## Uso

### Interfaz web

```bash
python app.py
```

Se abrirá automáticamente en `http://localhost:8080`.

### CLI

```bash
python main.py
```

Pegar la bitácora clínica y presionar `Ctrl+D` (Linux/Mac) o `Ctrl+Z` (Windows) para procesar.

---

## Ejemplo de salida

```
=== RESULTADOS: 3 papers encontrados ===

--- Paper 1 ---
Título: Tratamientos psicológicos empíricamente apoyados para el TDAH en adultos
Autores: Calle Chamorro, Allison Catalina, Reivan Ortiz, Geovanny Genaro
Año: 2025
DOI: 10.46652/rgn.v11i49.1582
Revista: Religación
Fuentes: crossref
Justificación: Esta revisión sistemática reciente evalúa específicamente la
eficacia de la TCC en adultos con TDAH, documentando su impacto positivo en la
regulación emocional y síntomas centrales, lo cual es fundamental para el
manejo clínico del paciente.
```

---

## Estructura del proyecto

```
.
├── agentes.py        # Agente principal, modelos de datos, lógica de búsqueda
├── app.py            # Interfaz web con NiceGUI
├── main.py           # CLI para uso desde terminal
├── .env.example      # Plantilla de variables de entorno
├── pyproject.toml    # Dependencias y configuración del proyecto
└── uv.lock           # Lockfile de dependencias
```

---

## Licencia

MIT

---

## Autor

**Ruben Yuberth** — [GitHub](https://github.com/RubenYuberth)
