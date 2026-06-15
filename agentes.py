from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
import asyncio
from pydantic import BaseModel, Field, HttpUrl
import httpx
import logging
import os
import functools
import xml.etree.ElementTree as ET

load_dotenv()

# Configurar logging para debugging de errores de APIs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# DEFINICIÓN DE MODELOS DE DATOS
# ═══════════════════════════════════════════════════════════

class SearchQueryModel(BaseModel):
    """Modelo estructurado para los parámetros de búsqueda bibliográfica.
    
    El agente extrae estos términos de la bitácora clínica en español,
    pero DEBE traducirlos al inglés técnico para que las APIs funcionen.
    """
    keywords: list[str] = Field(
        ..., 
        description="Palabras clave principales en INGLÉS, extraídas del caso clínico. Ejemplo: ['depression', 'cognitive behavioral therapy', 'adolescents']"
    )
    synonyms: list[str] = Field(
        default=[], 
        description="Sinónimos o términos relacionados en INGLÉS que amplíen la búsqueda. Ejemplo: ['major depressive disorder', 'CBT', 'teenagers']"
    )
    concepts: list[str] = Field(
        default=[], 
        description="Conceptos clínicos generales en INGLÉS. Ejemplo: ['psychotherapy', 'mental health']"
    )
    year_range: tuple[int, int] | None = Field(
        default=None, 
        description="Rango de años de publicación si es relevante. Ejemplo: (2020, 2024)"
    )
    open_access_only: bool = Field(
        default=False, 
        description="Si se deben priorizar solo artículos de acceso abierto"
    )
    study_types: list[str] = Field(
        default=[], 
        description="Tipos de estudio relevantes en INGLÉS. Ejemplo: ['RCT', 'meta-analysis', 'systematic review', 'cohort study']"
    )


class Paper(BaseModel):
    """Modelo de salida final para un artículo científico relevante.
    
    El agente debe mapear los resultados de las APIs a este formato estructurado.
    """
    titulo: str = Field(
        description="Título exacto del artículo científico en su idioma original (generalmente inglés)."
    )
    autores: list[str] = Field(
        description="Lista de autores del artículo científico, comenzando por el primer autor. Cada autor debe estar en formato 'Apellido, Nombre'."
    )
    anio: int = Field(
        description="Año de publicación del artículo científico."
    )
    doi: str | None = Field(
        default=None,
        description="DOI oficial del artículo científico. Debe tener formato estándar de DOI. Ejemplo: '10.1000/xyz123' sin URL.",
        examples=["10.1000/xyz123", "10.1037/0003-066X.59.1.29", "10.1000/182"]
    )
    abstract: str | None = Field(
        default=None,
        description="Resumen del artículo científico en su idioma original (generalmente inglés). Abstract original del paper, sin traducir. No inventar contenido. Si no existe, devolver null."
    )
    journal: str | None = Field(
        default=None,
        description="Nombre de la revista donde se publicó el artículo científico."
    )
    url: HttpUrl | None = Field(
        default=None,
        description="URL del artículo científico."
    )
    citas: int | None = Field(
        default=None,
        description="Número de citas del artículo científico. Si no se conoce, devolver null."
    )
    referencia: str | None = Field(
        default=None,
        description="Referencia del artículo científico en APA 7ma edición, con formato: Apellido, A. A. (Año). Título del artículo. Nombre de la Revista, Volumen(Número), páginas. DOI o URL.",
        examples=[
            "Smith, J. (2020). The impact of AI on society. Journal of AI Research, 15(3), 123-145. https://doi.org/10.1000/xyz123",
            "Doe, A., & Roe, B. (2019). Advances in machine learning. AI Review, 10(2), 50-75. https://doi.org/10.1037/0003-066X.59.1.29"
        ]
    )
    # ---- CAMPOS NUEVOS PARA TRAZABILIDAD ----
    source_apis: list[str] = Field(
        default=[],
        description="APIs de búsqueda donde fue encontrado este artículo. Ejemplo: ['openalex', 'pubmed']"
    )
    raw_data: dict | None = Field(
        default=None,
        description="Datos crudos originales de la API para debugging y auditoría."
    )


# ═══════════════════════════════════════════════════════════
# CONFIGURACIÓN DEL AGENTE (Lazy Initialization)
# ═══════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Eres un agente de investigación científica especializado en psicología clínica. 
Tu objetivo es ayudar a psicólogos a encontrar papers relevantes para sus casos clínicos.

INSTRUCCIONES CRÍTICAS:
1. El usuario te enviará una bitácora clínica, resumen de sesión o apuntes del paciente en ESPAÑOL.
2. Tu PRIMER paso SIEMPRE debe ser usar la herramienta `extract_search_query` para analizar el texto y extraer términos de búsqueda estructurados.
3. Todos los términos de búsqueda (keywords, synonyms, concepts, study_types) DEBEN estar en INGLÉS técnico, ya que las APIs de búsqueda científica trabajan en inglés.
4. NO inventes términos que no estén presentes o implícitos en el texto del usuario. Si hay duda, usa los términos más generales que aparecen en el caso.
5. Si el texto es ambiguo, usa los términos más comunes y ampliamente indexados en bases de datos psicológicas.
6. Tu SEGUNDO paso debe ser usar la herramienta `search_all_sources` con el SearchQueryModel generado.
7. La tool `search_all_sources` devolverá SOLO una selección pre-filtrada de los mejores papers (generalmente la mitad superior).
8. De los papers que recibas de `search_all_sources`, tu tarea final es seleccionar EXACTAMENTE LOS 2 PAPERS MÁS RELEVANTES para el caso clínico del usuario.
9. El output final DEBE ser una lista con EXACTAMENTE 2 objetos `Paper` (o menos, si no hay suficientes resultados; pero NUNCA más de 2).
10. Los abstracts deben permanecer en inglés (idioma original). No los traduzcas.
11. Si no encuentras resultados relevantes, devuelve una lista vacía.
12. Sé riguroso con los DOIs: solo incluye DOIs válidos, no inventes.

EJEMPLO DE FLUJO DE TRABAJO:
- Usuario: "Paciente de 12 años con TDAH y ansiedad social, tratamiento con metilfenidato"
- Paso 1: Llamas a extract_search_query(bitacora_clinica=...)
- Resultado: keywords=["ADHD", "social anxiety", "methylphenidate"], synonyms=["attention deficit hyperactivity disorder", "social phobia"], concepts=["pediatric", "psychopharmacology"]
- Paso 2: Llamas a search_all_sources(query_model=...)
- Resultado: list[Paper] con papers relevantes
"""

EXTRACTOR_PROMPT = """Eres un extractor de términos de búsqueda para bases de datos científicas.

Recibirás un texto clínico en ESPAÑOL y debes extraer los términos de búsqueda más relevantes.

REGLAS:
1. Todos los términos DEBEN estar en INGLÉS técnico.
2. Extrae keywords principales (términos centrales del caso).
3. Extrae sinónimos ampliamente usados en literatura científica.
4. Extrae conceptos generales del área (ej. psychotherapy, pediatrics).
5. Si el usuario menciona tipos de estudio (ej. "ensayo clínico"), inclúyelo en study_types.
6. NO inventes términos que no estén en el texto o implícitos en el contexto.
7. Si no hay información sobre años, deja year_range como null.

EJEMPLO:
Texto: "Paciente de 12 años con TDAH y ansiedad social, tratamiento con metilfenidato"
Resultado:
keywords: ["ADHD", "social anxiety", "methylphenidate"]
synonyms: ["attention deficit hyperactivity disorder", "social phobia"]
concepts: ["pediatric", "psychopharmacology"]
year_range: null
open_access_only: false
study_types: []
"""


# Agentes se inicializan lazy para evitar errores al importar el módulo sin API key
_agente_investigacion = None
_query_extractor = None


def _get_query_extractor() -> Agent:
    """Inicializa y devuelve el sub-agente extractor de queries."""
    global _query_extractor
    if _query_extractor is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY no está configurada. "
                "Configúrala en un archivo .env o como variable de entorno."
            )
        _query_extractor = Agent(
            'google:gemini-2.5-flash',
            system_prompt=EXTRACTOR_PROMPT,
            output_type=SearchQueryModel,
        )
    return _query_extractor


def _get_agente_investigacion() -> Agent:
    """Inicializa y devuelve el agente principal de investigación."""
    global _agente_investigacion
    if _agente_investigacion is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY no está configurada. "
                "Configúrala en un archivo .env o como variable de entorno."
            )
        _agente_investigacion = Agent(
            'google:gemini-2.5-flash',
            system_prompt=SYSTEM_PROMPT,
            output_type=list[Paper],
        )
        # Registrar tools con el agente
        _register_tools(_agente_investigacion)
    return _agente_investigacion


# ═══════════════════════════════════════════════════════════
# TOOLS (nivel de módulo para que pydantic-ai pueda
# generar los esquemas JSON correctamente)
# ═══════════════════════════════════════════════════════════

async def extract_search_query(ctx: RunContext[None], bitacora_clinica: str) -> SearchQueryModel:
    """Analiza una bitácora clínica, resumen de sesión o apuntes del paciente
    y extrae los parámetros de búsqueda bibliográfica estructurados.
    
    Esta tool usa un sub-agente especializado para procesar el texto clínico
    en español y generar un SearchQueryModel con términos en inglés técnico.
    
    Args:
        bitacora_clinica: Texto en español con la bitácora, resumen de sesión o apuntes del caso.
    
    Returns:
        SearchQueryModel con los términos de búsqueda extraídos y traducidos al inglés.
    """
    extractor = _get_query_extractor()
    result = await extractor.run(bitacora_clinica)
    return result.output


async def search_all_sources(ctx: RunContext[None], query_model: SearchQueryModel) -> list[Paper]:
    """Busca artículos científicos en paralelo en OpenAlex, PubMed y Semantic Scholar.
    
    Recibe un SearchQueryModel estructurado y realiza búsquedas en las tres APIs
    simultáneamente. Deduplica los resultados por DOI, tolera fallos de APIs
    individuales, mapea los resultados al modelo Paper, y finalmente aplica un
    ranking preliminar para devolver solo la mitad superior (ahorrando tokens
    al agente LLM que hará la selección final).
    
    Args:
        query_model: Objeto SearchQueryModel con los parámetros de búsqueda estructurados.
    
    Returns:
        Lista con la mitad superior de objetos Paper, ordenados por relevancia preliminar.
    """
    # Construir queries específicas para cada API
    openalex_query = _build_openalex_query(query_model)
    pubmed_query = _build_pubmed_query(query_model)
    semantic_query = _build_semantic_scholar_query(query_model)
    
    logger.info(f"OpenAlex query: {openalex_query}")
    logger.info(f"PubMed query: {pubmed_query}")
    logger.info(f"Semantic Scholar query: {semantic_query}")
    
    # Ejecutar búsquedas en paralelo con tolerancia a fallos
    tasks = [
        _search_with_fallback("openalex", _search_openalex, openalex_query),
        _search_with_fallback("pubmed", _search_pubmed, pubmed_query),
        _search_with_fallback("semantic_scholar", _search_semantic_scholar, semantic_query),
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Recopilar todos los papers crudos
    all_raw_papers = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Error en una de las búsquedas: {result}")
            continue
        all_raw_papers.extend(result)
    
    logger.info(f"Total papers crudos recolectados: {len(all_raw_papers)}")
    
    # Deduplicar y mapear
    papers = _deduplicate_and_map(all_raw_papers)
    logger.info(f"Total papers después de deduplicación: {len(papers)}")
    
    # Ranking preliminar: filtrar la mitad superior para ahorrar tokens al LLM
    papers = _rank_and_filter(papers, query_model)
    logger.info(f"Total papers después de filtrado (top 50%): {len(papers)}")
    
    return papers


def _register_tools(agent: Agent):
    """Registra las tools con el agente principal."""
    agent.tool(extract_search_query)
    agent.tool(search_all_sources)


# Creamos una clase wrapper para poder exportar como propiedad
class _AgenteInvestigacionWrapper:
    async def run(self, prompt, **kwargs):
        """Ejecuta el agente principal de forma asíncrona."""
        agent = _get_agente_investigacion()
        return await agent.run(prompt, **kwargs)
    
    def run_sync(self, prompt, **kwargs):
        """Ejecuta el agente principal de forma síncrona."""
        agent = _get_agente_investigacion()
        return agent.run_sync(prompt, **kwargs)

agente_investigacion = _AgenteInvestigacionWrapper()


# ═══════════════════════════════════════════════════════════
# CONSTRUCTORES DE QUERY (internos, no son tools)
# ═══════════════════════════════════════════════════════════

def _build_openalex_query(query: SearchQueryModel) -> str:
    """Construye una query string para la API de OpenAlex.
    
    OpenAlex usa búsqueda de texto libre con algunos filtros posibles.
    """
    parts = []
    
    # Términos principales y sinónimos
    all_terms = query.keywords + query.synonyms + query.concepts
    if all_terms:
        parts.append(" ".join(all_terms))
    
    # Filtros de año
    if query.year_range:
        from_year, to_year = query.year_range
        # Nota: OpenAlex soporta filtros en la URL, no en el search string
        # Los dejamos para los parámetros de la URL
    
    # Open access
    # También se maneja en parámetros de URL
    
    return " ".join(parts)


def _build_pubmed_query(query: SearchQueryModel) -> str:
    """Construye una query string para la API de PubMed (E-utilities).
    
    PubMed usa sintaxis AND/OR y campos específicos.
    """
    parts = []
    
    # Keywords y synonyms combinados con AND
    all_terms = query.keywords + query.synonyms
    if all_terms:
        # Envolver en paréntesis para agrupar
        terms_str = " OR ".join([f'"{t}"' for t in all_terms])
        parts.append(f"({terms_str})")
    
    # Concepts adicionales
    if query.concepts:
        concepts_str = " OR ".join([f'"{c}"' for c in query.concepts])
        parts.append(f"({concepts_str})")
    
    # Study types
    if query.study_types:
        study_str = " OR ".join([f'"{s}"' for s in query.study_types])
        parts.append(f"({study_str})")
    
    # Combinar todo con AND
    query_str = " AND ".join(parts)
    
    # Filtros de año (se manejan en parámetros separados de la API)
    
    return query_str


def _build_semantic_scholar_query(query: SearchQueryModel) -> str:
    """Construye una query string para la API de Semantic Scholar.
    
    Semantic Scholar usa búsqueda de texto libre.
    """
    parts = []
    
    all_terms = query.keywords + query.synonyms + query.concepts
    if all_terms:
        parts.append(" ".join(all_terms))
    
    if query.study_types:
        parts.append(" ".join(query.study_types))
    
    return " ".join(parts)


# ═══════════════════════════════════════════════════════════
# FUNCIONES DE BÚSQUEDA (internas, no son tools)
# ═══════════════════════════════════════════════════════════

async def _search_with_fallback(
    api_name: str, 
    search_func, 
    query: str, 
    per_page: int = 10
) -> list:
    """Wrapper que ejecuta una función de búsqueda y loguea errores sin romper el flujo."""
    try:
        return await search_func(query, per_page)
    except Exception as e:
        logger.error(f"Error en API {api_name}: {e}")
        return []


async def _search_openalex(query: str, per_page: int = 10) -> list[dict]:
    """Busca en OpenAlex y devuelve resultados crudos."""
    async with httpx.AsyncClient(timeout=30) as client:
        params = {
            "search": query,
            "per-page": per_page,
        }
        response = await client.get(
            "https://api.openalex.org/works",
            params=params
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        
        # Añadir metadato de fuente
        for r in results:
            r["_source_api"] = "openalex"
        
        return results


async def _search_pubmed(query: str, per_page: int = 10) -> list[dict]:
    """Busca en PubMed usando E-utilities (efetch) y devuelve resultados crudos con abstract."""
    async with httpx.AsyncClient(timeout=30) as client:
        # Paso 1: esearch para obtener IDs
        esearch_params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": per_page,
            "sort": "relevance"
        }
        esearch_response = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params=esearch_params
        )
        esearch_response.raise_for_status()
        esearch_data = esearch_response.json()
        
        idlist = esearch_data.get("esearchresult", {}).get("idlist", [])
        
        if not idlist:
            return []
        
        # Paso 2: efetch para obtener detalles completos (incluyendo abstract)
        # efetch devuelve XML por defecto
        efetch_params = {
            "db": "pubmed",
            "id": ",".join(idlist),
            "retmode": "xml",
        }
        efetch_response = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params=efetch_params
        )
        efetch_response.raise_for_status()
        
        # Parsear XML
        root = ET.fromstring(efetch_response.text)
        
        results = []
        for article in root.findall(".//PubmedArticle"):
            medline = article.find("MedlineCitation")
            if medline is None:
                continue
            
            pmid_elem = medline.find("PMID")
            pmid = pmid_elem.text if pmid_elem is not None else None
            
            article_elem = medline.find("Article")
            if article_elem is None:
                continue
            
            # Título
            title_elem = article_elem.find("ArticleTitle")
            title = title_elem.text if title_elem is not None else None
            
            # Abstract
            abstract_elem = article_elem.find("Abstract/AbstractText")
            abstract = abstract_elem.text if abstract_elem is not None else None
            
            # Autores
            authors = []
            for author in article_elem.findall("AuthorList/Author"):
                lastname = author.find("LastName")
                forename = author.find("ForeName")
                if lastname is not None:
                    name = lastname.text
                    if forename is not None:
                        name += f", {forename.text}"
                    authors.append(name)
            
            # Año
            year_elem = article_elem.find("Journal/JournalIssue/PubDate/Year")
            year = int(year_elem.text) if year_elem is not None else None
            
            if not year:
                # Intentar extraer de MedlineDate
                medline_date = article_elem.find("Journal/JournalIssue/PubDate/MedlineDate")
                if medline_date is not None and medline_date.text:
                    parts = medline_date.text.split()
                    if parts:
                        try:
                            year = int(parts[0])
                        except ValueError:
                            pass
            
            # Journal
            journal_elem = article_elem.find("Journal/Title")
            journal = journal_elem.text if journal_elem is not None else None
            
            # DOI
            doi = None
            for id_elem in article_elem.findall("ELocationID"):
                if id_elem.get("EIdType") == "doi":
                    doi = id_elem.text
                    break
            
            # URL
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
            
            doc = {
                "_source_api": "pubmed",
                "_uid": pmid,
                "title": title,
                "abstract": abstract,
                "authors": [{"name": a} for a in authors],
                "pubdate": str(year) if year else None,
                "fulljournalname": journal,
                "doi": doi,
                "url": url,
            }
            
            results.append(doc)
        
        return results


async def _search_semantic_scholar(query: str, per_page: int = 10) -> list[dict]:
    """Busca en Semantic Scholar y devuelve resultados crudos.
    
    Si SEMANTIC_SCHOLAR_API_KEY está configurada, se usa para aumentar el rate limit.
    """
    headers = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    
    async with httpx.AsyncClient(timeout=30) as client:
        params = {
            "query": query,
            "limit": per_page,
            "fields": "title,authors,year,abstract,externalIds,publicationVenue,citationCount,openAccessPdf"
        }
        response = await client.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params,
            headers=headers
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("data", [])
        
        for r in results:
            r["_source_api"] = "semantic_scholar"
        
        return results


# ═══════════════════════════════════════════════════════════
# DEDUPLICACIÓN Y MAPEO
# ═══════════════════════════════════════════════════════════

def _deduplicate_and_map(raw_papers: list[dict]) -> list[Paper]:
    """Deduplica papers por DOI y mapea al modelo Paper.
    
    Si un paper aparece en múltiples APIs, se agrupa su source_apis.
    """
    seen_dois: dict[str, Paper] = {}
    no_doi_papers: list[Paper] = []
    
    for raw in raw_papers:
        # Extraer DOI
        doi = _extract_doi(raw)
        source_api = raw.get("_source_api", "unknown")
        
        # Intentar mapear a Paper
        try:
            paper = _map_single_paper(raw)
        except Exception as e:
            logger.warning(f"Error mapeando paper de {source_api}: {e}")
            continue
        
        if doi:
            if doi in seen_dois:
                # Agregar la fuente adicional
                if source_api not in seen_dois[doi].source_apis:
                    seen_dois[doi].source_apis.append(source_api)
                # Fusionar raw_data si es necesario
                if seen_dois[doi].raw_data:
                    seen_dois[doi].raw_data[f"additional_{source_api}"] = raw
            else:
                seen_dois[doi] = paper
        else:
            # Sin DOI, no podemos deduplicar, lo agregamos directamente
            no_doi_papers.append(paper)
    
    # Combinar: papers con DOI + papers sin DOI
    final_list = list(seen_dois.values()) + no_doi_papers
    
    return final_list


def _rank_and_filter(papers: list[Paper], query_model: SearchQueryModel) -> list[Paper]:
    """Ranking preliminar algorítmico: filtra la mitad superior de papers.
    
    Calcula un score basado en cuántos términos del query aparecen en el
    título y abstract de cada paper. Ordena por score (descendente) y, en
    caso de empate, por año (descendente). Devuelve la mitad superior.
    
    Este paso reduce la carga de tokens para el LLM que hará la selección
    final de los 2 papers más relevantes.
    
    Args:
        papers: Lista de objetos Paper deduplicados.
        query_model: SearchQueryModel con los términos de búsqueda.
    
    Returns:
        Lista con la mitad superior de papers ordenados por relevancia preliminar.
    """
    if len(papers) <= 2:
        # Si hay 2 o menos, no tiene sentido filtrar
        return papers
    
    all_terms = [t.lower() for t in query_model.keywords + query_model.synonyms + query_model.concepts]
    
    if not all_terms:
        # Si no hay términos, ordenar solo por año y devolver mitad superior
        sorted_papers = sorted(papers, key=lambda p: p.anio, reverse=True)
        midpoint = (len(sorted_papers) + 1) // 2
        return sorted_papers[:midpoint]
    
    scored_papers = []
    for paper in papers:
        text = (paper.titulo + " " + (paper.abstract or "")).lower()
        score = sum(1 for term in all_terms if term in text)
        scored_papers.append((score, paper))
    
    # Ordenar: score desc, año desc (para desempatar)
    scored_papers.sort(key=lambda x: (x[0], x[1].anio), reverse=True)
    sorted_papers = [paper for score, paper in scored_papers]
    
    # Devolver mitad superior (redondeo hacia arriba)
    midpoint = (len(sorted_papers) + 1) // 2
    top_papers = sorted_papers[:midpoint]
    
    logger.info(f"Ranking preliminar: top {len(top_papers)} de {len(papers)} papers")
    return top_papers


def _extract_doi(raw: dict) -> str | None:
    """Extrae el DOI de un resultado crudo, intentando múltiples formatos."""
    # OpenAlex
    if "doi" in raw and raw["doi"]:
        return raw["doi"].replace("https://doi.org/", "").replace("http://doi.org/", "")
    
    # PubMed (nuevo formato efetch)
    if "doi" in raw and raw["doi"]:
        return raw["doi"]
    
    # PubMed (formato viejo esummary, por si acaso)
    if "articleids" in raw:
        for aid in raw["articleids"]:
            if aid.get("idtype") == "doi":
                return aid.get("value")
    
    # Semantic Scholar
    if "externalIds" in raw and raw["externalIds"]:
        doi = raw["externalIds"].get("DOI")
        if doi:
            return doi
    
    return None


def _extract_authors(raw: dict, source_api: str) -> list[str]:
    """Extrae autores del formato crudo según la API."""
    authors = []
    
    if source_api == "openalex":
        for auth in raw.get("authorships", []):
            author_name = auth.get("author", {}).get("display_name", "")
            if author_name:
                authors.append(author_name)
    
    elif source_api == "pubmed":
        for author in raw.get("authors", []):
            name = author.get("name", "")
            if name:
                authors.append(name)
    
    elif source_api == "semantic_scholar":
        for auth in raw.get("authors", []):
            name = auth.get("name", "")
            if name:
                authors.append(name)
    
    return authors


def _extract_year(raw: dict, source_api: str) -> int | None:
    """Extrae el año de publicación."""
    if source_api == "openalex":
        year = raw.get("publication_year")
        if year:
            return int(year)
    
    elif source_api == "pubmed":
        pubdate = raw.get("pubdate", "")
        if pubdate:
            # Formato típico: "2023 Jan 15" o "2023"
            parts = pubdate.split()
            if parts:
                try:
                    return int(parts[0])
                except ValueError:
                    pass
    
    elif source_api == "semantic_scholar":
        year = raw.get("year")
        if year:
            return int(year)
    
    return None


def _extract_title(raw: dict, source_api: str) -> str | None:
    """Extrae el título."""
    if source_api == "openalex":
        return raw.get("display_name")
    elif source_api == "pubmed":
        return raw.get("title")
    elif source_api == "semantic_scholar":
        return raw.get("title")
    return None


def _extract_abstract(raw: dict, source_api: str) -> str | None:
    """Extrae el abstract."""
    if source_api == "openalex":
        return raw.get("abstract")
    elif source_api == "pubmed":
        # Ahora disponible gracias a efetch (antes era None con esummary)
        return raw.get("abstract")
    elif source_api == "semantic_scholar":
        return raw.get("abstract")
    return None


def _extract_journal(raw: dict, source_api: str) -> str | None:
    """Extrae el nombre de la revista."""
    if source_api == "openalex":
        source = raw.get("primary_location", {}).get("source", {})
        return source.get("display_name")
    elif source_api == "pubmed":
        return raw.get("fulljournalname") or raw.get("source")
    elif source_api == "semantic_scholar":
        venue = raw.get("publicationVenue")
        if venue:
            return venue.get("name")
    return None


def _extract_url(raw: dict, source_api: str) -> str | None:
    """Extrae la URL del paper."""
    if source_api == "openalex":
        return raw.get("id")  # OpenAlex ID URL
    elif source_api == "pubmed":
        uid = raw.get("_uid")
        if uid:
            return f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"
    elif source_api == "semantic_scholar":
        return raw.get("url")
    return None


def _extract_citations(raw: dict, source_api: str) -> int | None:
    """Extrae el número de citas."""
    if source_api == "openalex":
        return raw.get("cited_by_count")
    elif source_api == "pubmed":
        # No disponible directamente en esummary
        return None
    elif source_api == "semantic_scholar":
        return raw.get("citationCount")
    return None


def _map_single_paper(raw: dict) -> Paper:
    """Mapea un resultado crudo de API al modelo Paper.
    
    Lanza excepción si no puede mapear los campos mínimos.
    """
    source_api = raw.get("_source_api", "unknown")
    
    # Campos obligatorios
    title = _extract_title(raw, source_api)
    authors = _extract_authors(raw, source_api)
    year = _extract_year(raw, source_api)
    
    if not title:
        raise ValueError(f"No se pudo extraer título de {source_api}")
    if not year:
        raise ValueError(f"No se pudo extraer año de {source_api}")
    
    doi = _extract_doi(raw)
    abstract = _extract_abstract(raw, source_api)
    journal = _extract_journal(raw, source_api)
    url = _extract_url(raw, source_api)
    citations = _extract_citations(raw, source_api)
    
    # Construir referencia APA 7
    referencia = _build_apa_reference(title, authors, year, journal, doi, url)
    
    paper = Paper(
        titulo=title,
        autores=authors,
        anio=year,
        doi=doi,
        abstract=abstract,
        journal=journal,
        url=url if url else None,
        citas=citations,
        referencia=referencia,
        source_apis=[source_api],
        raw_data=raw
    )
    
    return paper


def _build_apa_reference(
    title: str, 
    authors: list[str], 
    year: int, 
    journal: str | None,
    doi: str | None,
    url: str | None
) -> str:
    """Construye una referencia en formato APA 7ma edición."""
    if not authors:
        authors_str = "Autor desconocido"
    elif len(authors) == 1:
        authors_str = authors[0]
    elif len(authors) == 2:
        authors_str = f"{authors[0]} & {authors[1]}"
    else:
        authors_str = f"{authors[0]} et al."
    
    journal_str = f"{journal}" if journal else ""
    
    doi_url = f"https://doi.org/{doi}" if doi else (url if url else "")
    
    if journal_str:
        ref = f"{authors_str} ({year}). {title}. {journal_str}."
    else:
        ref = f"{authors_str} ({year}). {title}."
    
    if doi_url:
        ref += f" {doi_url}"
    
    return ref


# ═══════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════

# Exportamos el agente principal como propiedad
# Para que main.py pueda usarlo: from agentes_cote import agente_investigacion
# Y al acceder a agente_investigacion, se inicializa lazy

# Creamos una clase wrapper para poder exportar como propiedad
class _AgenteInvestigacionWrapper:
    async def run(self, prompt, **kwargs):
        """Ejecuta el agente principal de forma asíncrona."""
        agent = _get_agente_investigacion()
        return await agent.run(prompt, **kwargs)
    
    def run_sync(self, prompt, **kwargs):
        """Ejecuta el agente principal de forma síncrona."""
        agent = _get_agente_investigacion()
        return agent.run_sync(prompt, **kwargs)

agente_investigacion = _AgenteInvestigacionWrapper()
