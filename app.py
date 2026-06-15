"""Interfaz web para el Agente de Búsqueda Bibliográfica en Psicología Clínica.

Esta aplicación usa NiceGUI para ofrecer una interfaz web moderna y reactiva
que permite a psicólogos pegar una bitácora clínica y obtener los 3 papers
más relevantes de múltiples fuentes (PubMed, OpenAlex, Semantic Scholar, CrossRef).

Ejecución:
    python app.py

La aplicación se abrirá automáticamente en el navegador (por defecto http://localhost:8080).
"""

import csv
import io
import os
from pathlib import Path

from fpdf import FPDF
from nicegui import ui

from agentes import agente_investigacion

# ═══════════════════════════════════════════════════════════
# CONFIGURACIÓN VISUAL
# ═══════════════════════════════════════════════════════════

SOURCE_COLORS = {
    "pubmed": "bg-blue-100 text-blue-800 border-blue-200",
    "openalex_en": "bg-orange-100 text-orange-800 border-orange-200",
    "openalex_es": "bg-yellow-100 text-yellow-800 border-yellow-200",
    "semantic_scholar": "bg-purple-100 text-purple-800 border-purple-200",
    "crossref": "bg-red-100 text-red-800 border-red-200",
}

SOURCE_DISPLAY_NAMES = {
    "pubmed": "PubMed",
    "openalex_en": "OpenAlex EN",
    "openalex_es": "OpenAlex ES",
    "semantic_scholar": "Semantic Scholar",
    "crossref": "CrossRef",
}

# Ruta a una fuente Unicode del sistema para generar PDFs con acentos y eñes
UNICODE_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
HAS_UNICODE_FONT = os.path.exists(UNICODE_FONT_PATH)

# ═══════════════════════════════════════════════════════════
# GENERACIÓN DE ARCHIVOS DE DESCARGA
# ═══════════════════════════════════════════════════════════


def _safe_text(text: str | None) -> str:
    """Devuelve texto seguro o un string vacío si es None."""
    return text if text is not None else ""


def _format_authors(authors: list[str]) -> str:
    """Formatea la lista de autores para mostrar."""
    if not authors:
        return "Autor desconocido"
    if len(authors) <= 3:
        return ", ".join(authors)
    return f"{', '.join(authors[:3])} et al."


def generate_pdf(papers: list) -> bytes:
    """Genera un PDF con los papers seleccionados.

    Usa DejaVuSans si está disponible para soportar caracteres Unicode
    (acentos, eñes, etc.). De lo contrario, usa latin-1 con reemplazo.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    if HAS_UNICODE_FONT:
        pdf.add_font("DejaVu", "", UNICODE_FONT_PATH, uni=True)
        pdf.add_font("DejaVu", "B", UNICODE_FONT_PATH.replace("Sans.ttf", "Sans-Bold.ttf"), uni=True)
        font_name = "DejaVu"
    else:
        font_name = "Arial"

    # Título
    pdf.set_font(font_name, "B", 16)
    pdf.cell(0, 10, "Papers relevantes para el caso clínico", ln=True, align="C")
    pdf.set_font(font_name, "", 10)
    pdf.cell(0, 6, f"Total de papers: {len(papers)}", ln=True, align="C")
    pdf.ln(8)

    for i, paper in enumerate(papers, 1):
        # Título
        pdf.set_font(font_name, "B", 12)
        pdf.multi_cell(0, 6, f"{i}. {_safe_text(paper.titulo)}")

        # Metadatos
        pdf.set_font(font_name, "", 10)
        pdf.cell(0, 5, f"Autores: {_format_authors(paper.autores)}", ln=True)
        pdf.cell(0, 5, f"Año: {paper.anio}  |  Revista: {_safe_text(paper.journal)}", ln=True)
        pdf.cell(0, 5, f"Citas: {paper.citas if paper.citas is not None else 'No disponible'}", ln=True)
        pdf.cell(0, 5, f"Fuentes: {', '.join(SOURCE_DISPLAY_NAMES.get(s, s) for s in paper.source_apis)}", ln=True)

        if paper.doi:
            pdf.set_text_color(0, 0, 255)
            pdf.cell(0, 5, f"DOI: https://doi.org/{paper.doi}", ln=True, link=f"https://doi.org/{paper.doi}")
            pdf.set_text_color(0, 0, 0)

        # Justificación
        if paper.justificacion:
            pdf.set_font(font_name, "B", 10)
            pdf.cell(0, 5, "Justificación de relevancia:", ln=True)
            pdf.set_font(font_name, "", 10)
            pdf.multi_cell(0, 5, _safe_text(paper.justificacion))

        # Abstract
        if paper.abstract:
            pdf.set_font(font_name, "B", 10)
            pdf.cell(0, 5, "Abstract original:", ln=True)
            pdf.set_font(font_name, "", 10)
            pdf.multi_cell(0, 5, _safe_text(paper.abstract))

        pdf.ln(6)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

    output = pdf.output(dest="S")
    return bytes(output) if isinstance(output, bytearray) else output.encode("latin-1")


def generate_csv(papers: list) -> bytes:
    """Genera un CSV con los papers seleccionados."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Paper #",
        "Título",
        "Autores",
        "Año",
        "DOI",
        "Revista",
        "Citas",
        "Fuentes",
        "Justificación",
        "Abstract",
        "URL",
    ])

    for i, paper in enumerate(papers, 1):
        writer.writerow([
            i,
            paper.titulo,
            "; ".join(paper.autores),
            paper.anio,
            paper.doi or "",
            paper.journal or "",
            paper.citas if paper.citas is not None else "",
            ", ".join(paper.source_apis),
            paper.justificacion or "",
            paper.abstract or "",
            str(paper.url) if paper.url else "",
        ])

    return output.getvalue().encode("utf-8")


# ═══════════════════════════════════════════════════════════
# COMPONENTES DE LA INTERFAZ
# ═══════════════════════════════════════════════════════════


def _build_source_badges(source_apis: list[str]) -> None:
    """Crea badges de colores para cada fuente de búsqueda."""
    for source in source_apis:
        color_class = SOURCE_COLORS.get(source, "bg-gray-100 text-gray-800 border-gray-200")
        display_name = SOURCE_DISPLAY_NAMES.get(source, source)
        ui.badge(display_name).classes(f"{color_class} border px-2 py-1 rounded-full text-xs font-medium")


def _render_paper_card(index: int, paper) -> None:
    """Renderiza una tarjeta con toda la información de un paper."""
    with ui.card().classes("w-full mb-5 shadow-md rounded-xl border border-gray-100 bg-white"):
        # Header: número + fuentes
        with ui.row().classes("w-full items-center justify-between mb-2"):
            ui.label(f"📄 Paper {index}").classes("text-xl font-bold text-gray-800")
            with ui.row().classes("gap-1"):
                _build_source_badges(paper.source_apis)

        # Título con link
        title_classes = "text-lg font-semibold text-primary hover:underline cursor-pointer"
        if paper.url:
            ui.link(paper.titulo, str(paper.url), new_tab=True).classes(title_classes)
        else:
            ui.label(paper.titulo).classes("text-lg font-semibold text-gray-900")

        # Metadatos
        with ui.row().classes("w-full flex-wrap gap-x-6 gap-y-1 text-sm text-gray-600 mt-2"):
            ui.label(f"👤 {_format_authors(paper.autores)}")
            ui.label(f"📅 {paper.anio}")
            if paper.citas is not None:
                ui.label(f"📊 {paper.citas} citas")
            if paper.journal:
                ui.label(f"📰 {paper.journal}")

        # DOI
        if paper.doi:
            with ui.row().classes("mt-2"):
                ui.link(
                    f"🔗 DOI: {paper.doi}",
                    f"https://doi.org/{paper.doi}",
                    new_tab=True,
                ).classes("text-sm text-blue-600 hover:underline")

        # Justificación
        if paper.justificacion:
            with ui.card().classes("w-full mt-4 bg-blue-50 border-l-4 border-blue-500 rounded-r-lg"):
                ui.label("📝 Justificación de relevancia").classes("font-semibold text-blue-900")
                ui.label(paper.justificacion).classes("text-blue-900 mt-1 leading-relaxed")

        # Abstract
        if paper.abstract:
            with ui.expansion("📋 Abstract original", value=False).classes("w-full mt-4"):
                ui.label(paper.abstract).classes("text-gray-700 whitespace-pre-wrap leading-relaxed")


def _render_download_buttons(papers: list) -> None:
    """Renderiza botones para descargar PDF y CSV."""
    with ui.row().classes("w-full justify-center gap-4 mt-6 mb-8"):
        ui.button(
            "📥 Descargar PDF",
            on_click=lambda: ui.download(generate_pdf(papers), "papers_relevantes.pdf"),
        ).classes("bg-red-600 hover:bg-red-700 text-white font-semibold px-6 py-2 rounded-lg shadow")

        ui.button(
            "📥 Descargar CSV",
            on_click=lambda: ui.download(generate_csv(papers), "papers_relevantes.csv"),
        ).classes("bg-green-600 hover:bg-green-700 text-white font-semibold px-6 py-2 rounded-lg shadow")


# ═══════════════════════════════════════════════════════════
# PÁGINA PRINCIPAL
# ═══════════════════════════════════════════════════════════


@ui.page("/")
def main_page():
    """Define la página principal de la aplicación NiceGUI."""
    ui.colors(primary="#2563eb", secondary="#475569")
    ui.query("body").classes("bg-slate-50")

    # Contenedor principal centrado
    with ui.column().classes("w-full max-w-4xl mx-auto p-4 md:p-6"):
        # Header
        with ui.column().classes("w-full items-center mb-2"):
            ui.label("🔬 Agente de Búsqueda Bibliográfica").classes(
                "text-3xl md:text-4xl font-bold text-center text-primary"
            )
            ui.label("para Psicología Clínica").classes(
                "text-xl md:text-2xl text-center text-gray-600 mt-1"
            )
            ui.label(
                "Ingresa la bitácora clínica, resumen de sesión o apuntes del paciente. "
                "El agente buscará en PubMed, Semantic Scholar, OpenAlex (inglés/español) y CrossRef."
            ).classes("text-center text-gray-500 mt-3 max-w-2xl")

        # Input section
        with ui.card().classes("w-full mt-6 shadow-sm rounded-xl"):
            ui.label("📝 Caso clínico").classes("text-lg font-semibold text-gray-800")
            bitacora_input = ui.textarea(
                placeholder=(
                    "Ejemplo: Paciente de 28 años con diagnóstico de trastorno de ansiedad generalizada. "
                    "Refiere preocupación excesiva, tensión muscular y dificultades para conciliar el sueño. "
                    "Se inició tratamiento con técnicas cognitivo-conductuales..."
                )
            ).props("rows=8 outlined").classes("w-full mt-3")

            with ui.row().classes("w-full justify-end mt-4"):
                ui.button(
                    "🔍 Buscar papers relevantes",
                    on_click=lambda: perform_search(bitacora_input.value),
                ).classes(
                    "bg-primary hover:bg-blue-700 text-white font-semibold px-8 py-2 rounded-lg shadow"
                )

        # Results section
        results_container = ui.column().classes("w-full mt-6")

    async def perform_search(bitacora: str) -> None:
        """Ejecuta la búsqueda y renderiza los resultados.

        Esta función es async para poder usar `await` directamente con
        el agente de pydantic-ai, aprovechando el soporte nativo de NiceGUI.
        """
        if not bitacora or not bitacora.strip():
            ui.notify("Por favor ingresa un caso clínico antes de buscar.", type="warning")
            return

        # Limpiar resultados anteriores
        results_container.clear()

        # Mostrar spinner de carga
        with results_container:
            with ui.row().classes("w-full justify-center items-center gap-4 py-12"):
                ui.spinner("dots", size="3em", color="primary")
                ui.label("Buscando papers relevantes en múltiples fuentes...").classes(
                    "text-gray-600 text-lg"
                )

        try:
            prompt = f"Busca papers relevantes para este caso clínico:\n\n{bitacora.strip()}"
            result = await agente_investigacion.run(prompt)
            papers = result.output

            # Limpiar spinner
            results_container.clear()

            with results_container:
                if not papers:
                    ui.icon("sentiment_dissatisfied", size="3em").classes("text-gray-400 mx-auto")
                    ui.label("No se encontraron papers relevantes para este caso.").classes(
                        "text-center text-gray-500 text-lg mt-2"
                    )
                    return

                ui.label(f"📊 {len(papers)} papers encontrados").classes(
                    "text-2xl font-bold text-gray-800 mb-4"
                )

                for i, paper in enumerate(papers, 1):
                    _render_paper_card(i, paper)

                _render_download_buttons(papers)

        except Exception as e:
            results_container.clear()
            with results_container:
                with ui.card().classes("w-full bg-red-50 border-l-4 border-red-500 rounded-r-lg"):
                    ui.label("❌ Error al procesar la búsqueda").classes(
                        "text-lg font-semibold text-red-800"
                    )
                    ui.label(str(e)).classes("text-red-700 mt-2")
                ui.label("Por favor revisa que GOOGLE_API_KEY esté configurada correctamente.").classes(
                    "text-center text-gray-500 mt-4"
                )


# ═══════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="Agente de Búsqueda Bibliográfica",
        favicon="🔬",
        port=8080,
        reload=False,
        show=True,
    )
