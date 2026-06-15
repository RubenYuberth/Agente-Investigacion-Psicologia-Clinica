import asyncio
from agentes_cote import agente_investigacion


async def main():
    print("=== Agente de Búsqueda Bibliográfica para Psicología Clínica ===")
    print("Ingrese la bitácora clínica, resumen de sesión o apuntes del paciente:")
    print("(Presione Ctrl+D o Ctrl+Z para terminar la entrada)\n")
    
    # Leer entrada multilínea
    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass
    
    bitacora = "\n".join(lines)
    
    if not bitacora.strip():
        print("No se recibió entrada. Saliendo.")
        return
    
    print("\n--- Procesando caso clínico... ---\n")
    
    # Ejecutar el agente
    result = await agente_investigacion.run(
        f"Busca papers relevantes para este caso clínico:\n\n{bitacora}"
    )
    
    papers = result.output
    
    print(f"\n=== RESULTADOS: {len(papers)} papers encontrados ===\n")
    
    for i, paper in enumerate(papers, 1):
        print(f"--- Paper {i} ---")
        print(f"Título: {paper.titulo}")
        print(f"Autores: {', '.join(paper.autores) if paper.autores else 'No disponible'}")
        print(f"Año: {paper.anio}")
        print(f"DOI: {paper.doi or 'No disponible'}")
        print(f"Revista: {paper.journal or 'No disponible'}")
        print(f"Citas: {paper.citas or 'No disponible'}")
        print(f"Fuentes: {', '.join(paper.source_apis)}")
        print(f"URL: {paper.url or 'No disponible'}")
        if paper.abstract:
            print(f"Abstract: {paper.abstract[:300]}...")
        print(f"Referencia APA: {paper.referencia}")
        print()
    
    print("=== Fin de resultados ===")


if __name__ == "__main__":
    asyncio.run(main())
