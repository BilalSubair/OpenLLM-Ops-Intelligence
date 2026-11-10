from pathlib import Path

from rag.document_loader import load_document
from rag.ingestion import ingest_document


def process_document(file_path: str):

    path = Path(file_path)

    # Step 1: Extract document text
    text = load_document(file_path)

    # Step 2: Generate document ID
    document_id = path.stem

    # Step 3: Ingest into ChromaDB
    result = ingest_document(
        text=text,
        document_id=document_id,
        source=path.name,
        category="infrastructure"
    )

    return result
