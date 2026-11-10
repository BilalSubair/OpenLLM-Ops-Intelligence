from pathlib import Path
from rag.pipeline import process_document


ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


def process_uploaded_file(file_path: str):

    path = Path(file_path)

    # Validate file format
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file format")

    # Process and store document
    result = process_document(str(path))

    return {
        "filename": path.name,
        "document_id": result["document_id"],
        "total_chunks": result["total_chunks"],
        "message": "File uploaded and processed successfully"
    }
