from pathlib import Path
from pypdf import PdfReader


def load_document(file_path: str):

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    # Extract text from PDF
    if extension == ".pdf":

        reader = PdfReader(str(path))

        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    # Read TXT files
    elif extension == ".txt":
        text = path.read_text(encoding="utf-8")

    # Read Markdown files
    elif extension in [".md", ".markdown"]:
        text = path.read_text(encoding="utf-8")

    else:
        raise ValueError("Unsupported file format")

    if not text.strip():
        raise ValueError("No readable text found in document")

    return text
