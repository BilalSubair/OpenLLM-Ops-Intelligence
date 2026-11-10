from rag.document_loader import load_document

file_path = "data/documents/sample.txt"

try:
    text = load_document(file_path)

    print("\nDocument loaded successfully!")
    print("Total characters:", len(text))

    print("\nExtracted content:")
    print(text[:1000])

except Exception as error:
    print("Error:", error)	
