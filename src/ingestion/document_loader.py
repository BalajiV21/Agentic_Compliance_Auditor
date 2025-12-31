"""
Document loading and preprocessing module
Supports PDF, TXT, and DOCX formats
"""
import os
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
import fitz  # PyMuPDF
from docx import Document


class DocumentLoader:
    """Load and preprocess documents from various formats"""

    def __init__(self):
        self.supported_formats = ['.pdf', '.txt', '.docx']

    def load_document(self, file_path: str) -> Dict[str, any]:
        """
        Load a document and extract text content

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary containing document text and metadata
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        logger.info(f"Loading document: {file_path.name}")

        if file_path.suffix.lower() == '.pdf':
            return self._load_pdf(file_path)
        elif file_path.suffix.lower() == '.txt':
            return self._load_txt(file_path)
        elif file_path.suffix.lower() == '.docx':
            return self._load_docx(file_path)

    def _load_pdf(self, file_path: Path) -> Dict[str, any]:
        """Load PDF file using PyMuPDF"""
        doc = fitz.open(file_path)
        text_content = []

        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            text_content.append({
                'page': page_num,
                'content': text
            })

        full_text = "\n\n".join([page['content'] for page in text_content])

        metadata = {
            'filename': file_path.name,
            'source': str(file_path),
            'document_type': self._infer_document_type(file_path.name),
            'num_pages': len(doc),
            'format': 'pdf'
        }

        doc.close()

        return {
            'text': full_text,
            'pages': text_content,
            'metadata': metadata
        }

    def _load_txt(self, file_path: Path) -> Dict[str, any]:
        """Load plain text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        metadata = {
            'filename': file_path.name,
            'source': str(file_path),
            'document_type': self._infer_document_type(file_path.name),
            'format': 'txt'
        }

        return {
            'text': text,
            'metadata': metadata
        }

    def _load_docx(self, file_path: Path) -> Dict[str, any]:
        """Load DOCX file"""
        doc = Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        text = "\n\n".join(paragraphs)

        metadata = {
            'filename': file_path.name,
            'source': str(file_path),
            'document_type': self._infer_document_type(file_path.name),
            'num_paragraphs': len(paragraphs),
            'format': 'docx'
        }

        return {
            'text': text,
            'metadata': metadata
        }

    def _infer_document_type(self, filename: str) -> str:
        """Infer regulation type from filename"""
        filename_lower = filename.lower()

        if 'gdpr' in filename_lower:
            return 'GDPR'
        elif 'hipaa' in filename_lower:
            return 'HIPAA'
        elif 'soc2' in filename_lower or 'soc 2' in filename_lower:
            return 'SOC2'
        elif 'pci' in filename_lower:
            return 'PCI-DSS'
        elif 'iso' in filename_lower:
            return 'ISO-27001'
        else:
            return 'General'

    def load_directory(self, directory_path: str) -> List[Dict[str, any]]:
        """
        Load all documents from a directory

        Args:
            directory_path: Path to directory containing documents

        Returns:
            List of document dictionaries
        """
        directory = Path(directory_path)

        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Invalid directory: {directory_path}")

        documents = []

        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    doc = self.load_document(str(file_path))
                    documents.append(doc)
                    logger.info(f"Successfully loaded: {file_path.name}")
                except Exception as e:
                    logger.error(f"Error loading {file_path.name}: {str(e)}")

        logger.info(f"Loaded {len(documents)} documents from {directory_path}")
        return documents


if __name__ == "__main__":
    # Test the document loader
    loader = DocumentLoader()

    # Load sample documents
    sample_dir = Path(__file__).parent.parent.parent / "data" / "sample_docs"

    if sample_dir.exists():
        docs = loader.load_directory(str(sample_dir))

        for doc in docs:
            print(f"\nDocument: {doc['metadata']['filename']}")
            print(f"Type: {doc['metadata']['document_type']}")
            print(f"Text length: {len(doc['text'])} characters")
            print(f"Preview: {doc['text'][:200]}...")
