"""
Setup script for Agentic Compliance Auditor
Initializes the vector database with sample documents
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from ingestion import DocumentLoader, RegulationChunker
from retrieval import VectorStore
from config import settings
from loguru import logger


def main():
    """Setup the compliance auditor system"""
    print("="*60)
    print("🚀 Agentic Compliance Auditor - Setup")
    print("="*60)

    # Step 1: Initialize components
    print("\n📦 Initializing components...")
    loader = DocumentLoader()
    chunker = RegulationChunker(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    vector_store = VectorStore(
        persist_directory=settings.CHROMA_PERSIST_DIR,
        collection_name='compliance_documents',
        embedding_model=settings.EMBEDDING_MODEL
    )

    # Step 2: Load sample documents
    print("\n📄 Loading sample documents...")
    sample_dir = Path(__file__).parent / "data" / "sample_docs"

    if not sample_dir.exists():
        print(f"❌ Sample documents directory not found: {sample_dir}")
        return False

    docs = loader.load_directory(str(sample_dir))

    if not docs:
        print("❌ No documents found in sample_docs directory")
        return False

    print(f"✓ Loaded {len(docs)} documents:")
    for doc in docs:
        print(f"  - {doc['metadata']['filename']} ({doc['metadata']['document_type']})")

    # Step 3: Chunk documents
    print("\n✂️ Chunking documents...")
    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  - {doc['metadata']['filename']}: {len(chunks)} chunks")

    print(f"✓ Created {len(all_chunks)} chunks total")

    # Step 4: Store in vector database
    print("\n💾 Storing in vector database...")
    print("  (This may take a few minutes...)")

    vector_store.add_documents(all_chunks)

    # Step 5: Verify
    print("\n✅ Verifying setup...")
    stats = vector_store.get_collection_stats()
    print(f"  - Collection: {stats['collection_name']}")
    print(f"  - Total chunks: {stats['total_chunks']}")
    print(f"  - Embedding dimension: {stats['embedding_model']}")

    # Step 6: Test search
    print("\n🔍 Testing search...")
    results = vector_store.collection.query(
        query_texts=["data retention"],
        n_results=1
    )

    if results and results['ids'] and results['ids'][0]:
        print("✓ Search is working correctly")
    else:
        print("⚠️ Search test did not return results")

    print("\n"+"="*60)
    print("✅ Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Start Ollama: ollama serve")
    print("2. Start API: cd src/api && python main.py")
    print("3. Start UI: cd ui && streamlit run streamlit_app.py")
    print("\nFor more information, see README.md")

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        logger.exception("Setup error")
        sys.exit(1)
