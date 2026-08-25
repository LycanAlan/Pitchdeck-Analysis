from rag_pipeline.embedder import get_embedder

def test_embedder():
    embedder = get_embedder()
    print("✅ Embedder object created successfully.")

    sample_text = "SensoVision automates visual inspection using AI-powered vision systems."
    embedding = embedder.embed_query(sample_text)

    print(f"✅ Got embedding of length: {len(embedding)}")
    print("🔢 First 10 embedding values:", embedding[:10])

if __name__ == "__main__":
    test_embedder()
