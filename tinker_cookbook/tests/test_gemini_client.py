"""
Test for Gemini client and embedding generation
Tests get_gemini_client() and get_gemini_embedding() from embedding.py
"""
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables from repo root
repo_root = os.path.join(os.path.dirname(__file__), '../..')
env_path = os.path.join(repo_root, '.env')
load_dotenv(env_path)

from tinker_cookbook.recipes.tool_use.search.embedding import (
    get_gemini_client,
    get_gemini_embedding,
)


async def test_client_creation():
    """Test get_gemini_client() with environment variables"""
    print("=" * 80)
    print("TEST 1: Client Creation")
    print("=" * 80)

    # Check environment variables
    print("\nEnvironment Variables:")
    print(f"  GCP_VERTEXAI_PROJECT_NUMBER: {os.getenv('GCP_VERTEXAI_PROJECT_NUMBER')}")
    print(f"  GCP_VERTEXAI_REGION: {os.getenv('GCP_VERTEXAI_REGION')}")
    print(f"  GOOGLE_GENAI_USE_VERTEXAI: {os.getenv('GOOGLE_GENAI_USE_VERTEXAI', 'True')}")
    print()

    try:
        client = get_gemini_client()
        print("✓ Client created successfully")
        print(f"  Client type: {type(client)}")
        return client
    except Exception as e:
        print(f"✗ Client creation failed: {e}")
        raise


async def test_single_text_embedding(client):
    """Test embedding generation with a single text"""
    print("\n" + "=" * 80)
    print("TEST 2: Single Text Embedding")
    print("=" * 80)

    texts = ["What is the capital of France?"]
    print(f"\nInput texts: {texts}")

    try:
        embeddings = await get_gemini_embedding(
            client=client,
            texts=texts,
            model="gemini-embedding-001",
            embedding_dim=768,
            task_type="RETRIEVAL_QUERY",
        )

        print(f"✓ Embeddings generated successfully")
        print(f"  Number of embeddings: {len(embeddings)}")
        print(f"  Embedding dimension: {len(embeddings[0])}")
        print(f"  First 5 values: {embeddings[0][:5]}")

        # Validate
        assert len(embeddings) == len(texts), f"Expected {len(texts)} embeddings, got {len(embeddings)}"
        assert len(embeddings[0]) == 768, f"Expected dimension 768, got {len(embeddings[0])}"
        print("✓ Validation passed")

    except Exception as e:
        print(f"✗ Embedding generation failed: {e}")
        raise


async def test_multiple_texts_embedding(client):
    """Test embedding generation with multiple texts"""
    print("\n" + "=" * 80)
    print("TEST 3: Multiple Texts Embedding")
    print("=" * 80)

    texts = [
        "What is machine learning?",
        "How does reinforcement learning work?",
        "Explain neural networks",
    ]
    print(f"\nInput texts ({len(texts)} texts):")
    for i, text in enumerate(texts):
        print(f"  {i+1}. {text}")

    try:
        embeddings = await get_gemini_embedding(
            client=client,
            texts=texts,
            model="gemini-embedding-001",
            embedding_dim=768,
            task_type="RETRIEVAL_QUERY",
        )

        print(f"\n✓ Embeddings generated successfully")
        print(f"  Number of embeddings: {len(embeddings)}")
        for i, emb in enumerate(embeddings):
            print(f"  Embedding {i+1} dimension: {len(emb)}, first 3 values: {emb[:3]}")

        # Validate
        assert len(embeddings) == len(texts), f"Expected {len(texts)} embeddings, got {len(embeddings)}"
        for i, emb in enumerate(embeddings):
            assert len(emb) == 768, f"Embedding {i} has dimension {len(emb)}, expected 768"
        print("✓ Validation passed")

    except Exception as e:
        print(f"✗ Embedding generation failed: {e}")
        raise


async def test_different_embedding_dim(client):
    """Test with different embedding dimensions"""
    print("\n" + "=" * 80)
    print("TEST 4: Different Embedding Dimensions")
    print("=" * 80)

    texts = ["Test embedding with different dimensions"]
    dimensions = [256, 512, 768]

    for dim in dimensions:
        print(f"\nTesting dimension: {dim}")
        try:
            embeddings = await get_gemini_embedding(
                client=client,
                texts=texts,
                embedding_dim=dim,
            )

            actual_dim = len(embeddings[0])
            if actual_dim == dim:
                print(f"  ✓ Dimension {dim}: {actual_dim} values (correct)")
            else:
                print(f"  ⚠ Dimension {dim}: {actual_dim} values (expected {dim})")

        except Exception as e:
            print(f"  ✗ Failed with dimension {dim}: {e}")


async def test_different_task_types(client):
    """Test with different task types"""
    print("\n" + "=" * 80)
    print("TEST 5: Different Task Types")
    print("=" * 80)

    texts = ["Document to embed"]
    task_types = ["RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"]

    for task_type in task_types:
        print(f"\nTesting task type: {task_type}")
        try:
            embeddings = await get_gemini_embedding(
                client=client,
                texts=texts,
                task_type=task_type,
            )
            print(f"  ✓ Task type {task_type}: Generated {len(embeddings[0])} dimensional embedding")

        except Exception as e:
            print(f"  ✗ Failed with task type {task_type}: {e}")


async def test_error_handling(client):
    """Test error handling with invalid inputs"""
    print("\n" + "=" * 80)
    print("TEST 6: Error Handling")
    print("=" * 80)

    # Test 1: Empty list
    print("\nTest 6a: Empty text list")
    try:
        await get_gemini_embedding(client=client, texts=[])
        print("  ✗ Should have raised ValueError for empty list")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")

    # Test 2: Whitespace-only text
    print("\nTest 6b: Whitespace-only text")
    try:
        await get_gemini_embedding(client=client, texts=["   "])
        print("  ✗ Should have raised ValueError for whitespace text")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")

    # Test 3: Non-string input
    print("\nTest 6c: Non-string input")
    try:
        await get_gemini_embedding(client=client, texts=["valid text", 123])  # type: ignore
        print("  ✗ Should have raised ValueError for non-string")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")


async def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("GEMINI CLIENT AND EMBEDDING TESTS")
    print("=" * 80)
    print()

    try:
        # Test 1: Create client
        client = await test_client_creation()

        # Test 2: Single text embedding
        await test_single_text_embedding(client)

        # Test 3: Multiple texts embedding
        await test_multiple_texts_embedding(client)

        # Test 4: Different embedding dimensions
        await test_different_embedding_dim(client)

        # Test 5: Different task types
        await test_different_task_types(client)

        # Test 6: Error handling
        await test_error_handling(client)

        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"TESTS FAILED: {e}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    asyncio.run(main())
