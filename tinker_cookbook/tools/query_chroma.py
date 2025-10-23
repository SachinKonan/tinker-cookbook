#!/usr/bin/env python3
"""
Interactive Chroma DB Query Tool

Allows you to interactively query a Chroma vector database using Gemini embeddings,
exactly as an AI agent would during training.

Usage:
    uv run python -m tinker_cookbook.tools.query_chroma --host localhost --port 8000
    uv run python -m tinker_cookbook.tools.query_chroma --host localhost --port 8000 --collection-name my_collection
"""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from tinker_cookbook.recipes.tool_use.search.embedding import (
    get_gemini_client,
    get_gemini_embedding,
)
from tinker_cookbook.recipes.tool_use.search.tools import (
    ChromaToolClient,
    ChromaToolClientConfig,
    EmbeddingConfig,
    RetrievalConfig,
)

# Load environment variables from repo root
repo_root = Path(__file__).parent.parent.parent
env_path = repo_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


async def query_chroma_interactive(
    host: str,
    port: int,
    collection_name: str,
    n_results: int,
):
    """
    Interactive Chroma DB query loop
    """
    print("=" * 80)
    print("Interactive Chroma DB Query Tool")
    print("=" * 80)
    print(f"Connecting to Chroma DB at {host}:{port}")
    print(f"Collection: {collection_name}")
    print(f"Results per query: {n_results}")
    print("-" * 80)

    # Create ChromaToolClient configuration
    config = ChromaToolClientConfig(
        chroma_host=host,
        chroma_port=port,
        chroma_collection_name=collection_name,
        retrieval_config=RetrievalConfig(
            n_results=n_results,
            embedding_config=EmbeddingConfig(
                model_name="gemini-embedding-001",
                embedding_dim=768,
                task_type="RETRIEVAL_QUERY",
            ),
        ),
        max_retries=10,
        initial_retry_delay=1,
    )

    try:
        # Initialize ChromaToolClient
        print("Initializing Chroma client and Gemini embeddings...")
        chroma_client = await ChromaToolClient.create(config)
        print("✓ Connected successfully!")
        print()

        # List available collections
        print("Checking available collections...")
        collections = await chroma_client.chroma_client.list_collections()
        print(f"Available collections ({len(collections)}):")
        for coll in collections:
            print(f"  - {coll.name}")
        print()

        # Verify collection exists
        try:
            collection = await chroma_client.chroma_client.get_collection(collection_name)
            print(f"✓ Collection '{collection_name}' found (skipping document count for speed)")
        except Exception as e:
            print(f"✗ Error accessing collection '{collection_name}': {e}")
            print(f"\nPlease use one of the available collections listed above.")
            print(f"Use --collection-name <name> to specify a different collection.")
            return

        print()
        print("=" * 80)
        print("Ready to query! Type your search query and press Enter.")
        print("Type 'exit' or press Ctrl+C to quit.")
        print("=" * 80)
        print()

        # Interactive query loop
        while True:
            try:
                # Get user query
                query = input("Query: ").strip()

                if not query:
                    print("(empty query, skipping)")
                    continue

                if query.lower() in ["exit", "quit", "q"]:
                    print("\nGoodbye!")
                    break

                print()
                print("-" * 80)

                # Get embeddings for the query
                print(f"Generating Gemini embedding for: '{query}'")
                query_embeddings = await get_gemini_embedding(
                    chroma_client.gemini_client,
                    [query],
                    model=config.retrieval_config.embedding_config.model_name,
                    embedding_dim=config.retrieval_config.embedding_config.embedding_dim,
                    task_type=config.retrieval_config.embedding_config.task_type,
                )

                # Query Chroma DB
                print(f"Searching Chroma collection '{collection_name}'...")
                collection = await chroma_client.chroma_client.get_collection(collection_name)
                results = await collection.query(
                    query_embeddings=query_embeddings,
                    n_results=n_results,
                )

                # Display results
                print()
                print(f"Search Results for: '{query}'")
                print("=" * 80)

                if results["documents"] and results["documents"][0]:
                    documents = results["documents"][0]
                    distances = results.get("distances", [[]] * len(documents))[0]

                    for i, (doc, dist) in enumerate(zip(documents, distances), 1):
                        print(f"\n[Document {i}]")
                        if dist:
                            print(f"Distance: {dist:.4f}")
                        print("-" * 40)
                        print(doc)
                else:
                    print("No results found.")

                print()
                print("=" * 80)
                print()

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError processing query: {e}")
                print("Please try again.")
                print()

    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Chroma DB query tool using Gemini embeddings"
    )
    parser.add_argument(
        "--host",
        type=str,
        required=True,
        help="Chroma DB host (e.g., localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Chroma DB port (e.g., 8000)",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="wiki_embeddings",
        help="Chroma collection name (default: wiki_embeddings)",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=3,
        help="Number of results to return per query (default: 3)",
    )

    args = parser.parse_args()

    # Run the interactive query loop
    asyncio.run(
        query_chroma_interactive(
            host=args.host,
            port=args.port,
            collection_name=args.collection_name,
            n_results=args.n_results,
        )
    )


if __name__ == "__main__":
    main()
