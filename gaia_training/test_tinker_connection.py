"""
Test Tinker API connection in gaia_training
"""
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables from .env
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
print(f"Loading .env from: {env_path}")
load_dotenv(env_path)


async def test_tinker_connection():
    """Test that we can connect to Tinker service"""
    import tinker

    print("\n" + "=" * 70)
    print("Testing Tinker API Connection (GAIA Training)")
    print("=" * 70)

    # Check API key
    api_key = os.getenv("TINKER_API_KEY")
    if api_key:
        print(f"✓ TINKER_API_KEY found: {api_key[:20]}...")
    else:
        print("✗ TINKER_API_KEY not found!")
        return False

    try:
        # Create service client
        print("\n[1/2] Creating Tinker ServiceClient...")
        service_client = tinker.ServiceClient()
        print("✓ ServiceClient created successfully")

        print(
            await service_client.get_server_capabilities_async()
        )

        # Try to create a training client
        print("\n[2/2] Testing training client creation...")
        test_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        print(f"  Model: {test_model}")
        print(f"  LoRA rank: 32")

        training_client = await service_client.create_lora_training_client_async(
            test_model,
            rank=32
        )
        print("✓ Training client created successfully")

        print("\n" + "=" * 70)
        print("✅ All Tinker API tests passed!")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n✗ Error connecting to Tinker API:")
        print(f"  {type(e).__name__}: {e}")

        # Check if it's the known SSL error
        if "ssl.SSLError" in str(type(e)) or "SSLError" in str(e):
            print("\n⚠️  SSL/HTTPX configuration issue detected")

        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 70)
        print("❌ Tinker API connection failed")
        print("=" * 70)
        return False


if __name__ == "__main__":
    result = asyncio.run(test_tinker_connection())
    exit(0 if result else 1)
