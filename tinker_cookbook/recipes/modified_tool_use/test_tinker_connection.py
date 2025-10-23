"""
Test Tinker API connection
Quick test to verify Tinker service is accessible before running training
"""
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from repo root
repo_root = Path(__file__).parent.parent.parent.parent
env_path = repo_root / ".env"
print(f"Loading .env from: {env_path}")
if env_path.exists():
    load_dotenv(env_path)
else:
    print("Warning: .env not found, using system environment")
    load_dotenv()


async def test_tinker_connection():
    """Test that we can connect to Tinker service"""
    import tinker

    print("\n" + "=" * 70)
    print("Testing Tinker API Connection")
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

        # Try to create a training client (without actually starting training)
        print("\n[2/2] Testing training client creation...")
        test_model = "Qwen/Qwen3-4B-Instruct-2507"
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
            print("\n⚠️  This appears to be an SSL/HTTPX configuration issue.")
            print("   This is NOT a problem with the modified_tool_use code.")
            print("   Possible causes:")
            print("   - Python 3.13 SSL compatibility issue")
            print("   - Missing SSL certificates")
            print("   - Network/proxy configuration")
            print("   - Try with a different Python version (3.11 or 3.12)")

        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 70)
        print("❌ Tinker API connection failed")
        print("=" * 70)
        return False



async def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("MODIFIED_TOOL_USE CONNECTION TESTS")
    print("=" * 70)

    # Test Tinker connection
    tinker_ok = await test_tinker_connection()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Tinker API: {'✅ PASS' if tinker_ok else '❌ FAIL'}")
    print("=" * 70)

    if tinker_ok:
        print("\n🎉 All tests passed! Ready to run training.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
