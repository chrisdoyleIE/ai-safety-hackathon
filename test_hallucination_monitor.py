#!/usr/bin/env python3
"""
Simple test script for the hallucination monitor.
Tests both the semantic entropy implementation and Langfuse integration.
"""

import os
import asyncio
from dotenv import load_dotenv
from hallucination_monitor.monitor import HallucinationMonitor

load_dotenv()

async def test_hallucination_monitor():
    """Test the hallucination monitor with some example prompts."""
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        return
    
    print("🧠 Testing Hallucination Monitor")
    print("=" * 50)
    
    monitor = HallucinationMonitor()
    
    # Test cases: factual vs potentially confabulative prompts
    test_cases = [
        {
            "prompt": "What is the capital of France?",
            "response": "Paris",
            "expected": "Low confabulation (factual question)"
        },
        {
            "prompt": "What is the target of Sotorasib?",
            "response": "KRASG12C",  
            "expected": "May vary (medical question that could confabulate)"
        },
        {
            "prompt": "What was the revenue of XYZ Corp in Q3 2024?",
            "response": "127.3 million dollars",
            "expected": "High confabulation (arbitrary/unknown data)"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}:")
        print(f"Prompt: {test_case['prompt']}")
        print(f"Response: {test_case['response']}")
        print(f"Expected: {test_case['expected']}")
        print("Analyzing...")
        
        try:
            # Test the monitor
            result = await monitor.monitor_response(test_case["prompt"], test_case["response"])
            
            print(f"✅ Results:")
            print(f"  - Confabulation Score: {result.score:.2f}")
            print(f"  - Semantic Entropy: {result.semantic_entropy:.3f}")
            print(f"  - Likely Confabulation: {result.likely_confabulation}")
            print(f"  - Num Clusters: {result.num_clusters}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n{'=' * 50}")
    print("🎉 Testing completed!")
    print("The monitor is now integrated with the Streamlit app and will send")
    print("confabulation_likelihood scores to your Langfuse instance at localhost:3000")

def test_sync():
    """Test the synchronous wrapper."""
    print("\n🔧 Testing synchronous wrapper...")
    
    monitor = HallucinationMonitor()
    
    try:
        result = monitor.monitor_response_sync(
            "What is 2+2?", 
            "4"
        )
        print(f"✅ Sync test passed: score={result.score}")
    except Exception as e:
        print(f"❌ Sync test failed: {e}")

if __name__ == "__main__":
    # Run async tests
    asyncio.run(test_hallucination_monitor())
    
    # Run sync test
    test_sync()