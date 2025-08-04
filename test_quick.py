import requests
import json

def quick_test():
    """Quick test untuk Islamic curriculum"""
    
    # Test Islamic query
    islamic_test = {
        "question": "Jelaskan rukun Islam yang lima",
        "use_cache": False
    }
    
    # Test general query
    general_test = {
        "question": "Bagaimana cara menghitung luas segitiga?", 
        "use_cache": False
    }
    
    print("🧪 Quick Islamic Curriculum Test")
    print("=" * 40)
    
    try:
        # Test Islamic content
        print("1. Testing Islamic content...")
        response1 = requests.post("http://localhost:8000/ask", json=islamic_test, timeout=30)
        
        if response1.status_code == 200:
            data1 = response1.json()
            print(f"   ✅ Islamic query: {data1['processing_time']:.2f}s")
            print(f"   🕌 Subject: {data1.get('subject', 'unknown')}")
            print(f"   📚 Sources: {len(data1.get('sources', []))}")
        
        # Test general content
        print("2. Testing general content...")
        response2 = requests.post("http://localhost:8000/ask", json=general_test, timeout=30)
        
        if response2.status_code == 200:
            data2 = response2.json()
            print(f"   ✅ General query: {data2['processing_time']:.2f}s")
            print(f"   📐 Subject: {data2.get('subject', 'unknown')}")
            print(f"   📚 Sources: {len(data2.get('sources', []))}")
        
        print("\n🎉 Quick test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    quick_test()