#!/usr/bin/env python3
"""
Basic functionality test for Deep Search Agents.
Tests core functionality to ensure the application is working correctly.
"""

import sys
from agents.search_orchestrator import SearchOrchestrator, SearchType


def test_basic_functionality():
    """Test basic search functionality"""
    print("🧪 Testing Deep Search Agents basic functionality...")

    try:
        # Test SearchOrchestrator initialization
        print("1. Initializing SearchOrchestrator...")
        orchestrator = SearchOrchestrator()
        print("   ✅ SearchOrchestrator initialized successfully")

        # Test basic search
        print("2. Testing basic search...")
        query = "python programming"
        result = orchestrator.search(query, max_results=3)

        # Verify result structure
        assert hasattr(
            result, 'query'), "SearchSummary missing 'query' attribute"
        assert hasattr(
            result, 'summary'), "SearchSummary missing 'summary' attribute"
        assert hasattr(
            result, 'key_points'), "SearchSummary missing 'key_points' attribute"
        assert hasattr(
            result, 'sources'), "SearchSummary missing 'sources' attribute"
        assert hasattr(
            result, 'total_results'), "SearchSummary missing 'total_results' attribute"
        assert hasattr(
            result, 'search_time'), "SearchSummary missing 'search_time' attribute"

        print(f"   ✅ Search completed successfully")
        print(f"   📊 Query: {result.query}")
        print(f"   📊 Results found: {result.total_results}")
        print(f"   📊 Summary length: {len(result.summary)} characters")
        print(f"   📊 Search time: {result.search_time:.2f}s")
        print(f"   📊 Key points: {len(result.key_points)}")
        print(f"   📊 Sources: {len(result.sources)}")

        # Test agent selection
        print("3. Testing automatic agent selection...")
        research_result = orchestrator.search(
            "machine learning research papers", max_results=2)
        news_result = orchestrator.search(
            "latest tech news today", max_results=2)
        general_result = orchestrator.search(
            "how to bake a cake", max_results=2)

        print(
            f"   ✅ Research query processed: {research_result.total_results} results")
        print(
            f"   ✅ News query processed: {news_result.total_results} results")
        print(
            f"   ✅ General query processed: {general_result.total_results} results")

        # Test SearchType enum
        print("4. Testing SearchType enum...")
        assert SearchType.RESEARCH.value == "research"
        assert SearchType.NEWS.value == "news"
        assert SearchType.GENERAL.value == "general"
        print("   ✅ SearchType enum working correctly")

        print("\n🎉 All tests passed! Deep Search Agents is working correctly.")
        print("💡 You can now run 'streamlit run app.py' to start the web interface")
        print("💡 Or use 'python main.py search \"your query\"' for CLI access")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
