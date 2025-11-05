#!/usr/bin/env python3
"""
Simple search test to debug why we're getting 0 results
"""

import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def test_simple_search():
    """Test search with minimal filters"""
    
    # Test 1: No filters at all (but with user_id)
    print("=== Test 1: Only user_id ===")
    response = requests.post(f"{BASE_URL}/api/v1/search", json={"user_id": 123})
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Results: {len(data.get('results', []))}")
        print(f"Search type: {data.get('search_type', 'unknown')}")
        print(f"Candidates found: {data.get('candidates_found', 0)}")
        print(f"PGVector available: {data.get('pgvector_available', False)}")
    else:
        print(f"Error: {response.text}")
    
    # Test 2: Only limit
    print("\n=== Test 2: user_id + limit ===")
    response = requests.post(f"{BASE_URL}/api/v1/search", json={"user_id": 123, "limit": 5})
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Results: {len(data.get('results', []))}")
        print(f"Search type: {data.get('search_type', 'unknown')}")
        if data.get('results'):
            print("First result:")
            print(json.dumps(data['results'][0], indent=2, default=str))
    else:
        print(f"Error: {response.text}")
    
    # Test 3: Position filter
    print("\n=== Test 3: Position filter ===")
    response = requests.post(f"{BASE_URL}/api/v1/search", json={
        "user_id": 123,
        "position": "any",
        "limit": 5
    })
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Results: {len(data.get('results', []))}")
        print(f"Search type: {data.get('search_type', 'unknown')}")
        if data.get('results'):
            print("First result:")
            print(json.dumps(data['results'][0], indent=2, default=str))
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    test_simple_search()