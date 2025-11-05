#!/usr/bin/env python3
"""
Quick test script for Search Vector API endpoints.
Simple and fast validation of all endpoints.

Usage: python quick_test.py
"""

import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def test_endpoint(method, url, data=None, params=None):
    """Test a single endpoint and return result"""
    try:
        if method == "GET":
            response = requests.get(url, params=params)
        elif method == "POST":
            response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
        
        return {
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "data": response.json() if response.status_code == 200 else None,
            "error": response.text if response.status_code != 200 else None
        }
    except Exception as e:
        return {
            "status_code": None,
            "success": False,
            "data": None,
            "error": str(e)
        }

def main():
    print("🚀 Quick API Test")
    print("=" * 30)
    
    # 1. Health Check
    print("1. Testing Health...")
    health = test_endpoint("GET", f"{BASE_URL}/health")
    print(f"   ✅ Health: {health['success']}" if health['success'] else f"   ❌ Health: {health['error']}")
    
    # 2. Search
    print("2. Testing Search...")
    search_data = {"user_id": 1, "limit": 3}
    search = test_endpoint("POST", f"{BASE_URL}/api/v1/search", search_data)
    player_id = None
    if search['success'] and search['data']['results']:
        player_id = search['data']['results'][0]['id']
        print(f"   ✅ Search: Found {search['data']['total']} players")
    else:
        print(f"   ❌ Search: {search['error']}")
    
    # 3. Recommendations
    print("3. Testing Recommendations...")
    if player_id:
        recs = test_endpoint("GET", f"{BASE_URL}/api/v1/recommendations/{player_id}", params={"limit": 3})
        print(f"   ✅ Recommendations: {recs['success']}" if recs['success'] else f"   ❌ Recommendations: {recs['error']}")
    else:
        print("   ⏭️  Recommendations: Skipped (no player ID)")
    
    # 4. Events
    print("4. Testing Events...")
    if player_id:
        event_data = {
            "user_id": 1,
            "player_id": player_id,
            "event_type": "impression",
            "result_position": 1
        }
        events = test_endpoint("POST", f"{BASE_URL}/api/v1/events", event_data)
        print(f"   ✅ Events: {events['success']}" if events['success'] else f"   ❌ Events: {events['error']}")
    else:
        print("   ⏭️  Events: Skipped (no player ID)")
    
    print("\n🎉 Quick test complete!")

if __name__ == "__main__":
    main()