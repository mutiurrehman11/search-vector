#!/usr/bin/env python3
"""
Comprehensive test script for the Search Vector API endpoints.
This script tests all available endpoints with sample data.

Usage:
    python test_endpoints.py

Make sure the Flask app is running on http://127.0.0.1:5000 before running this script.
"""

import requests
import json
import time
from typing import Dict, Any, List

# Configuration
BASE_URL = "http://127.0.0.1:5000"
HEADERS = {"Content-Type": "application/json"}

class APITester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
        
    def log_test(self, endpoint: str, method: str, status: str, details: str = ""):
        """Log test results"""
        result = {
            "endpoint": endpoint,
            "method": method,
            "status": status,
            "details": details,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.test_results.append(result)
        
        # Color coding for console output
        color = "\033[92m" if status == "PASS" else "\033[91m"  # Green for PASS, Red for FAIL
        reset = "\033[0m"
        print(f"{color}[{status}]{reset} {method} {endpoint} - {details}")
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        print("\n=== Testing Health Endpoint ===")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                self.log_test("/health", "GET", "PASS", 
                            f"Status: {data.get('status')}, DB: {data.get('services', {}).get('database')}")
                return data
            else:
                self.log_test("/health", "GET", "FAIL", f"Status code: {response.status_code}")
                return None
        except Exception as e:
            self.log_test("/health", "GET", "FAIL", f"Error: {str(e)}")
            return None
    
    def test_search_endpoint(self):
        """Test the search endpoint with various filters"""
        print("\n=== Testing Search Endpoint ===")
        
        # Test 1: Basic search with position filter
        test_data = {
            "user_id": 1,
            "position": "midfielder",
            "limit": 5
        }
        
        try:
            response = self.session.post(f"{self.base_url}/api/v1/search", 
                                       json=test_data, headers=HEADERS)
            if response.status_code == 200:
                data = response.json()
                self.log_test("/api/v1/search", "POST", "PASS", 
                            f"Found {data.get('total', 0)} players, search_type: {data.get('search_type', 'unknown')}")
                return data.get('results', [])
            else:
                self.log_test("/api/v1/search", "POST", "FAIL", 
                            f"Status: {response.status_code}, Response: {response.text}")
                return []
        except Exception as e:
            self.log_test("/api/v1/search", "POST", "FAIL", f"Exception: {str(e)}")
            return []
        
        # Test 2: Geographic search
        geo_test_data = {
            "user_id": 1,
            "latitude": 40.7128,
            "longitude": -74.0060,
            "max_distance_km": 50,
            "limit": 3
        }
        
        try:
            response = self.session.post(f"{self.base_url}/api/v1/search", 
                                       json=geo_test_data, headers=HEADERS)
            if response.status_code == 200:
                data = response.json()
                self.log_test("/api/v1/search (geo)", "POST", "PASS", 
                            f"Found {data.get('total', 0)} players near NYC, pgvector: {data.get('pgvector_available', False)}")
            else:
                self.log_test("/api/v1/search (geo)", "POST", "FAIL", 
                            f"Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            self.log_test("/api/v1/search (geo)", "POST", "FAIL", f"Exception: {str(e)}")
        
        # Test 3: Skill level filter
        skill_test_data = {
            "user_id": 1,
            "min_skill": 8,
            "max_skill": 10,
            "limit": 3
        }
        
        try:
            response = self.session.post(f"{self.base_url}/api/v1/search", 
                                       json=skill_test_data, headers=HEADERS)
            if response.status_code == 200:
                data = response.json()
                self.log_test("/api/v1/search (skill)", "POST", "PASS", 
                            f"Found {data.get('total', 0)} high-skill players")
            else:
                self.log_test("/api/v1/search (skill)", "POST", "FAIL", 
                            f"Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            self.log_test("/api/v1/search (skill)", "POST", "FAIL", f"Exception: {str(e)}")
    
    def test_recommendations_endpoint(self, results: List):
        """Test the recommendations endpoint"""
        print("\n=== Testing Recommendations Endpoint ===")
        
        if not results or len(results) == 0:
            self.log_test("/api/v1/recommendations", "GET", "SKIP", "No player results available")
            return
        
        player_id = results[0].get('id')
        if not player_id:
            self.log_test("/api/v1/recommendations", "GET", "SKIP", "No player ID in results")
            return
        
        try:
            # Test with different limits
            for limit in [3, 5, 10]:
                response = self.session.get(f"{self.base_url}/api/v1/recommendations/{player_id}?limit={limit}")
                
                if response.status_code == 200:
                    data = response.json()
                    total_recs = data.get('total', 0)
                    self.log_test("/api/v1/recommendations", "GET", "PASS", 
                                f"Found {total_recs} recommendations (limit={limit})")
                else:
                    self.log_test("/api/v1/recommendations", "GET", "FAIL", 
                                f"Status code: {response.status_code} (limit={limit})")
        except Exception as e:
            self.log_test("/api/v1/recommendations", "GET", "FAIL", f"Error: {str(e)}")
    
    def test_events_endpoint(self, player_id: str):
        """Test the events endpoint with different event types"""
        print("\n=== Testing Events Endpoint ===")
        
        if not player_id:
            self.log_test("/api/v1/events", "POST", "SKIP", "No player ID available")
            return
        
        # Valid event types from the schema
        event_types = ["impression", "profile_view", "follow", "message", "save_to_playlist"]
        
        for i, event_type in enumerate(event_types, 1):
            try:
                payload = {
                    "user_id": 1,
                    "player_id": player_id,
                    "event_type": event_type,
                    "result_position": i
                }
                
                response = self.session.post(f"{self.base_url}/api/v1/events", 
                                           json=payload, headers=HEADERS)
                
                if response.status_code == 200:
                    self.log_test("/api/v1/events", "POST", "PASS", 
                                f"Logged '{event_type}' event")
                else:
                    self.log_test("/api/v1/events", "POST", "FAIL", 
                                f"Failed to log '{event_type}': {response.status_code}")
            except Exception as e:
                self.log_test("/api/v1/events", "POST", "FAIL", 
                            f"Error logging '{event_type}': {str(e)}")
    
    def test_error_cases(self):
        """Test various error cases and edge conditions"""
        print("\n=== Testing Error Cases ===")
        
        # Test 1: Invalid endpoint
        try:
            response = self.session.get(f"{self.base_url}/api/v1/invalid")
            if response.status_code == 404:
                self.log_test("/api/v1/invalid", "GET", "PASS", "Correctly returned 404")
            else:
                self.log_test("/api/v1/invalid", "GET", "FAIL", f"Expected 404, got {response.status_code}")
        except Exception as e:
            self.log_test("/api/v1/invalid", "GET", "FAIL", f"Error: {str(e)}")
        
        # Test 2: Invalid search payload
        try:
            payload = {"invalid_field": "test"}
            response = self.session.post(f"{self.base_url}/api/v1/search", 
                                       json=payload, headers=HEADERS)
            if response.status_code == 400:
                self.log_test("/api/v1/search", "POST", "PASS", "Correctly rejected invalid payload")
            else:
                self.log_test("/api/v1/search", "POST", "FAIL", 
                            f"Expected 400 for invalid payload, got {response.status_code}")
        except Exception as e:
            self.log_test("/api/v1/search", "POST", "FAIL", f"Error: {str(e)}")
        
        # Test 3: Invalid event type
        try:
            payload = {
                "user_id": 1,
                "player_id": "test_id",
                "event_type": "invalid_event",
                "result_position": 1
            }
            response = self.session.post(f"{self.base_url}/api/v1/events", 
                                       json=payload, headers=HEADERS)
            if response.status_code == 400:
                self.log_test("/api/v1/events", "POST", "PASS", "Correctly rejected invalid event type")
            else:
                self.log_test("/api/v1/events", "POST", "FAIL", 
                            f"Expected 400 for invalid event, got {response.status_code}")
        except Exception as e:
            self.log_test("/api/v1/events", "POST", "FAIL", f"Error: {str(e)}")
    
    def run_all_tests(self):
        """Run all endpoint tests"""
        print("🚀 Starting comprehensive API endpoint testing...")
        print(f"Testing API at: {self.base_url}")
        print("=" * 60)
        
        # Test health endpoint first
        health_data = self.test_health_endpoint()
        
        # Test search endpoint and get search results
        search_results = self.test_search_endpoint()
        
        # Test recommendations endpoint
        self.test_recommendations_endpoint(search_results)
        
        # Test events endpoint - get player ID from search results
        player_id = search_results[0].get('id') if search_results else None
        self.test_events_endpoint(player_id)
        
        # Test error cases
        self.test_error_cases()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['status'] == 'PASS'])
        failed_tests = len([r for r in self.test_results if r['status'] == 'FAIL'])
        skipped_tests = len([r for r in self.test_results if r['status'] == 'SKIP'])
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⏭️  Skipped: {skipped_tests}")
        
        if failed_tests > 0:
            print(f"\n🔍 Failed Tests:")
            for result in self.test_results:
                if result['status'] == 'FAIL':
                    print(f"  - {result['method']} {result['endpoint']}: {result['details']}")
        
        success_rate = (passed_tests / (total_tests - skipped_tests)) * 100 if (total_tests - skipped_tests) > 0 else 0
        print(f"\n📈 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("🎉 Excellent! API is working great!")
        elif success_rate >= 70:
            print("👍 Good! Most endpoints are working.")
        else:
            print("⚠️  Some issues detected. Check failed tests above.")

def main():
    """Main function to run the tests"""
    print("🧪 Search Vector API Endpoint Tester")
    print("=" * 40)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"✅ Server is running at {BASE_URL}")
    except requests.exceptions.RequestException:
        print(f"❌ Server is not running at {BASE_URL}")
        print("Please start the Flask app with: python app.py")
        return
    
    # Run tests
    tester = APITester(BASE_URL)
    tester.run_all_tests()
    
    # Save results to file
    with open('test_results.json', 'w') as f:
        json.dump(tester.test_results, f, indent=2)
    print(f"\n💾 Detailed results saved to: test_results.json")

if __name__ == "__main__":
    main()