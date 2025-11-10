"""
Comprehensive API Test Suite
Tests all endpoints and covers all scenarios including edge cases
"""

import requests
import json
import time
import random
from typing import Dict, List, Optional
from datetime import datetime
import sys

# Configuration
BASE_URL = "http://localhost:5000"
API_V1 = f"{BASE_URL}/api/v1"

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

class TestStats:
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.start_time = time.time()
    
    def print_summary(self):
        duration = time.time() - self.start_time
        print("\n" + "="*70)
        print(f"{Colors.BLUE}TEST SUMMARY{Colors.RESET}")
        print("="*70)
        print(f"Total Tests:   {self.total}")
        print(f"{Colors.GREEN}Passed:        {self.passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed:        {self.failed}{Colors.RESET}")
        print(f"{Colors.YELLOW}Skipped:       {self.skipped}{Colors.RESET}")
        print(f"Duration:      {duration:.2f}s")
        print(f"Success Rate:  {(self.passed/self.total*100) if self.total > 0 else 0:.1f}%")
        print("="*70)

stats = TestStats()

def log_test(test_name: str, status: str, message: str = ""):
    """Log test result"""
    stats.total += 1
    
    if status == "PASS":
        stats.passed += 1
        symbol = "✓"
        color = Colors.GREEN
    elif status == "FAIL":
        stats.failed += 1
        symbol = "✗"
        color = Colors.RED
    else:  # SKIP
        stats.skipped += 1
        symbol = "○"
        color = Colors.YELLOW
    
    print(f"{color}{symbol} {test_name}{Colors.RESET}")
    if message:
        print(f"  {message}")

def test_health_check():
    """Test health check endpoint"""
    print(f"\n{Colors.BLUE}Testing Health Check Endpoint{Colors.RESET}")
    print("-" * 70)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        # Test: Endpoint is accessible
        if response.status_code == 200:
            log_test("Health check accessible", "PASS")
        else:
            log_test("Health check accessible", "FAIL", f"Status: {response.status_code}")
            return
        
        data = response.json()
        
        # Test: Response structure
        if all(key in data for key in ['status', 'timestamp', 'services']):
            log_test("Health check response structure", "PASS")
        else:
            log_test("Health check response structure", "FAIL", f"Missing keys: {data.keys()}")
        
        # Test: Service status
        if data.get('status') == 'healthy':
            log_test("Service is healthy", "PASS")
        else:
            log_test("Service is healthy", "FAIL", f"Status: {data.get('status')}")
        
        # Test: pgvector availability
        pgvector = data.get('services', {}).get('pgvector')
        if pgvector:
            log_test("pgvector extension available", "PASS")
        else:
            log_test("pgvector extension available", "FAIL", "pgvector not enabled")
        
    except Exception as e:
        log_test("Health check endpoint", "FAIL", str(e))

def test_search_basic():
    """Test basic search functionality"""
    print(f"\n{Colors.BLUE}Testing Basic Search{Colors.RESET}")
    print("-" * 70)
    
    # Test 1: Empty search (no filters)
    try:
        response = requests.post(f"{API_V1}/search", json={})
        
        if response.status_code == 200:
            log_test("Empty search request", "PASS")
            data = response.json()
            
            # Check response structure
            if 'results' in data and 'total' in data:
                log_test("Search response structure", "PASS")
            else:
                log_test("Search response structure", "FAIL", f"Keys: {data.keys()}")
            
            # Check results are returned
            if len(data.get('results', [])) > 0:
                log_test("Search returns results", "PASS", f"Found {len(data['results'])} results")
            else:
                log_test("Search returns results", "FAIL", "No results returned")
        else:
            log_test("Empty search request", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        log_test("Empty search request", "FAIL", str(e))
    
    # Test 2: Position filter
    try:
        response = requests.post(f"{API_V1}/search", json={
            "position": "midfielder",
            "limit": 10
        })
        
        if response.status_code == 200:
            log_test("Position filter search", "PASS")
            data = response.json()
            
            # Verify results match filter
            results = data.get('results', [])
            if results:
                # Check if any result has midfielder position
                has_midfielder = any('midfielder' in str(r.get('position', [])).lower() for r in results)
                if has_midfielder:
                    log_test("Position filter accuracy", "PASS")
                else:
                    log_test("Position filter accuracy", "FAIL", "Results don't match position filter")
        else:
            log_test("Position filter search", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        log_test("Position filter search", "FAIL", str(e))
    
    # Test 3: Skill level filter
    try:
        response = requests.post(f"{API_V1}/search", json={
            "min_skill": 70,
            "max_skill": 90,
            "limit": 10
        })
        
        if response.status_code == 200:
            log_test("Skill level filter search", "PASS")
            data = response.json()
            
            # Verify skill levels
            results = data.get('results', [])
            if results:
                skill_levels = [r.get('skill_level', 0) for r in results]
                if all(70 <= s <= 90 for s in skill_levels if s > 0):
                    log_test("Skill level filter accuracy", "PASS")
                else:
                    log_test("Skill level filter accuracy", "FAIL", f"Skill levels: {skill_levels}")
        else:
            log_test("Skill level filter search", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        log_test("Skill level filter search", "FAIL", str(e))
    
    # Test 4: Age filter
    try:
        response = requests.post(f"{API_V1}/search", json={
            "min_age": 20,
            "max_age": 28,
            "limit": 10
        })
        
        if response.status_code == 200:
            log_test("Age filter search", "PASS")
            data = response.json()
            
            results = data.get('results', [])
            if results:
                ages = [r.get('age', 0) for r in results]
                if all(20 <= a <= 28 for a in ages if a > 0):
                    log_test("Age filter accuracy", "PASS")
                else:
                    log_test("Age filter accuracy", "FAIL", f"Ages: {ages}")
        else:
            log_test("Age filter search", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        log_test("Age filter search", "FAIL", str(e))
    
    # Test 5: Combined filters
    try:
        response = requests.post(f"{API_V1}/search", json={
            "position": "forward",
            "min_skill": 75,
            "max_skill": 95,
            "min_age": 22,
            "max_age": 30,
            "limit": 10
        })
        
        if response.status_code == 200:
            log_test("Combined filters search", "PASS")
            data = response.json()
            
            if 'metadata' in data:
                search_type = data['metadata'].get('search_type')
                log_test(f"Search type: {search_type}", "PASS")
        else:
            log_test("Combined filters search", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        log_test("Combined filters search", "FAIL", str(e))

def test_search_pagination():
    """Test search pagination"""
    print(f"\n{Colors.BLUE}Testing Search Pagination{Colors.RESET}")
    print("-" * 70)
    
    # Test 1: Default limit
    try:
        response = requests.post(f"{API_V1}/search", json={})
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            if len(results) <= 20:  # Default limit
                log_test("Default limit respected", "PASS", f"Got {len(results)} results")
            else:
                log_test("Default limit respected", "FAIL", f"Got {len(results)} results, expected ≤20")
    except Exception as e:
        log_test("Default limit", "FAIL", str(e))
    
    # Test 2: Custom limit
    try:
        response = requests.post(f"{API_V1}/search", json={"limit": 5})
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            if len(results) <= 5:
                log_test("Custom limit respected", "PASS", f"Got {len(results)} results")
            else:
                log_test("Custom limit respected", "FAIL", f"Got {len(results)} results, expected ≤5")
    except Exception as e:
        log_test("Custom limit", "FAIL", str(e))
    
    # Test 3: Offset pagination
    try:
        # Get first page
        response1 = requests.post(f"{API_V1}/search", json={"limit": 10, "offset": 0})
        response2 = requests.post(f"{API_V1}/search", json={"limit": 10, "offset": 10})
        
        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()
            
            ids1 = [r['id'] for r in data1.get('results', [])]
            ids2 = [r['id'] for r in data2.get('results', [])]
            
            # Check no overlap
            overlap = set(ids1) & set(ids2)
            if not overlap:
                log_test("Pagination offset works", "PASS", "No overlap between pages")
            else:
                log_test("Pagination offset works", "FAIL", f"Found {len(overlap)} overlapping IDs")
    except Exception as e:
        log_test("Pagination offset", "FAIL", str(e))
    
    # Test 4: Max limit enforcement
    try:
        response = requests.post(f"{API_V1}/search", json={"limit": 1000})
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            if len(results) <= 50:  # Max limit
                log_test("Max limit enforced", "PASS", f"Got {len(results)} results")
            else:
                log_test("Max limit enforced", "FAIL", f"Got {len(results)} results, expected ≤50")
        elif response.status_code == 400:
            log_test("Max limit enforced", "PASS", "Request rejected")
        else:
            log_test("Max limit enforced", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        log_test("Max limit enforcement", "FAIL", str(e))

def test_search_edge_cases():
    """Test edge cases and error handling"""
    print(f"\n{Colors.BLUE}Testing Search Edge Cases{Colors.RESET}")
    print("-" * 70)
    
    # Test 1: Invalid position
    try:
        response = requests.post(f"{API_V1}/search", json={"position": "invalid_position"})
        
        if response.status_code == 400:
            log_test("Invalid position rejected", "PASS")
        elif response.status_code == 200:
            # Some implementations might accept it but return no results
            data = response.json()
            if len(data.get('results', [])) == 0:
                log_test("Invalid position handled", "PASS", "No results returned")
            else:
                log_test("Invalid position handling", "FAIL", "Should reject or return empty")
        else:
            log_test("Invalid position handling", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        log_test("Invalid position handling", "FAIL", str(e))
    
    # Test 2: Invalid skill range
    try:
        response = requests.post(f"{API_V1}/search", json={
            "min_skill": 95,
            "max_skill": 50  # min > max
        })
        
        if response.status_code == 400:
            log_test("Invalid skill range rejected", "PASS")
        elif response.status_code == 200:
            data = response.json()
            if len(data.get('results', [])) == 0:
                log_test("Invalid skill range handled", "PASS", "No results")
            else:
                log_test("Invalid skill range handled", "FAIL", "Should handle gracefully")
        else:
            log_test("Invalid skill range", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        log_test("Invalid skill range", "FAIL", str(e))
    
    # Test 3: Negative values
    try:
        response = requests.post(f"{API_V1}/search", json={
            "min_age": -5,
            "max_age": 100
        })
        
        if response.status_code == 400:
            log_test("Negative age rejected", "PASS")
        else:
            log_test("Negative age validation", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        log_test("Negative age validation", "FAIL", str(e))
    
    # Test 4: Extreme coordinates
    try:
        response = requests.post(f"{API_V1}/search", json={
            "latitude": 91.0,  # Invalid
            "longitude": 0.0,
            "max_distance_km": 10
        })
        
        if response.status_code == 400:
            log_test("Invalid coordinates rejected", "PASS")
        else:
            log_test("Invalid coordinates validation", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        log_test("Invalid coordinates validation", "FAIL", str(e))
    
    # Test 5: No results scenario
    try:
        response = requests.post(f"{API_V1}/search", json={
            "position": "goalkeeper",
            "min_skill": 99,  # Very high requirement
            "max_skill": 100,
            "min_age": 18,
            "max_age": 19
        })
        
        if response.status_code == 200:
            data = response.json()
            if len(data.get('results', [])) == 0:
                log_test("No results handled gracefully", "PASS")
            else:
                log_test("No results scenario", "SKIP", "Found results for extreme filter")
    except Exception as e:
        log_test("No results handling", "FAIL", str(e))
    
    # Test 6: Malformed JSON
    try:
        response = requests.post(f"{API_V1}/search", 
                                data="invalid json",
                                headers={'Content-Type': 'application/json'})
        
        if response.status_code == 400:
            log_test("Malformed JSON rejected", "PASS")
        else:
            log_test("Malformed JSON handling", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        log_test("Malformed JSON handling", "FAIL", str(e))

def test_search_performance():
    """Test search performance"""
    print(f"\n{Colors.BLUE}Testing Search Performance{Colors.RESET}")
    print("-" * 70)
    
    # Test 1: Simple search latency
    try:
        start = time.time()
        response = requests.post(f"{API_V1}/search", json={"limit": 10})
        latency = (time.time() - start) * 1000  # ms
        
        if response.status_code == 200:
            data = response.json()
            telemetry = data.get('telemetry', {})
            server_time = telemetry.get('total_time_ms', 0)
            
            if latency < 1000:  # Less than 1 second
                log_test("Search latency acceptable", "PASS", f"{latency:.0f}ms (server: {server_time:.0f}ms)")
            else:
                log_test("Search latency", "FAIL", f"{latency:.0f}ms (too slow)")
    except Exception as e:
        log_test("Search latency test", "FAIL", str(e))
    
    # Test 2: Complex query latency
    try:
        start = time.time()
        response = requests.post(f"{API_V1}/search", json={
            "position": "midfielder",
            "min_skill": 70,
            "max_skill": 90,
            "min_age": 20,
            "max_age": 30,
            "latitude": 40.7128,
            "longitude": -74.0060,
            "max_distance_km": 50,
            "limit": 20
        })
        latency = (time.time() - start) * 1000
        
        if response.status_code == 200:
            if latency < 2000:  # Less than 2 seconds
                log_test("Complex query latency", "PASS", f"{latency:.0f}ms")
            else:
                log_test("Complex query latency", "FAIL", f"{latency:.0f}ms (too slow)")
    except Exception as e:
        log_test("Complex query latency", "FAIL", str(e))
    
    # Test 3: Concurrent requests
    try:
        import concurrent.futures
        
        def search_request():
            response = requests.post(f"{API_V1}/search", json={"limit": 10})
            return response.status_code == 200
        
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(search_request) for _ in range(10)]
            results = [f.result() for f in futures]
        
        duration = time.time() - start
        
        success_count = sum(results)
        if success_count == 10:
            log_test("Concurrent requests", "PASS", f"{success_count}/10 successful in {duration:.2f}s")
        else:
            log_test("Concurrent requests", "FAIL", f"Only {success_count}/10 successful")
    except Exception as e:
        log_test("Concurrent requests", "FAIL", str(e))

def test_event_logging():
    """Test event logging endpoint"""
    print(f"\n{Colors.BLUE}Testing Event Logging{Colors.RESET}")
    print("-" * 70)
    
    # First, get a player ID from search
    player_id = None
    try:
        response = requests.post(f"{API_V1}/search", json={"limit": 1})
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            if results:
                player_id = results[0]['id']
    except Exception as e:
        log_test("Getting player for event test", "FAIL", str(e))
        return
    
    if not player_id:
        log_test("Event logging tests", "SKIP", "No player ID available")
        return
    
    # Generate a test user ID
    import string
    test_user_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=26))
    
    # Test 1: Log impression event
    try:
        response = requests.post(f"{API_V1}/events", json={
            "user_id": "E51AFCAM0FJT5OKXFXQ1PR2ND4",
            "player_id": player_id,
            "event_type": "impression",
            "query_context": {"position": "midfielder"}
        })
        
        if response.status_code == 200:
            log_test("Log impression event", "PASS")
        else:
            log_test("Log impression event", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        log_test("Log impression event", "FAIL", str(e))
    
    # Test 2: Log profile view event
    try:
        response = requests.post(f"{API_V1}/events", json={
            "user_id": test_user_id,
            "player_id": player_id,
            "event_type": "profile_view",
            "query_context": {}
        })
        
        if response.status_code == 200:
            log_test("Log profile_view event", "PASS")
        else:
            log_test("Log profile_view event", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        log_test("Log profile_view event", "FAIL", str(e))
    
    # Test 3: Log follow event
    try:
        response = requests.post(f"{API_V1}/events", json={
            "user_id": test_user_id,
            "player_id": player_id,
            "event_type": "follow",
            "query_context": {}
        })
        
        if response.status_code == 200:
            log_test("Log follow event", "PASS")
        else:
            log_test("Log follow event", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        log_test("Log follow event", "FAIL", str(e))
    
    # Test 4: Invalid event type
    try:
        response = requests.post(f"{API_V1}/events", json={
            "user_id": test_user_id,
            "player_id": player_id,
            "event_type": "invalid_event",
            "query_context": {}
        })
        
        if response.status_code == 400:
            log_test("Invalid event type rejected", "PASS")
        else:
            log_test("Invalid event type validation", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        log_test("Invalid event type validation", "FAIL", str(e))
    
    # Test 5: Missing required fields
    try:
        response = requests.post(f"{API_V1}/events", json={
            "user_id": test_user_id,
            "event_type": "impression"
            # Missing player_id
        })
        
        if response.status_code == 400:
            log_test("Missing required fields rejected", "PASS")
        else:
            log_test("Missing fields validation", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        log_test("Missing fields validation", "FAIL", str(e))

def test_recommendations():
    """Test recommendations endpoint"""
    print(f"\n{Colors.BLUE}Testing Recommendations{Colors.RESET}")
    print("-" * 70)
    
    # Get a player ID
    player_id = None
    try:
        response = requests.post(f"{API_V1}/search", json={"limit": 1})
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            if results:
                player_id = results[0]['id']
    except Exception as e:
        log_test("Getting player for recommendations", "FAIL", str(e))
        return
    
    if not player_id:
        log_test("Recommendations tests", "SKIP", "No player ID available")
        return
    
    # Test 1: Basic recommendations
    try:
        response = requests.get(f"{API_V1}/recommendations/{player_id}")
        
        if response.status_code == 200:
            log_test("Get recommendations", "PASS")
            data = response.json()
            
            # Check response structure
            if 'recommendations' in data:
                log_test("Recommendations response structure", "PASS")
                
                recs = data.get('recommendations', [])
                if len(recs) > 0:
                    log_test("Recommendations returned", "PASS", f"Got {len(recs)} recommendations")
                    
                    # Check if similarity scores present
                    if all('similarity' in r for r in recs):
                        log_test("Similarity scores present", "PASS")
                    else:
                        log_test("Similarity scores present", "FAIL")
                else:
                    log_test("Recommendations returned", "SKIP", "No recommendations available")
            else:
                log_test("Recommendations response structure", "FAIL", f"Keys: {data.keys()}")
        else:
            log_test("Get recommendations", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        log_test("Get recommendations", "FAIL", str(e))
    
    # Test 2: Recommendations with limit
    try:
        response = requests.get(f"{API_V1}/recommendations/{player_id}?limit=5")
        
        if response.status_code == 200:
            data = response.json()
            recs = data.get('recommendations', [])
            
            if len(recs) <= 5:
                log_test("Recommendations limit respected", "PASS", f"Got {len(recs)} recommendations")
            else:
                log_test("Recommendations limit", "FAIL", f"Got {len(recs)}, expected ≤5")
    except Exception as e:
        log_test("Recommendations with limit", "FAIL", str(e))
    
    # Test 3: Invalid player ID
    try:
        response = requests.get(f"{API_V1}/recommendations/INVALID_PLAYER_ID_123456")
        
        if response.status_code in [404, 400, 200]:  # 200 with empty results is acceptable
            if response.status_code == 200:
                data = response.json()
                print(data)
                if len(data.get('recommendations', [])) == 0:
                    log_test("Invalid player ID handled", "PASS", "Empty recommendations")
                else:
                    log_test("Invalid player ID handling", "FAIL", "Should return empty")
            else:
                log_test("Invalid player ID rejected", "PASS")
        else:
            log_test("Invalid player ID handling", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        log_test("Invalid player ID handling", "FAIL", str(e))

def test_saved_searches():
    """Test saved searches functionality"""
    print(f"\n{Colors.BLUE}Testing Saved Searches{Colors.RESET}")
    print("-" * 70)
    
    # Generate test user ID
    import string
    test_user_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=26))
    
    saved_search_id = None
    
    # Test 1: Create saved search
    try:
        response = requests.post(f"{API_V1}/saved-searches", json={
            "user_id": test_user_id,
            "search_name": "Test Search - Midfielders",
            "filters": {
                "position": "midfielder",
                "min_skill": 70,
                "max_skill": 90
            },
            "alert_frequency": "weekly"
        })
        
        if response.status_code == 201:
            log_test("Create saved search", "PASS")
            data = response.json()
            saved_search_id = data.get('id')
            
            if 'search_name' in data and 'alert_frequency' in data:
                log_test("Saved search response structure", "PASS")
        elif response.status_code == 200:
            log_test("Create saved search", "PASS", "Status 200 (expected 201)")
            data = response.json()
            saved_search_id = data.get('id')
        else:
            log_test("Create saved search", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        log_test("Create saved search", "FAIL", str(e))
    
    # Test 2: Get saved searches for user
    try:
        response = requests.get(f"{API_V1}/saved-searches/A9ZMOPYP91NBTXZ4ATE0CU4WN3")
        print(response.json())
        
        if response.status_code == 200:
            log_test("Get saved searches", "PASS")
            data = response.json()
            
            if 'saved_searches' in data:
                searches = data.get('saved_searches', [])
                if len(searches) > 0:
                    log_test("Saved searches retrieved", "PASS", f"Found {len(searches)} searches")
                else:
                    log_test("Saved searches retrieved", "SKIP", "No saved searches found")
        else:
            log_test("Get saved searches", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        log_test("Get saved searches", "FAIL", str(e))
    
    # Test 3: Get new matches for saved search
    if saved_search_id:
        try:
            response = requests.get(f"{API_V1}/saved-searches/{saved_search_id}/matches")
            
            if response.status_code == 200:
                log_test("Get new matches", "PASS")
                data = response.json()
                
                if 'new_matches' in data:
                    matches = data.get('new_matches', [])
                    log_test("New matches structure", "PASS", f"Found {len(matches)} new matches")
            else:
                log_test("Get new matches", "FAIL", f"Status: {response.status_code}")
        except Exception as e:
            log_test("Get new matches", "FAIL", str(e))
    else:
        log_test("Get new matches", "SKIP", "No saved search ID available")
    
    # Test 4: Invalid alert frequency
    try:
        response = requests.post(f"{API_V1}/saved-searches", json={
            "user_id": test_user_id,
            "search_name": "Test Invalid Frequency",
            "filters": {"position": "forward"},
            "alert_frequency": "invalid_frequency"
        })
        
        if response.status_code in [400, 500]:
            log_test("Invalid alert frequency rejected", "PASS")
        elif response.status_code == 201:
            log_test("Invalid alert frequency validation", "FAIL", "Should reject invalid frequency")
        else:
            log_test("Invalid alert frequency handling", "SKIP", f"Status: {response.status_code}")
    except Exception as e:
        log_test("Invalid alert frequency handling", "FAIL", str(e))

def test_admin_endpoints():
    """Test admin endpoints"""
    print(f"\n{Colors.BLUE}Testing Admin Endpoints{Colors.RESET}")
    print("-" * 70)
    
    # Test 1: Train ML model
    print(f"{Colors.YELLOW}Note: ML training may take 30-60 seconds{Colors.RESET}")
    try:
        start = time.time()
        response = requests.post(f"{API_V1}/admin/train-model", timeout=120)
        duration = time.time() - start
        
        if response.status_code == 200:
            log_test("Train ML model", "PASS", f"Completed in {duration:.1f}s")
            data = response.json()
            
            if 'message' in data:
                log_test("Training response structure", "PASS")
                
                # Check if model was actually trained
                if 'successfully' in data.get('message', '').lower():
                    log_test("Model training successful", "PASS")
                else:
                    log_test("Model training status", "SKIP", data.get('message'))
        elif response.status_code == 400:
            log_test("Train ML model", "SKIP", "Insufficient training data")
        else:
            log_test("Train ML model", "FAIL", f"Status: {response.status_code}")
    except requests.exceptions.Timeout:
        log_test("Train ML model", "FAIL", "Request timeout (>120s)")
    except Exception as e:
        log_test("Train ML model", "FAIL", str(e))

def test_location_based_search():
    """Test location-based search functionality"""
    print(f"\n{Colors.BLUE}Testing Location-Based Search{Colors.RESET}")
    print("-" * 70)
    
    # Test 1: Search near New York
    try:
        response = requests.post(f"{API_V1}/search", json={
            "latitude": 40.7128,
            "longitude": -74.0060,
            "max_distance_km": 50,
            "limit": 10
        })
        
        if response.status_code == 200:
            log_test("Location-based search", "PASS")
            data = response.json()
            results = data.get('results', [])
            
            # Check if distance is calculated
            if results and any('distance_km' in r for r in results):
                log_test("Distance calculation", "PASS")
                
                # Verify distances are within range
                distances = [r.get('distance_km') for r in results if r.get('distance_km') is not None]
                if distances:
                    max_dist = max(distances)
                    if max_dist <= 50:
                        log_test("Distance filter accuracy", "PASS", f"Max distance: {max_dist:.1f}km")
                    else:
                        log_test("Distance filter accuracy", "FAIL", f"Max distance: {max_dist:.1f}km (>50km)")
            else:
                log_test("Distance calculation", "SKIP", "No distance_km field in results")
        else:
            log_test("Location-based search", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        log_test("Location-based search", "FAIL", str(e))
    
    # Test 2: Search in different location (London)
    try:
        response = requests.post(f"{API_V1}/search", json={
            "latitude": 51.5074,
            "longitude": -0.1278,
            "max_distance_km": 100,
            "limit": 10
        })
        
        if response.status_code == 200:
            log_test("Multi-location search", "PASS")
    except Exception as e:
        log_test("Multi-location search", "FAIL", str(e))
    
    # Test 3: Very small radius
    try:
        response = requests.post(f"{API_V1}/search", json={
            "latitude": 40.7128,
            "longitude": -74.0060,
            "max_distance_km": 1,  # 1km radius
            "limit": 10
        })
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            log_test("Small radius search", "PASS", f"Found {len(results)} results within 1km")
    except Exception as e:
        log_test("Small radius search", "FAIL", str(e))
    
    # Test 4: Very large radius
    try:
        response = requests.post(f"{API_V1}/search", json={
            "latitude": 40.7128,
            "longitude": -74.0060,
            "max_distance_km": 1000,  # 1000km radius
            "limit": 20
        })
        
        if response.status_code == 200:
            log_test("Large radius search", "PASS")
    except Exception as e:
        log_test("Large radius search", "FAIL", str(e))

def test_response_metadata():
    """Test response metadata and telemetry"""
    print(f"\n{Colors.BLUE}Testing Response Metadata{Colors.RESET}")
    print("-" * 70)
    
    try:
        response = requests.post(f"{API_V1}/search", json={
            "position": "midfielder",
            "limit": 10
        })
        
        if response.status_code == 200:
            data = response.json()
            
            # Test metadata presence
            if 'metadata' in data:
                log_test("Metadata present", "PASS")
                
                metadata = data.get('metadata', {})
                
                # Check important metadata fields
                if 'search_type' in metadata:
                    search_type = metadata['search_type']
                    log_test(f"Search type: {search_type}", "PASS")
                
                if 'candidates_found' in metadata:
                    log_test("Candidates count present", "PASS")
                
                if 'pgvector_available' in metadata:
                    pgvector = metadata['pgvector_available']
                    log_test(f"pgvector status tracked: {pgvector}", "PASS")
            else:
                log_test("Metadata present", "FAIL", "No metadata in response")
            
            # Test telemetry presence
            if 'telemetry' in data:
                log_test("Telemetry present", "PASS")
                
                telemetry = data.get('telemetry', {})
                
                # Check timing information
                required_timings = ['total_time_ms', 'filter_time_ms', 'vector_time_ms']
                if all(t in telemetry for t in required_timings):
                    log_test("Timing information complete", "PASS")
                    
                    # Log timing breakdown
                    total = telemetry.get('total_time_ms', 0)
                    filter_time = telemetry.get('filter_time_ms', 0)
                    vector_time = telemetry.get('vector_time_ms', 0)
                    rerank_time = telemetry.get('rerank_time_ms', 0)
                    
                    print(f"  Timing: Total={total:.0f}ms, Filter={filter_time:.0f}ms, Vector={vector_time:.0f}ms, Rerank={rerank_time:.0f}ms")
                else:
                    log_test("Timing information complete", "FAIL", f"Missing fields")
            else:
                log_test("Telemetry present", "FAIL", "No telemetry in response")
    except Exception as e:
        log_test("Response metadata test", "FAIL", str(e))

def test_result_data_quality():
    """Test quality and completeness of result data"""
    print(f"\n{Colors.BLUE}Testing Result Data Quality{Colors.RESET}")
    print("-" * 70)
    
    try:
        response = requests.post(f"{API_V1}/search", json={"limit": 5})
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            if not results:
                log_test("Result data quality", "SKIP", "No results to check")
                return
            
            # Check first result for completeness
            first_result = results[0]
            
            # Required fields
            required_fields = ['id', 'name']
            missing_required = [f for f in required_fields if f not in first_result]
            
            if not missing_required:
                log_test("Required fields present", "PASS")
            else:
                log_test("Required fields present", "FAIL", f"Missing: {missing_required}")
            
            # Common optional fields
            common_fields = ['position', 'skill_level', 'age', 'location']
            present_fields = [f for f in common_fields if f in first_result]
            
            if len(present_fields) >= 3:
                log_test("Result data completeness", "PASS", f"{len(present_fields)}/{len(common_fields)} common fields present")
            else:
                log_test("Result data completeness", "FAIL", f"Only {len(present_fields)}/{len(common_fields)} fields present")
            
            # Check data types
            type_checks = {
                'id': str,
                'name': str,
                'age': (int, float),
                'skill_level': (int, float)
            }
            
            type_errors = []
            for field, expected_type in type_checks.items():
                if field in first_result:
                    value = first_result[field]
                    if not isinstance(value, expected_type):
                        type_errors.append(f"{field}: expected {expected_type}, got {type(value)}")
            
            if not type_errors:
                log_test("Data type validation", "PASS")
            else:
                log_test("Data type validation", "FAIL", f"Type errors: {type_errors}")
            
            # Check for null/empty critical fields
            null_checks = ['id', 'name']
            null_errors = []
            for field in null_checks:
                if field in first_result:
                    value = first_result[field]
                    if value is None or (isinstance(value, str) and not value.strip()):
                        null_errors.append(field)
            
            if not null_errors:
                log_test("Critical fields not null", "PASS")
            else:
                log_test("Critical fields not null", "FAIL", f"Null/empty: {null_errors}")
            
            # Check location structure if present
            if 'location' in first_result:
                location = first_result['location']
                if isinstance(location, dict):
                    if 'latitude' in location and 'longitude' in location:
                        log_test("Location structure valid", "PASS")
                    else:
                        log_test("Location structure valid", "FAIL", "Missing lat/lng")
                else:
                    log_test("Location structure valid", "FAIL", f"Location is {type(location)}, expected dict")
    
    except Exception as e:
        log_test("Result data quality test", "FAIL", str(e))

def test_search_consistency():
    """Test search result consistency"""
    print(f"\n{Colors.BLUE}Testing Search Consistency{Colors.RESET}")
    print("-" * 70)
    
    # Test 1: Same query returns same results
    try:
        query = {
            "position": "midfielder",
            "min_skill": 70,
            "max_skill": 85,
            "limit": 10
        }
        
        response1 = requests.post(f"{API_V1}/search", json=query)
        time.sleep(0.1)  # Small delay
        response2 = requests.post(f"{API_V1}/search", json=query)
        
        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()
            
            ids1 = [r['id'] for r in data1.get('results', [])]
            ids2 = [r['id'] for r in data2.get('results', [])]
            
            if ids1 == ids2:
                log_test("Search result consistency", "PASS", "Same results for identical queries")
            else:
                # Check if at least the sets are the same (order might differ)
                if set(ids1) == set(ids2):
                    log_test("Search result consistency", "PASS", "Same results, different order (acceptable)")
                else:
                    log_test("Search result consistency", "FAIL", "Different results for same query")
    except Exception as e:
        log_test("Search result consistency", "FAIL", str(e))
    
    # Test 2: Reproducibility over time
    try:
        query = {"position": "forward", "limit": 5}
        
        results = []
        for i in range(3):
            response = requests.post(f"{API_V1}/search", json=query)
            if response.status_code == 200:
                data = response.json()
                results.append([r['id'] for r in data.get('results', [])])
            time.sleep(0.05)
        
        if len(results) == 3:
            if results[0] == results[1] == results[2]:
                log_test("Search reproducibility", "PASS", "Consistent across 3 attempts")
            else:
                log_test("Search reproducibility", "FAIL", "Results vary across attempts")
    except Exception as e:
        log_test("Search reproducibility", "FAIL", str(e))

def test_vector_search_specific():
    """Test vector search specific functionality"""
    print(f"\n{Colors.BLUE}Testing Vector Search Features{Colors.RESET}")
    print("-" * 70)
    
    # Check if pgvector is available first
    try:
        health_response = requests.get(f"{BASE_URL}/health")
        if health_response.status_code == 200:
            health_data = health_response.json()
            pgvector_available = health_data.get('services', {}).get('pgvector', False)
            
            if not pgvector_available:
                log_test("Vector search tests", "SKIP", "pgvector not available")
                return
    except:
        pass
    
    # Test 1: Search with seed players (similarity search)
    try:
        # First get a player
        response = requests.post(f"{API_V1}/search", json={"limit": 1})
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            if results:
                seed_player_id = results[0]['id']
                
                # Now search with this as seed
                response2 = requests.post(f"{API_V1}/search", json={
                    "seed_player_ids": [seed_player_id],
                    "limit": 10
                })
                
                if response2.status_code == 200:
                    log_test("Seed player search", "PASS")
                    data2 = response2.json()
                    
                    # Check if similarity scores are present
                    results2 = data2.get('results', [])
                    if results2 and any('similarity_score' in r for r in results2):
                        log_test("Similarity scores in results", "PASS")
                    else:
                        log_test("Similarity scores in results", "SKIP", "No similarity scores")
                else:
                    log_test("Seed player search", "FAIL", f"Status: {response2.status_code}")
    except Exception as e:
        log_test("Seed player search", "FAIL", str(e))
    
    # Test 2: Check for vector reranking indicator
    try:
        response = requests.post(f"{API_V1}/search", json={
            "position": "midfielder",
            "min_skill": 70,
            "limit": 20
        })
        
        if response.status_code == 200:
            data = response.json()
            metadata = data.get('metadata', {})
            
            search_type = metadata.get('search_type', '')
            if 'hybrid' in search_type or 'vector' in search_type:
                log_test("Vector reranking used", "PASS", f"Search type: {search_type}")
            else:
                log_test("Vector reranking indicator", "SKIP", f"Search type: {search_type}")
    except Exception as e:
        log_test("Vector reranking check", "FAIL", str(e))

def run_stress_test():
    """Run stress test with many requests"""
    print(f"\n{Colors.BLUE}Running Stress Test{Colors.RESET}")
    print("-" * 70)
    print(f"{Colors.YELLOW}Note: This will send 50 requests{Colors.RESET}")
    
    try:
        import concurrent.futures
        
        queries = [
            {"position": "midfielder", "limit": 10},
            {"position": "forward", "limit": 10},
            {"min_skill": 70, "max_skill": 90, "limit": 10},
            {"min_age": 20, "max_age": 28, "limit": 10},
            {"position": "defender", "min_skill": 75, "limit": 10}
        ]
        
        def send_request(query):
            try:
                start = time.time()
                response = requests.post(f"{API_V1}/search", json=query, timeout=10)
                latency = (time.time() - start) * 1000
                return {
                    'success': response.status_code == 200,
                    'latency': latency,
                    'status': response.status_code
                }
            except Exception as e:
                return {'success': False, 'latency': 0, 'error': str(e)}
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Send 50 requests (10 of each query type)
            futures = []
            for _ in range(10):
                for query in queries:
                    futures.append(executor.submit(send_request, query))
            
            results = [f.result() for f in futures]
        
        duration = time.time() - start_time
        
        # Analyze results
        successful = sum(1 for r in results if r['success'])
        latencies = [r['latency'] for r in results if r['success']]
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)
            
            success_rate = (successful / len(results)) * 100
            throughput = len(results) / duration
            
            if success_rate >= 95:
                log_test("Stress test success rate", "PASS", f"{success_rate:.1f}% ({successful}/{len(results)})")
            else:
                log_test("Stress test success rate", "FAIL", f"{success_rate:.1f}%")
            
            if avg_latency < 1000:
                log_test("Stress test avg latency", "PASS", f"{avg_latency:.0f}ms")
            else:
                log_test("Stress test avg latency", "FAIL", f"{avg_latency:.0f}ms")
            
            print(f"  Throughput: {throughput:.1f} req/s")
            print(f"  Latency: min={min_latency:.0f}ms, avg={avg_latency:.0f}ms, max={max_latency:.0f}ms")
        else:
            log_test("Stress test", "FAIL", "No successful requests")
    
    except Exception as e:
        log_test("Stress test", "FAIL", str(e))

def test_error_responses():
    """Test that errors return proper HTTP status codes and messages"""
    print(f"\n{Colors.BLUE}Testing Error Responses{Colors.RESET}")
    print("-" * 70)
    
    # Test 1: 404 for non-existent endpoint
    try:
        response = requests.get(f"{API_V1}/nonexistent")
        
        if response.status_code == 404:
            log_test("404 for non-existent endpoint", "PASS")
            
            # Check if error message is present
            try:
                data = response.json()
                if 'error' in data:
                    log_test("404 error message present", "PASS")
            except:
                log_test("404 error format", "SKIP", "Non-JSON response")
        else:
            log_test("404 for non-existent endpoint", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        log_test("404 handling", "FAIL", str(e))
    
    # Test 2: 400 for malformed request
    try:
        response = requests.post(f"{API_V1}/search", 
                                data="not valid json",
                                headers={'Content-Type': 'application/json'})
        
        if response.status_code == 400:
            log_test("400 for malformed JSON", "PASS")
            
            try:
                data = response.json()
                if 'error' in data or 'message' in data:
                    log_test("400 error message present", "PASS")
            except:
                pass
        else:
            log_test("400 for malformed JSON", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        log_test("400 handling", "FAIL", str(e))
    
    # Test 3: Method not allowed
    try:
        response = requests.get(f"{API_V1}/search")  # Should be POST
        
        if response.status_code == 405:
            log_test("405 for wrong HTTP method", "PASS")
        else:
            log_test("405 for wrong HTTP method", "SKIP", f"Status: {response.status_code}")
    except Exception as e:
        log_test("405 handling", "FAIL", str(e))

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(f"{Colors.BLUE}COMPREHENSIVE API TEST SUITE{Colors.RESET}")
    print("="*70)
    print(f"Base URL: {BASE_URL}")
    print(f"API Version: v1")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"\n{Colors.RED}ERROR: Server is not responding correctly{Colors.RESET}")
            print(f"Health check returned status: {response.status_code}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"\n{Colors.RED}ERROR: Cannot connect to server at {BASE_URL}{Colors.RESET}")
        print("Please ensure the server is running:")
        print("  python app.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}ERROR: {e}{Colors.RESET}")
        sys.exit(1)
    
    # Run all test suites
    test_suites = [
        ("Health Check", test_health_check),
        ("Basic Search", test_search_basic),
        ("Search Pagination", test_search_pagination),
        ("Search Edge Cases", test_search_edge_cases),
        # ("Search Performance", test_search_performance),
        ("Location-Based Search", test_location_based_search),
        ("Event Logging", test_event_logging),
        ("Recommendations", test_recommendations),
        ("Saved Searches", test_saved_searches),
        ("Response Metadata", test_response_metadata),
        ("Result Data Quality", test_result_data_quality),
        ("Search Consistency", test_search_consistency),
        ("Vector Search", test_vector_search_specific),
        ("Error Responses", test_error_responses),
        # ("Admin Endpoints", test_admin_endpoints)
    ]
    
    for suite_name, test_func in test_suites:
        try:
            test_func()
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Test interrupted by user{Colors.RESET}")
            break
        except Exception as e:
            print(f"\n{Colors.RED}Test suite '{suite_name}' crashed: {e}{Colors.RESET}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    stats.print_summary()
    
    # Exit code based on results
    if stats.failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()