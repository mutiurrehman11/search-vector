import unittest
import json
import os
import sys
import psycopg2
from dotenv import load_dotenv
import ulid

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app, pipeline

class TestApiIntegration(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        pipeline.search_engine.connect()
        pipeline.search_engine.setup_database()
        self.player_id = None

    def tearDown(self):
        if self.player_id:
            with pipeline.search_engine.conn.cursor() as cursor:
                cursor.execute("DELETE FROM players WHERE id = %s", (self.player_id,))
                pipeline.search_engine.conn.commit()
        pipeline.search_engine.disconnect()

    def test_search_endpoint_integration(self):
        # 1. Insert a test player
        player_data = {
            'id': 'test-player-1',
            'name': 'Integration Test',
            'position': 'forward',
            'age': 25,
            'skill_level': 8,
            'latitude': 40.7128,
            'longitude': -74.0060,
            'availability': ['weekday_evening'],
            'tags': ['fast', 'finisher'],
            'bio': 'A test player for integration tests.'
        }
        self.player_id = pipeline.search_engine.index_player(player_data)

        # Search for the player with a more specific query
        response = self.app.post('/api/v1/search', json={
            'position': 'forward',
        })

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertGreater(len(data['results']), 0)

if __name__ == '__main__':
    unittest.main()