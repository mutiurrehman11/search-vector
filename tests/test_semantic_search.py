import os
import sys
import unittest
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, pipeline

class SemanticSearchTestCase(unittest.TestCase):

    def setUp(self):
        """Set up a test client for the Flask application."""
        self.app = app.test_client()
        self.app.testing = True

    @patch('app.pipeline.search_players')
    def test_semantic_search_ordering(self, mock_search_players):
        """Test that semantic search results are ordered by similarity score."""
        mock_search_players.return_value = {
            'results': [
                {'id': '1', 'name': 'Player A', 'similarity_score': 0.9},
                {'id': '2', 'name': 'Player B', 'similarity_score': 0.8}
            ],
            'total_count': 2,
            'metadata': {}
        }

        response = self.app.post('/api/v1/search', json={'position': 'defender'})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(len(data['results']), 2)
        self.assertEqual(data['results'][0]['similarity_score'], 0.9)
        self.assertEqual(data['results'][1]['similarity_score'], 0.8)


if __name__ == '__main__':
    unittest.main()