import os
import sys
import unittest
import json
from unittest.mock import patch, MagicMock

# Add the parent directory to the sys.path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, pipeline
from datetime import datetime

from Pipeline.SearchPipeline import SearchPipeline

class FlaskAppTestCase(unittest.TestCase):

    def setUp(self):
        """Set up a test client for the Flask application."""
        self.app = app.test_client()
        self.app.testing = True
        # Connect to the database and set up the schema for integration tests
        pipeline.search_engine.connect()
        pipeline.search_engine.setup_database()

    def tearDown(self):
        """Clean up after each test."""
        # Disconnect from the database
        pipeline.search_engine.disconnect()

    def test_health_check(self):
        """Test the health check endpoint."""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data['status'], 'healthy')
        self.assertEqual(data['services']['database'], 'connected')

    @patch('app.pipeline.search_players')
    def test_search_valid_query(self, mock_search_players):
        """Test search with a valid query returns 200."""
        mock_search_players.return_value = {
            'results': [{'id': '1', 'name': 'Test Player'}],
            'total_count': 1,
            'metadata': {}
        }
        response = self.app.post('/api/v1/search', json={'position': 'defender'})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('results', data)
        self.assertEqual(len(data['results']), 1)

    def test_search_invalid_query(self):
        """Test the search endpoint with an invalid query."""
        response = self.app.post('/api/v1/search', json={'min_skill': 11})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'Validation failed')

    @patch('app.pipeline.search_players')
    def test_search_no_results(self, mock_search_players):
        """Test the search endpoint with a query that returns no results."""
        mock_search_players.return_value = {
            'results': [],
            'total_count': 0,
            'metadata': {}
        }

        response = self.app.post('/api/v1/search', json={'position': 'goalkeeper'})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('results', data)
        self.assertEqual(len(data['results']), 0)

    @patch('app.pipeline.search_players')
    def test_search_with_filter(self, mock_search_players):
        """Test search with a filter returns 200."""
        mock_search_players.return_value = {
            'results': [{'id': '2', 'name': 'Filtered Player'}],
            'total_count': 1,
            'metadata': {}
        }
        response = self.app.post('/api/v1/search', json={'position': 'midfielder', 'min_skill': 5})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('results', data)
        self.assertEqual(len(data['results']), 1)

    @patch('app.pipeline.search_players')
    def test_search_with_multiple_filters(self, mock_search_players):
        """Test search with multiple filters returns 200."""
        mock_search_players.return_value = {
            'results': [{'id': '3', 'name': 'Multi-Filtered Player'}],
            'total_count': 1,
            'metadata': {}
        }
        response = self.app.post('/api/v1/search', json={'position': 'forward', 'min_skill': 7, 'max_age': 25})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('results', data)
        self.assertEqual(len(data['results']), 1)

    @patch('app.pipeline.search_players')
    def test_search_with_pagination(self, mock_search_players):
        """Test search with pagination returns 200."""
        mock_search_players.return_value = {
            'results': [{'id': '4', 'name': 'Paginated Player'}],
            'total_count': 1,
            'metadata': {}
        }
        response = self.app.post('/api/v1/search', json={'position': 'defender', 'limit': 1, 'offset': 1})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('results', data)
        self.assertEqual(len(data['results']), 1)

    @patch('app.pipeline.search_players')
    def test_search_with_tag_boosts(self, mock_search_players):
        """Test search with tag boosts returns 200."""
        mock_search_players.return_value = {
            'results': [{'id': '5', 'name': 'Boosted Player'}],
            'total_count': 1,
            'metadata': {}
        }
        response = self.app.post('/api/v1/search', json={'position': 'midfielder', 'tag_boosts': {'fast': 2.0}})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('results', data)
        self.assertEqual(len(data['results']), 1)

    @patch('app.pipeline.search_players')
    def test_search_with_seed_players(self, mock_search_players):
        """Test search with seed players returns 200."""
        mock_search_players.return_value = {
            'results': [{'id': '6', 'name': 'Similar Player'}],
            'total_count': 1,
            'metadata': {}
        }
        response = self.app.post('/api/v1/search', json={'seed_player_ids': ['1', '2']})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('results', data)
        self.assertEqual(len(data['results']), 1)

    @patch('app.pipeline.search_players')
    def test_search_with_geo_filter(self, mock_search_players):
        """Test search with a geographic filter returns 200."""
        mock_search_players.return_value = {
            'results': [{'id': '7', 'name': 'Nearby Player'}],
            'total_count': 1,
            'metadata': {}
        }
        response = self.app.post('/api/v1/search', json={
            'latitude': 40.7128,
            'longitude': -74.0060,
            'max_distance_km': 10
        })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('results', data)
        self.assertEqual(len(data['results']), 1)

    @patch('app.pipeline.search_players')
    def test_recommendations_valid_player_id(self, mock_search_players):
        """Test the recommendations endpoint with a valid player ID."""
        mock_search_players.return_value = {
            'results': [
                {'id': 2, 'first_name': 'Player', 'last_name': '2', 'positions': ['Forward'], 'skills': 'Shooting, Passing'},
                {'id': 3, 'first_name': 'Player', 'last_name': '3', 'positions': ['Forward'], 'skills': 'Dribbling, Passing'}
            ],
            'total_count': 2,
            'metadata': {}
        }

        response = self.app.get('/api/v1/recommendations/1')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('recommendations', data)
        self.assertEqual(len(data['recommendations']), 2)
        self.assertEqual(data['recommendations'][0]['name'], 'Player 2')

    @patch('app.pipeline.search_players')
    def test_recommendations_invalid_player_id(self, mock_search_players):
        """Test the recommendations endpoint with an invalid player ID."""
        mock_search_players.return_value = {
            'results': [],
            'total_count': 0,
            'metadata': {}
        }

        response = self.app.get('/api/v1/recommendations/999')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('recommendations', data)
        self.assertEqual(len(data['recommendations']), 0)

    @patch('app.pipeline.log_interaction')
    def test_events_valid_event(self, mock_log_interaction):
        """Test the events endpoint with a valid event."""
        response = self.app.post('/api/v1/events', json={
            'user_id': 1,
            'player_id': '1',
            'event_type': 'profile_view',
        })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data['status'], 'logged')

    @patch('app.pipeline.log_interaction')
    def test_events_complex_event(self, mock_log_interaction):
        """Test the events endpoint with a more complex event."""
        response = self.app.post('/api/v1/events', json={
            'user_id': 2,
            'player_id': '3',
            'event_type': 'impression',
            'query_context': {'search_query': 'fast winger', 'page': 2}
        })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data['status'], 'logged')

    def test_events_invalid_event(self):
        """Test the events endpoint with an invalid event."""
        response = self.app.post('/api/v1/events', json={
            'player_id': 1,
            'event_type': 'invalid_event_type',
            'context': {'page': 'player_profile', 'player_id': 2}
        })
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'Validation failed')

    @patch('app.pipeline.search_players')
    def test_database_connection_error(self, mock_search_players):
        """Test a database connection error."""
        mock_search_players.side_effect = Exception('Database connection error')

        response = self.app.post('/api/v1/search', json={'position': 'defender'})
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'Search failed')

    def test_pgvector_usage(self):
        """Test that pgvector is being used."""
        self.assertTrue(pipeline.search_engine.pgvector_available)

    def test_ml_model_usage(self):
        """Test that the ML model is being used."""
        self.assertIsNotNone(pipeline.ml_reranker.model)

    @patch('app.pipeline.search_players')
    def test_search_database_error(self, mock_search_players):
        """Test a database error during a search query."""
        mock_search_players.side_effect = Exception('Database query error')

        response = self.app.post('/api/v1/search', json={'position': 'defender'})
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'Search failed')

    @patch('app.pipeline.get_personalized_recommendations')
    def test_personalized_recommendations_valid(self, mock_get_personalized_recommendations):
        """Test personalized recommendations with a valid user ID."""
        mock_get_personalized_recommendations.return_value = {
            'results': [{'id': '10', 'name': 'Personalized Player'}],
            'total_count': 1,
            'metadata': {}
        }
        response = self.app.get('/api/v1/recommendations/personalized/1')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('recommendations', data)
        self.assertEqual(len(data['recommendations']), 1)

    @patch('app.pipeline.get_personalized_recommendations')
    def test_personalized_recommendations_invalid(self, mock_get_personalized_recommendations):
        """Test personalized recommendations with an invalid user ID."""
        mock_get_personalized_recommendations.side_effect = Exception('User not found')
        response = self.app.get('/api/v1/recommendations/personalized/999')
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('error', data)

    @patch('app.pipeline.saved_search_mgr.save_search')
    def test_save_search_valid(self, mock_save_search):
        """Test saving a search with valid data."""
        mock_save_search.return_value = 1
        response = self.app.post('/api/v1/saved-searches', json={
            'user_id': 1,
            'search_name': 'Test Search',
            'filters': {'position': 'defender'},
            'alert_frequency': 'daily'
        })
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data['id'], 1)

    @patch.object(pipeline.saved_search_mgr, 'get_new_matches')
    def test_get_new_matches_valid(self, mock_get_new_matches):
        mock_get_new_matches.return_value = {'matches': [{'id': 1, 'name': 'Test Player'}], 'total': 1}
        response = self.app.get('/api/v1/saved-searches/1/new-matches')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(len(data['matches']), 1)
        mock_get_new_matches.assert_called_once_with(1)

    @patch('app.pipeline.search_engine.conn')
    def test_get_saved_searches_valid(self, mock_conn):
        mock_cursor = mock_conn.cursor.return_value.__enter__.return_value
        mock_cursor.fetchall.return_value = [
            (1, 'Test Search', '{}', 'daily', datetime.now(), datetime.now())
        ]
        response = self.app.get('/api/v1/saved-searches/1')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(len(data['saved_searches']), 1)
        mock_conn.cursor.assert_called_once()

    @patch('app.pipeline.train_ml_model')
    def test_train_model_endpoint(self, mock_train_ml_model):
        """Test the model training endpoint."""
        mock_train_ml_model.return_value = {'success': True, 'message': 'Training started', 'metadata': {}}
        response = self.app.post('/api/v1/admin/train-model')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data['message'], 'Training started')

    def test_search_endpoint_integration(self):
        """Test the search endpoint with a real query after inserting data."""
        # Insert a test player
        player_data = {
            'id': 'test-player-1',
            'name': 'Integration Test',
            'position': 'forward',
            'age': 25,
            'skill_level': 8,
            'latitude': 40.7128,
            'longitude': -74.0060,
            'tenure': 5,
            'shots_total': 150,
            'goals_total': 30,
            'passes_total': 1200,
            'interceptions_total': 200,
            'availability': ['weekday_evening'],
            'tags': ['fast', 'finisher'],
            'bio': 'A test player for integration tests.',
            'embedding': [0.1] * 128  # Example embedding
        }
        with app.app_context():
            pipeline.search_engine.index_player(player_data)

        # Search for the player
        response = self.app.post('/api/v1/search', json={'position': 'forward'})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertGreater(len(data['results']), 0)

        # Clean up the inserted player
        with pipeline.search_engine.conn.cursor() as cursor:
            cursor.execute("DELETE FROM players WHERE id = %s", (player_data['id'],))
            pipeline.search_engine.conn.commit()

if __name__ == '__main__':
    unittest.main()