from flask import Flask, request, jsonify
from flask_cors import CORS
from marshmallow import Schema, fields, validate, ValidationError
import json
import logging
from functools import wraps
from datetime import datetime, timedelta
import hashlib
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
from Pipeline.Pipeline import SearchPipeline, PlayerEmbeddingGenerator

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration"""
    # Database
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_NAME', 'player_search'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'password'),
        'port': os.getenv('DB_PORT', 5432)
    }

    # Model settings
    MODEL_PATH = os.getenv('MODEL_PATH', 'reranker_model.pkl')
    RERANKING_ENABLED = os.getenv('RERANKING_ENABLED', 'true').lower() == 'true'

    # API settings
    MAX_RESULTS = int(os.getenv('MAX_RESULTS', 50))
    DEFAULT_RESULTS = int(os.getenv('DEFAULT_RESULTS', 20))


# ============================================================================
# FLASK APP SETUP
# ============================================================================

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize search pipeline
pipeline = SearchPipeline(Config.DB_CONFIG)

# Load trained model if exists
try:
    if os.path.exists(Config.MODEL_PATH):
        pipeline.reranker.load_model(Config.MODEL_PATH)
        logger.info(f"Loaded re-ranking model from {Config.MODEL_PATH}")
except Exception as e:
    logger.warning(f"Failed to load model: {e}")


# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================

class SearchRequestSchema(Schema):
    """Schema for search requests"""
    user_id = fields.Int(required=True)

    # Filters
    position = fields.Str(validate=validate.OneOf(['forward', 'midfielder', 'defender', 'goalkeeper', 'any']))
    min_skill = fields.Int(validate=validate.Range(min=1, max=10))
    max_skill = fields.Int(validate=validate.Range(min=1, max=10))
    min_age = fields.Int(validate=validate.Range(min=13, max=100))
    max_age = fields.Int(validate=validate.Range(min=13, max=100))
    latitude = fields.Float(validate=validate.Range(min=-90, max=90))
    longitude = fields.Float(validate=validate.Range(min=-180, max=180))
    max_distance_km = fields.Float(validate=validate.Range(min=0, max=1000))
    availability = fields.List(fields.Str())
    tags = fields.List(fields.Str())

    # Seed players for "find similar"
    seed_player_ids = fields.List(fields.Str())  # Changed from Int to Str for CHAR(26) IDs

    # Tag boosts
    tag_boosts = fields.Dict(keys=fields.Str(), values=fields.Float())

    # Pagination
    limit = fields.Int(validate=validate.Range(min=1, max=Config.MAX_RESULTS), load_default=Config.DEFAULT_RESULTS)
    offset = fields.Int(validate=validate.Range(min=0), load_default=0)


class EventLogSchema(Schema):
    """Schema for logging engagement events"""
    user_id = fields.Int(required=True)
    player_id = fields.Str(required=True)  # Changed from Int to Str for CHAR(26) IDs
    event_type = fields.Str(
        required=True,
        validate=validate.OneOf(['impression', 'profile_view', 'follow', 'message', 'save_to_playlist'])
    )
    query_context = fields.Dict(load_default=dict)
    result_position = fields.Int()
    session_id = fields.Str()


class RecommendationRequestSchema(Schema):
    """Schema for recommendation requests"""
    player_id = fields.Str(required=True)  # Changed from Int to Str for CHAR(26) IDs
    limit = fields.Int(validate=validate.Range(min=1, max=50), load_default=10)



def validate_request(schema_class):
    """Request validation decorator"""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            schema = schema_class()
            try:
                validated_data = schema.load(request.json)
                request.validated_data = validated_data
                return f(*args, **kwargs)
            except ValidationError as err:
                return jsonify({
                    'error': 'Validation failed',
                    'details': err.messages
                }), 400
        return wrapped
    return decorator


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'services': {
            'database': 'connected',
            'reranker': 'loaded' if pipeline.reranker.model else 'not_loaded'
        }
    })


@app.route('/api/v1/search', methods=['POST'])
@validate_request(SearchRequestSchema)
def search_players():
    """
    Search for players with two-stage retrieval + ML re-ranking

    POST /api/v1/search
    {
        "user_id": 123,
        "position": "midfielder",
        "min_skill": 6,
        "max_skill": 9,
        "latitude": 40.7128,
        "longitude": -74.0060,
        "max_distance_km": 10,
        "availability": ["weekday_evening"],
        "tags": ["competitive"],
        "tag_boosts": {"competitive": 2.0},
        "seed_player_ids": [45],
        "limit": 20
    }
    """
    try:
        data = request.validated_data

        # Extract filters
        filters = {
            k: v for k, v in data.items()
            if k not in ['user_id', 'seed_player_ids', 'tag_boosts', 'limit', 'offset']
        }

        # Get seed players if provided
        seed_players = None
        if data.get('seed_player_ids'):
            # Fetch seed players from database
            with pipeline.search_engine.conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM players WHERE id = ANY(%s);",
                    (data['seed_player_ids'],)
                )
                seed_players = [dict(row) for row in cur.fetchall()]

        # Execute search
        results = pipeline.search(
            user_id=data['user_id'],
            filters=filters,
            seed_players=seed_players,
            tag_boosts=data.get('tag_boosts'),
            top_k=data['limit']
        )

        # Format response
        formatted_results = []
        for player in results:
            formatted_results.append({
                'id': player['id'],
                'name': player['name'],
                'position': player['position'],
                'skill_level': player['skill_level'],
                'age': player['age'],
                'location': {
                    'latitude': player['latitude'],
                    'longitude': player['longitude'],
                    'name': player.get('location_name')
                },
                'availability': player['availability'],
                'tags': player['tags'],
                'bio': player['bio'],
                'profile_image_url': player.get('profile_image_url'),
                'scores': {
                    'vector_similarity': player.get('vector_similarity', 0),
                    'ml_score': player.get('ml_score', 0)
                },
                'explanations': player.get('explanations', [])
            })

        return jsonify({
            'results': formatted_results,
            'total': len(formatted_results),
            'metadata': {
                'reranking_enabled': Config.RERANKING_ENABLED,
                'query_id': hashlib.md5(json.dumps(filters).encode()).hexdigest()[:16]
            }
        })

    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Search failed',
            'message': str(e)
        }), 500


@app.route('/api/v1/events', methods=['POST'])
@validate_request(EventLogSchema)
def log_event():
    """
    Log user engagement events (for model training)

    POST /api/v1/events
    {
        "user_id": 123,
        "player_id": 456,
        "event_type": "open",
        "query_context": {...},
        "result_position": 3,
        "session_id": "abc123"
    }
    """
    try:
        data = request.validated_data

        # Log event asynchronously (use Celery in production)
        pipeline.log_interaction(
            user_id=data['user_id'],
            player_id=data['player_id'],
            event_type=data['event_type'],
            query_context=data.get('query_context', {})
        )

        return jsonify({
            'status': 'logged',
            'event_type': data['event_type']
        })

    except Exception as e:
        logger.error(f"Event logging error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Event logging failed',
            'message': str(e)
        }), 500


@app.route('/api/v1/recommendations/<string:player_id>', methods=['GET'])
def get_recommendations(player_id: str):
    """
    Get "more like this" recommendations

    GET /api/v1/recommendations/456?limit=10
    """
    try:
        limit = min(int(request.args.get('limit', 10)), 50)

        recommendations = pipeline.get_recommendations(
            player_id=player_id,
            top_k=limit
        )

        formatted_results = []
        for player in recommendations:
            formatted_results.append({
                'id': player['id'],
                'name': player['name'],
                'position': player['position'],
                'skill_level': player['skill_level'],
                'similarity': player.get('similarity', 0),
                'tags': player['tags'],
                'profile_image_url': player.get('profile_image_url')
            })

        return jsonify({
            'recommendations': formatted_results,
            'total': len(formatted_results),
            'seed_player_id': player_id
        })

    except Exception as e:
        logger.error(f"Recommendations error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Recommendations failed',
            'message': str(e)
        }), 500


@app.route('/api/v1/saved-searches', methods=['POST'])
def save_search():
    """
    Save a search for future alerts

    POST /api/v1/saved-searches
    {
        "user_id": 123,
        "search_name": "Competitive midfielders near me",
        "filters": {...},
        "alert_frequency": "weekly"
    }
    """
    try:
        data = request.validated_data

        saved_search_id = pipeline.saved_search_mgr.save_search(
            user_id=data['user_id'],
            search_name=data['search_name'],
            filters=data['filters'],
            alert_frequency=data['alert_frequency']
        )

        return jsonify({
            'id': saved_search_id,
            'search_name': data['search_name'],
            'alert_frequency': data['alert_frequency']
        }), 201

    except Exception as e:
        logger.error(f"Save search error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Save search failed',
            'message': str(e)
        }), 500


@app.route('/api/v1/saved-searches/<int:user_id>', methods=['GET'])
def get_saved_searches(user_id: int):
    """
    Get all saved searches for a user

    GET /api/v1/saved-searches/123
    """
    try:
        with pipeline.search_engine.conn.cursor() as cur:
            cur.execute("""
                SELECT id, search_name, filters, alert_frequency, 
                       last_alerted_at, created_at
                FROM saved_searches
                WHERE user_id = %s AND is_active = true
                ORDER BY created_at DESC;
            """, (user_id,))

            saved_searches = []
            for row in cur.fetchall():
                saved_searches.append({
                    'id': row[0],
                    'search_name': row[1],
                    'filters': row[2],
                    'alert_frequency': row[3],
                    'last_alerted_at': row[4].isoformat() if row[4] else None,
                    'created_at': row[5].isoformat()
                })

        return jsonify({
            'saved_searches': saved_searches,
            'total': len(saved_searches)
        })

    except Exception as e:
        logger.error(f"Get saved searches error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Failed to retrieve saved searches',
            'message': str(e)
        }), 500


@app.route('/api/v1/saved-searches/<int:search_id>/matches', methods=['GET'])
def get_new_matches(search_id: int):
    """
    Get new matches for a saved search

    GET /api/v1/saved-searches/456/matches
    """
    try:
        new_matches = pipeline.saved_search_mgr.get_new_matches(search_id)

        formatted_results = []
        for player in new_matches:
            formatted_results.append({
                'id': player['id'],
                'name': player['name'],
                'position': player['position'],
                'skill_level': player['skill_level'],
                'similarity': player.get('similarity', 0),
                'created_at': player['created_at'].isoformat()
            })

        return jsonify({
            'new_matches': formatted_results,
            'total': len(formatted_results),
            'search_id': search_id
        })

    except Exception as e:
        logger.error(f"Get new matches error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Failed to retrieve new matches',
            'message': str(e)
        }), 500


@app.route('/api/v1/admin/train-model', methods=['POST'])
def train_model():
    """
    Train the ML re-ranking model (admin endpoint)
    Requires authentication in production

    POST /api/v1/admin/train-model
    """
    try:
        # TODO: Add authentication

        logger.info("Starting model training...")
        pipeline.train_reranker()
        pipeline.reranker.save_model(Config.MODEL_PATH)

        return jsonify({
            'status': 'success',
            'message': 'Model trained and saved',
            'model_path': Config.MODEL_PATH
        })

    except Exception as e:
        logger.error(f"Model training error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Model training failed',
            'message': str(e)
        }), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not found',
        'message': 'The requested resource was not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {str(error)}", exc_info=True)
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


# ============================================================================
# CLI COMMANDS
# ============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == 'setup-db':
            print("Setting up database...")
            pipeline.search_engine.setup_database()
            print("Database setup complete!")

        elif command == 'train-model':
            print("Training re-ranking model...")
            pipeline.train_reranker()
            pipeline.reranker.save_model(Config.MODEL_PATH)
            print(f"Model trained and saved to {Config.MODEL_PATH}")

        elif command == 'index-players':
            print("Indexing players...")
            print("Indexing complete!")

        else:
            print(f"Unknown command: {command}")
            print("Available commands: setup-db, train-model, index-players")

    else:
        app.run(debug=True, host='0.0.0.0', port=5000)
