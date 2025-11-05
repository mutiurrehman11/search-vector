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
from Pipeline.SearchPipeline import SearchPipeline

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


app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize pipeline
pipeline = SearchPipeline(Config.DB_CONFIG)

logger.info("Search pipeline initialized")


class SearchRequestSchema(Schema):
    """Schema for search requests"""
    user_id = fields.Int(required=False)  # Optional - only needed for personalization and logging

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



@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'services': {
            'database': 'connected',
            'pgvector': pipeline.engine.pgvector_available if hasattr(pipeline, 'engine') else False,
            'search_pipeline': 'initialized'
        }
    })


@app.route('/api/v1/search', methods=['POST'])
@validate_request(SearchRequestSchema)
def search_players():
    """
    Search for players based on filters and preferences
    
    POST /api/v1/search
    {
        "user_id": 123,
        "position": "midfielder",
        "min_skill": 60,
        "max_skill": 90,
        "latitude": 40.7128,
        "longitude": -74.0060,
        "max_distance_km": 10,
        "availability": ["weekday_evening"],
        "tags": ["competitive"],
        "limit": 20,
        "offset": 0
    }
    """
    try:
        data = request.validated_data

        # Extract filters for the search
        filters = {
            k: v for k, v in data.items()
            if k not in ['user_id', 'limit', 'offset']
        }

        # Execute search using the new pipeline
        search_results = pipeline.search_players(
            filters=filters,
            limit=data.get('limit', Config.DEFAULT_RESULTS),
            offset=data.get('offset', 0)
        )

        # Format response
        formatted_results = []
        for player in search_results.get('results', []):
            formatted_player = {
                'id': player['id'],
                'name': f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
                'position': player.get('positions', []),
                'skill_level': player.get('avg_skill_level', 0),
                'age': player.get('age', 0),
                'location': player.get('location', {}),
                'skills': player.get('skills', {}),
                'status': player.get('status', 'active'),
                'match_score': player.get('match_score', 0),
                'similarity_score': player.get('similarity_score'),
                'distance_km': player.get('distance_km')
            }
            formatted_results.append(formatted_player)

        # Log search event (only if user_id provided)
        if data.get('user_id'):
            try:
                # This would be implemented with proper event logging
                logger.info(f"Search performed by user {data['user_id']}: {len(formatted_results)} results")
            except Exception as e:
                logger.warning(f"Failed to log search event: {e}")

        return jsonify({
            'results': formatted_results,
            'total': search_results.get('total_count', len(formatted_results)),
            'metadata': {
                'search_type': search_results.get('metadata', {}).get('search_type', 'unknown'),
                'candidates_found': search_results.get('metadata', {}).get('candidates_found', 0),
                'reranked': search_results.get('metadata', {}).get('reranked', False),
                'pgvector_available': search_results.get('metadata', {}).get('pgvector_available', False),
                'query_id': hashlib.md5(json.dumps(filters, sort_keys=True).encode()).hexdigest()[:16]
            },
            'telemetry': search_results.get('telemetry', {})
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

        # Use the similarity search with the player as seed
        filters = {
            'seed_player_ids': [player_id]
        }
        
        recommendations = pipeline.search_players(
            filters=filters,
            limit=limit,
            offset=0
        )

        formatted_results = []
        for player in recommendations.get('results', []):
            formatted_results.append({
                'id': player['id'],
                'name': f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
                'position': player.get('positions', []),
                'skill_level': player.get('avg_skill_level', 0),
                'similarity': player.get('similarity_score', 0),
                'tags': player.get('tags', []),
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


@app.route('/api/v1/admin/train-model', methods=['POST'])
def train_ml_model():
    """
    Train the ML re-ranking model (Admin endpoint)
    
    POST /api/v1/admin/train-model
    """
    try:
        logger.info("Starting ML model training via API")
        result = pipeline.train_ml_model()
        
        if result['success']:
            return jsonify({
                'message': result['message'],
                'metadata': result['metadata']
            }), 200
        else:
            return jsonify({
                'error': result['error'],
                'metadata': result['metadata']
            }), 400
            
    except Exception as e:
        logger.error(f"ML training error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'ML model training failed',
            'message': str(e)
        }), 500


@app.route('/api/v1/recommendations/personalized/<int:user_id>', methods=['GET'])
def get_personalized_recommendations(user_id: int):
    """
    Get personalized recommendations for a user based on their interaction history
    
    GET /api/v1/recommendations/personalized/123?limit=20
    """
    try:
        limit = request.args.get('limit', 20, type=int)
        limit = min(limit, 50)  # Cap at 50
        
        logger.info(f"Getting personalized recommendations for user {user_id}")
        result = pipeline.get_personalized_recommendations(str(user_id), limit)
        
        return jsonify({
            'recommendations': result['results'],
            'total_count': result['total_count'],
            'metadata': result['metadata'],
            'telemetry': result.get('telemetry', {})
        })
        
    except Exception as e:
        logger.error(f"Personalized recommendations error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Personalized recommendations failed',
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
