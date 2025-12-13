from flask import Flask, request, jsonify
from flask_cors import CORS
from marshmallow import Schema, fields, validate, ValidationError
import json
import logging
from functools import wraps
from datetime import datetime
import hashlib
import os
import sys
import pytz
from dotenv import load_dotenv
from Pipeline.SearchPipeline import SearchPipeline

load_dotenv()


class Config:
    """Application configuration"""
    # Database
    DB_CONFIG = {
        'host': '167.86.115.58',
        'port': 5432,
        'dbname': 'prospects_dev',
        'user': 'devuser',
        'password': 'testdev123'
    }

    DB_URI = os.getenv('DATABASE_URL')

    # Model settings
    MODEL_PATH = os.getenv('MODEL_PATH', 'reranker_model.pkl')
    POST_MODEL_PATH = os.getenv('POST_MODEL_PATH', 'models/post_recommender.pkl')
    RERANKING_ENABLED = os.getenv('RERANKING_ENABLED', 'true').lower() == 'true'

    # API settings
    MAX_RESULTS = int(os.getenv('MAX_RESULTS', 50))
    DEFAULT_RESULTS = int(os.getenv('DEFAULT_RESULTS', 20))


app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Setup logging
log_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)

# File handler
file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

sys.stdout.reconfigure(line_buffering=True)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

# Get Flask's root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logging.getLogger('werkzeug').addHandler(console_handler)
logging.getLogger('werkzeug').addHandler(file_handler)

pipeline = SearchPipeline(Config.DB_CONFIG)

logger.info("Search pipeline initialized")


class SearchRequestSchema(Schema):
    """Schema for search requests"""
    user_id = fields.Int(required=False)
    role = fields.Str()
    position = fields.Str(validate=validate.OneOf(['forward', 'midfielder', 'defender', 'goalkeeper', 'any']))
    min_skill = fields.Int(validate=validate.Range(min=1, max=100))
    max_skill = fields.Int(validate=validate.Range(min=1, max=100))
    min_age = fields.Int(validate=validate.Range(min=13, max=100))
    max_age = fields.Int(validate=validate.Range(min=13, max=100))
    latitude = fields.Float(validate=validate.Range(min=-90, max=90))
    longitude = fields.Float(validate=validate.Range(min=-180, max=180))
    max_distance_km = fields.Float(validate=validate.Range(min=0, max=1000))
    availability = fields.List(fields.Str())
    tags = fields.List(fields.Str())
    seed_player_ids = fields.List(fields.Str())
    tag_boosts = fields.Dict(keys=fields.Str(), values=fields.Float())
    limit = fields.Int(validate=validate.Range(min=1, max=Config.MAX_RESULTS), load_default=Config.DEFAULT_RESULTS)
    offset = fields.Int(validate=validate.Range(min=0), load_default=0)


class EventLogSchema(Schema):
    """Schema for logging engagement events"""
    user_id = fields.Str(required=True)
    player_id = fields.Str(required=True)
    event_type = fields.Str(
        required=True,
        validate=validate.OneOf(['impression', 'profile_view', 'follow', 'message', 'save_to_playlist'])
    )
    query_context = fields.Dict(load_default=dict)


class PostInteractionSchema(Schema):
    """Schema for post interaction events"""
    user_id = fields.Str(required=True)
    post_id = fields.Str(required=True)
    interaction_type = fields.Str(
        required=True,
        validate=validate.OneOf(['view', 'like', 'comment', 'share', 'save'])
    )
    interaction_metadata = fields.Dict(load_default=dict)
    dwell_time_seconds = fields.Float(load_default=0.0)


class RecommendationRequestSchema(Schema):
    """Schema for recommendation requests"""
    player_id = fields.Str(required=True)
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
        'timestamp': datetime.now(pytz.UTC).isoformat(),
        'services': {
            'database': 'connected',
            'pgvector': pipeline.search_engine.pgvector_available,
            'search_pipeline': 'initialized'
        }
    })


@app.route('/api/v1/search', methods=['POST'])
@validate_request(SearchRequestSchema)
def search_players():
    """Search for players based on filters and preferences"""
    try:
        data = request.validated_data
        logger.info(f"Search request received: {data}")

        filters = {
            k: v for k, v in data.items()
            if k not in ['user_id', 'limit', 'offset']
        }

        if 'min_age' in filters:
            filters['min_age'] = int(filters['min_age'])
        if 'max_age' in filters:
            filters['max_age'] = int(filters['max_age'])

        search_results = pipeline.search_players(
            filters=filters,
            limit=data.get('limit', Config.DEFAULT_RESULTS),
            offset=data.get('offset', 0)
        )
        logger.info(f"Search results from pipeline: {search_results}")

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
                'rerank_score': player.get('rerank_score'),
                'distance_km': player.get('distance_km')
            }
            formatted_results.append(formatted_player)

        if data.get('user_id'):
            try:
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
    """Log user engagement events (for model training)"""
    try:
        data = request.validated_data

        print(f"Logging event: {data}")
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


# ===== NEW POST INTERACTION ROUTES =====

@app.route('/api/v1/posts/interactions', methods=['POST'])
@validate_request(PostInteractionSchema)
def log_post_interaction():
    """
    Log user interactions with posts (likes, comments, shares, etc.)

    POST /api/v1/posts/interactions
    {
        "user_id": "01HXA7B2C3D4E5F6G7H8J9K0M1",
        "post_id": "01HXA7B2C3D4E5F6G7H8J9K0M2",
        "interaction_type": "like",
        "interaction_metadata": {"source": "feed"},
        "dwell_time_seconds": 5.2
    }
    """
    try:
        data = request.validated_data
        logger.info(f"Post interaction logged: {data}")

        # Log the interaction to database
        with pipeline.search_engine.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO post_interactions 
                (user_id, post_id, interaction_type, interaction_metadata, dwell_time_seconds, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                RETURNING id;
            """, (
                data['user_id'],
                data['post_id'],
                data['interaction_type'],
                json.dumps(data.get('interaction_metadata', {})),
                data.get('dwell_time_seconds', 0.0)
            ))

            interaction_id = cur.fetchone()[0]
            pipeline.search_engine.conn.commit()

        return jsonify({
            'status': 'logged',
            'interaction_id': interaction_id,
            'interaction_type': data['interaction_type'],
            'timestamp': datetime.now(pytz.UTC).isoformat()
        }), 201

    except Exception as e:
        logger.error(f"Post interaction logging error: {str(e)}", exc_info=True)
        pipeline.search_engine.conn.rollback()
        return jsonify({
            'error': 'Failed to log post interaction',
            'message': str(e)
        }), 500


@app.route('/api/v1/posts/feed/<string:user_id>', methods=['GET'])
def get_personalized_feed(user_id: str):
    """
    Get personalized post feed for a user based on their interactions

    GET /api/v1/posts/feed/01HXA7B2C3D4E5F6G7H8J9K0M1?limit=20&offset=0
    """
    try:
        # Validate user_id format (ULID is 26 characters)
        if not user_id or len(user_id) != 26:
            return jsonify({
                'posts': [],
                'error': 'Invalid user ID format',
                'message': 'User ID must be a 26-character ULID'
            }), 400

        # Get pagination parameters
        try:
            limit = min(int(request.args.get('limit', 20)), 100)
            offset = int(request.args.get('offset', 0))
        except ValueError:
            return jsonify({
                'posts': [],
                'error': 'Invalid pagination parameters',
                'message': 'Limit and offset must be integers'
            }), 400

        logger.info(f"Fetching personalized feed for user {user_id} (limit={limit}, offset={offset})")

        # Get personalized post recommendations
        from Pipeline.PostRecommender import PostRecommendationEngine
        post_recommend = PostRecommendationEngine(pipeline.search_engine.conn, Config.POST_MODEL_PATH)

        recommended_posts = post_recommend.get_personalized_feed(
            user_id=user_id,
            limit=limit,
            offset=offset
        )

        return jsonify({
            'posts': recommended_posts.get('posts', []),
            'total': recommended_posts.get('total_count', 0),
            'user_id': user_id,
            'metadata': {
                'personalized': recommended_posts.get('personalized', False),
                'recommendation_strategy': recommended_posts.get('strategy', 'fallback'),
                'cached': False
            },
            'pagination': {
                'limit': limit,
                'offset': offset,
                'has_more': len(recommended_posts.get('posts', [])) == limit
            }
        })

    except Exception as e:
        logger.error(f"Failed to get personalized feed: {str(e)}", exc_info=True)
        return jsonify({
            'posts': [],
            'error': 'Failed to retrieve personalized feed',
            'message': str(e)
        }), 500


@app.route('/api/v1/posts/trending', methods=['GET'])
def get_trending_posts():
    """
    Get trending posts based on recent engagement

    GET /api/v1/posts/trending?limit=20&timeframe=24h
    """
    try:
        limit = min(int(request.args.get('limit', 20)), 100)
        timeframe = request.args.get('timeframe', '24h')

        # Parse timeframe
        hours_map = {'1h': 1, '6h': 6, '12h': 12, '24h': 24, '48h': 48, '7d': 168}
        hours = hours_map.get(timeframe, 24)

        logger.info(f"Fetching trending posts (limit={limit}, timeframe={timeframe})")

        with pipeline.search_engine.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    p.id,
                    p.player_id,
                    p.content,
                    p.media_urls,
                    p.created_at,
                    CONCAT(pl.first_name, ' ', pl.last_name) as author_name,
                    pl.profile_picture as author_avatar,
                    COUNT(DISTINCT CASE WHEN pi.interaction_type = 'like' THEN pi.id END) as like_count,
                    COUNT(DISTINCT CASE WHEN pi.interaction_type = 'comment' THEN pi.id END) as comment_count,
                    COUNT(DISTINCT CASE WHEN pi.interaction_type = 'share' THEN pi.id END) as share_count,
                    COUNT(DISTINCT pi.user_id) as unique_engagement_count,
                    -- Engagement score: weighted sum with recency boost
                    (
                        COUNT(DISTINCT CASE WHEN pi.interaction_type = 'like' THEN pi.id END) * 1 +
                        COUNT(DISTINCT CASE WHEN pi.interaction_type = 'comment' THEN pi.id END) * 3 +
                        COUNT(DISTINCT CASE WHEN pi.interaction_type = 'share' THEN pi.id END) * 5
                    ) * EXP(-EXTRACT(EPOCH FROM (NOW() - p.created_at)) / 86400.0) as engagement_score
                FROM posts p
                LEFT JOIN players pl ON p.player_id = pl.id
                LEFT JOIN post_interactions pi ON p.id = pi.post_id 
                    AND pi.created_at > NOW() - INTERVAL '%s hours'
                WHERE p.created_at > NOW() - INTERVAL '%s hours'
                    AND p.deleted_at IS NULL
                GROUP BY p.id, pl.first_name, pl.last_name, pl.profile_picture
                ORDER BY engagement_score DESC
                LIMIT %s;
            """, (hours, hours, limit))

            trending_posts = cur.fetchall()

        formatted_posts = []
        for post in trending_posts:
            formatted_posts.append({
                'id': post[0],
                'player_id': post[1],
                'content': post[2],
                'media_urls': post[3] if post[3] else [],
                'created_at': post[4].isoformat(),
                'author': {
                    'name': post[5],
                    'avatar': post[6]
                },
                'engagement': {
                    'likes': post[7],
                    'comments': post[8],
                    'shares': post[9],
                    'unique_users': post[10]
                },
                'engagement_score': float(post[11])
            })

        return jsonify({
            'posts': formatted_posts,
            'total': len(formatted_posts),
            'metadata': {
                'timeframe': timeframe,
                'hours': hours,
                'type': 'trending'
            }
        })

    except Exception as e:
        logger.error(f"Failed to get trending posts: {str(e)}", exc_info=True)
        return jsonify({
            'posts': [],
            'error': 'Failed to retrieve trending posts',
            'message': str(e)
        }), 500


@app.route('/api/v1/admin/train-post-model', methods=['POST'])
def train_post_recommendation_model():
    """
    Train the post recommendation model based on user interactions

    POST /api/v1/admin/train-post-model
    """
    try:
        logger.info("Starting post recommendation model training via API")

        from Pipeline.PostRecommender import PostRecommendationEngine
        post_recommender = PostRecommendationEngine(pipeline.search_engine.conn, Config.POST_MODEL_PATH)

        result = post_recommender.train_model()

        if result['success']:
            return jsonify({
                'message': result['message'],
                'metadata': result['metadata']
            }), 200
        else:
            return jsonify({
                'error': result.get('error', 'Training failed'),
                'metadata': result.get('metadata', {})
            }), 400

    except Exception as e:
        logger.error(f"Post model training error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Post model training failed',
            'message': str(e)
        }), 500


# ===== EXISTING ROUTES (kept as-is) =====

@app.route('/api/v1/recommendations/<string:player_id>', methods=['GET'])
def get_recommendations(player_id: str):
    """Get "more like this" recommendations"""
    try:
        limit = min(int(request.args.get('limit', 10)), 50)

        if not player_id or len(player_id) != 26:
            return jsonify({
                'recommendations': [],
                'error': 'Invalid player ID format',
                'message': 'Player ID must be a 26-character ULID'
            }), 400

        try:
            limit = min(int(request.args.get('limit', 10)), 50)
        except ValueError:
            return jsonify({
                'recommendations': [],
                'error': 'Invalid limit parameter',
                'message': 'Limit must be an integer'
            }), 400

        with pipeline.search_engine.conn.cursor() as cur:
            cur.execute("""
                SELECT id FROM players 
                WHERE id = %s 
                AND status = 'active' 
                AND deleted_at IS NULL
            """, (player_id,))

            if not cur.fetchone():
                return jsonify({
                    'recommendations': [],
                    'error': 'Player not found',
                    'message': f'No active player found with ID: {player_id}',
                    'player_id': player_id
                }), 404

        filters = {'seed_player_ids': [player_id]}
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
            'recommendations': [],
            'error': 'Recommendations failed',
            'message': str(e)
        }), 500


@app.route('/api/v1/admin/train-model', methods=['POST'])
def train_ml_model():
    """Train the ML re-ranking model (Admin endpoint)"""
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


@app.route('/api/v1/saved-searches', methods=['POST'])
def save_search():
    """Save a search for future alerts"""
    try:
        data = request.json
        if data["alert_frequency"] not in ["daily", "weekly"]:
            return jsonify({
                'error': 'Invalid alert frequency',
                'message': 'Alert frequency must be "daily" or "weekly"'
            }), 400

        saved_search_id = pipeline.saved_search_mgr.save_search(
            user_id=str(data['user_id']),
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


@app.route('/api/v1/saved-searches/<user_id>', methods=['GET'])
def get_saved_searches(user_id):
    """Get all saved searches for a user"""
    try:
        logger.info(f"Fetching saved searches for user_id: {user_id}")
        saved_searches = pipeline.saved_search_mgr.get_saved_searches(str(user_id))

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


@app.route('/api/v1/admin/generate-embeddings', methods=['POST'])
def generate_embeddings():
    """Generate and store embeddings for players missing them (Admin endpoint)"""
    try:
        logger.info("Starting embedding generation via API")
        result = pipeline.generate_and_store_embeddings()

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Embedding generation error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Embedding generation failed',
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == 'setup-db':
            print("Setting up database...")
            pipeline.search_engine.setup_database()
            print("Database setup complete!")

        elif command == 'train-model':
            print("Training re-ranking model...")
            pipeline.train_ml_model()
            pipeline.ml_reranker.save_model(Config.MODEL_PATH)
            print(f"Model trained and saved to {Config.MODEL_PATH}")

        elif command == 'train-post-model':
            print("Training post recommendation model...")
            from Pipeline.PostRecommender import PostRecommendationEngine

            post_recommender = PostRecommendationEngine(pipeline.search_engine.conn, Config.POST_MODEL_PATH)
            result = post_recommender.train_model()
            print(f"Post model training result: {result}")

        elif command == 'index-players':
            print("Indexing players...")
            print("Indexing complete!")

        else:
            print(f"Unknown command: {command}")
            print("Available commands: setup-db, train-model, train-post-model, index-players")

    else:
        app.run(debug=True, host='0.0.0.0', port=5000, reload=True)