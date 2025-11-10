"""
Corrected Pipeline - Fixed to work with embedding column in players table
"""

import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from pgvector.psycopg2 import register_vector
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
import logging
import pickle
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class PlayerEmbeddingGenerator:
    """Generate embeddings from structured player attributes"""

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim

    def generate_embedding(self, player: Dict) -> np.ndarray:
        """Generate complete embedding vector for a player"""
        # Simple embedding generation - can be replaced with more sophisticated approach
        components = []
        
        # Position encoding (32 dims)
        positions = player.get('positions', [])
        pos_encoding = np.zeros(32)
        position_map = {'forward': 0, 'midfielder': 8, 'defender': 16, 'goalkeeper': 24}
        for pos in positions:
            if pos in position_map:
                pos_encoding[position_map[pos]] = 1.0
        components.append(pos_encoding)
        
        # Skill level encoding (32 dims)
        skills = player.get('skills', {})
        skill_encoding = np.zeros(32)
        if skills:
            skill_values = list(skills.values())
            avg_skill = np.mean(skill_values) / 100.0  # Normalize to 0-1
            skill_encoding[0] = avg_skill
        components.append(skill_encoding)
        
        # Age encoding (16 dims)
        age = player.get('age', 25)
        age_encoding = np.zeros(16)
        age_encoding[0] = (age - 16) / 24.0  # Normalize 16-40 to 0-1
        components.append(age_encoding)
        
        # Location encoding (24 dims)
        location = player.get('location', {})
        loc_encoding = np.zeros(24)
        if 'latitude' in location:
            loc_encoding[0] = (location['latitude'] + 90) / 180.0
        if 'longitude' in location:
            loc_encoding[1] = (location['longitude'] + 180) / 360.0
        components.append(loc_encoding)
        
        # Physical attributes (24 dims)
        phys_encoding = np.zeros(24)
        if player.get('height'):
            phys_encoding[0] = (player['height'] - 150) / 50.0
        if player.get('weight'):
            phys_encoding[1] = (player['weight'] - 50) / 50.0
        components.append(phys_encoding)
        
        # Concatenate and normalize
        embedding = np.concatenate(components)[:self.embedding_dim].astype(np.float32)
        
        # Pad if necessary
        if len(embedding) < self.embedding_dim:
            padding = np.zeros(self.embedding_dim - len(embedding), dtype=np.float32)
            embedding = np.concatenate([embedding, padding])
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm
        
        return embedding


class QueryVectorBuilder:
    """Build query vectors from filters"""

    def __init__(self, embedding_generator: PlayerEmbeddingGenerator):
        self.embedding_gen = embedding_generator

    def build_from_filters(self, filters: Dict) -> np.ndarray:
        """Build query vector from filters"""
        # Convert filters to player-like dict for embedding generation
        query_player = {
            'positions': [filters.get('position', 'midfielder')],
            'skills': {},
            'age': (filters.get('min_age', 20) + filters.get('max_age', 30)) // 2,
            'location': {
                'latitude': filters.get('latitude', 0),
                'longitude': filters.get('longitude', 0)
            }
        }
        
        return self.embedding_gen.generate_embedding(query_player)


class PlayerSearchEngine:
    """Main search engine with pgvector integration"""

    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = psycopg2.connect(**self.db_config)
        self.embedding_gen = PlayerEmbeddingGenerator()
        self.query_builder = QueryVectorBuilder(self.embedding_gen)
        self.pgvector_available = False

    def connect(self):
        """Connect to PostgreSQL with pgvector"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            
            self.conn.autocommit = False
            
            try:
                register_vector(self.conn)
            except Exception as e:
                logger.warning(f"Could not register pgvector type: {e}")
            
            self.pgvector_available = self._check_pgvector_availability()
            if not self.pgvector_available:
                logger.warning("pgvector extension not available. Using fallback search.")
            
            return True
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return False

    def _check_pgvector_availability(self):
        """Check if pgvector extension is available"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                return cur.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking for pgvector extension: {e}")
            return False

    def strict_filter_stage(self, filters: Dict) -> List[str]:
        """Stage 1: Apply strict SQL filters"""
        
        # If no filters are provided, return all player IDs.
        if not filters:
            with self.conn.cursor() as cur:
                cur.execute("SELECT id FROM players WHERE status = 'active' AND deleted_at IS NULL LIMIT 1000")
                return [row[0] for row in cur.fetchall()]

        conditions = []
        params = []
        having_conditions = []
        having_params = []

        if filters.get('position') and filters['position'] != 'any':
            conditions.append("pt.position = %s")
            params.append(filters['position'])

        if filters.get('min_skill'):
            having_conditions.append("COALESCE(AVG(ps.level), 0) >= %s")
            having_params.append(filters['min_skill'])

        if filters.get('max_skill'):
            having_conditions.append("COALESCE(AVG(ps.level), 0) <= %s")
            having_params.append(filters['max_skill'])

        if filters.get('min_age'):
            conditions.append("EXTRACT(YEAR FROM AGE(p.birth_date)) >= %s")
            params.append(filters['min_age'])

        if filters.get('max_age'):
            conditions.append("EXTRACT(YEAR FROM AGE(p.birth_date)) <= %s")
            params.append(filters['max_age'])

        if filters.get('gender'):
            conditions.append("p.gender = %s")
            params.append(filters['gender'])

        # Geographic radius filter
        if filters.get('max_distance_km') and filters.get('latitude') and filters.get('longitude'):
            conditions.append("""
                (
                    6371 * acos(
                        cos(radians(%s)) * cos(radians((p.location->>'latitude')::float)) *
                        cos(radians((p.location->>'longitude')::float) - radians(%s)) +
                        sin(radians(%s)) * sin(radians((p.location->>'latitude')::float))
                    )
                ) < %s
            """)
            params.extend([
                filters['latitude'],
                filters['longitude'],
                filters['latitude'],
                filters['max_distance_km']
            ])

        where_clause = ' AND '.join(conditions) if conditions else '1=1'
        having_clause = ' AND '.join(having_conditions) if having_conditions else ''
        
        query = f"""
            SELECT p.id 
            FROM players p
            LEFT JOIN player_skills ps ON p.id = ps.player_id
            LEFT JOIN player_teams pt ON p.id = pt.player_id AND pt.end_at IS NULL
            WHERE p.status = 'active' AND p.deleted_at IS NULL
            AND {where_clause}
            GROUP BY p.id
        """
        
        if having_clause:
            query += f" HAVING {having_clause}"
        
        query += " LIMIT 1000"

        with self.conn.cursor() as cur:
            cur.execute(query, params + having_params)
            return [row[0] for row in cur.fetchall()]

    def get_players_by_ids(self, player_ids: List[str]) -> List[Dict]:
        """Fetch full player data for a list of IDs"""
        if not player_ids:
            return []

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    p.id,
                    p.first_name,
                    p.last_name,
                    p.location,
                    p.birth_date,
                    p.gender,
                    p.profile_picture,
                    p.description,
                    p.height,
                    p.weight,
                    p.tags,
                    p.availability,
                    p.created_at,
                    p.updated_at,
                    COALESCE(AVG(ps.level), 50) as avg_skill_level,
                    EXTRACT(YEAR FROM AGE(p.birth_date)) as age,
                    array_agg(DISTINCT pt.position) FILTER (WHERE pt.position IS NOT NULL) as positions,
                    json_object_agg(ps.skill, ps.level) FILTER (WHERE ps.skill IS NOT NULL) as skills
                FROM players p
                LEFT JOIN player_skills ps ON p.id = ps.player_id
                LEFT JOIN player_teams pt ON p.id = pt.player_id AND pt.end_at IS NULL
                WHERE p.id = ANY(%s)
                GROUP BY p.id
            """, (player_ids,))
            players = cur.fetchall()

        # Explicitly cast numeric fields
        for player in players:
            if player.get('age'):
                player['age'] = int(player['age'])
            if player.get('avg_skill_level'):
                player['avg_skill_level'] = float(player['avg_skill_level'])

        return [dict(p) for p in players]

    def vector_similarity_stage(self, query_vector: np.ndarray, candidate_ids: List[str], top_k: int) -> List[Dict]:
        """Stage 2: Vector similarity ranking"""
        if not self.pgvector_available or not candidate_ids:
            return []

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = """
                SELECT 
                    p.id,
                    1 - (p.embedding <=> %s::vector) AS similarity_score
                FROM players p
                WHERE p.id = ANY(%s) AND p.embedding IS NOT NULL
                ORDER BY p.embedding <=> %s::vector
                LIMIT %s
            """
            
            cur.execute(query, (
                query_vector.tolist(),
                candidate_ids,
                query_vector.tolist(),
                top_k
            ))
            results = cur.fetchall()

        return [dict(r) for r in results]

    def store_player_embedding(self, player_id: str, embedding: np.ndarray):
        """Store or update player embedding"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE players 
                    SET embedding = %s, updated_at = NOW()
                    WHERE id = %s
                """, (embedding.tolist(), player_id))
                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to store embedding for player {player_id}: {e}")
            raise

    def get_player_by_id(self, player_id: str) -> Optional[Dict]:
        """Get single player by ID"""
        players = self.get_players_by_ids([player_id])
        return players[0] if players else None

    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class ExplainabilityEngine:
    """Generate explanations for results"""

    @staticmethod
    def explain_result(player: Dict, query_context: Dict, features: Dict[str, float]) -> List[str]:
        """Generate human-readable explanations"""
        explanations = []

        if features.get('vector_similarity', 0) > 0.8:
            explanations.append("Strong overall match to your search criteria")

        if 'position' in query_context and query_context['position'] in player.get('positions', []):
            explanations.append(f"Plays in your desired position: {query_context['position']}")

        avg_skill = player.get('avg_skill_level', 0)
        min_skill = query_context.get('min_skill', 0)
        max_skill = query_context.get('max_skill', 100)
        
        if min_skill <= avg_skill <= max_skill:
            explanations.append(f"Skill level ({int(avg_skill)}) matches your requirements")

        return explanations[:3]


class RecommendationEngine:
    """k-NN based recommendations"""

    def __init__(self, search_engine: PlayerSearchEngine):
        self.search_engine = search_engine

    def more_like_this(self, player_id: str, top_k: int = 10) -> List[Dict]:
        """Find players similar to the given player"""
        seed_player = self.search_engine.get_player_by_id(player_id)
        
        if not seed_player or not self.search_engine.pgvector_available:
            return []

        with self.search_engine.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    p.id,
                    CONCAT(p.first_name, ' ', p.last_name) as name,
                    1 - (p.embedding <=> (SELECT embedding FROM players WHERE id = %s)) as similarity
                FROM players p
                WHERE p.id != %s 
                AND p.embedding IS NOT NULL
                AND p.status = 'active'
                AND p.deleted_at IS NULL
                ORDER BY p.embedding <=> (SELECT embedding FROM players WHERE id = %s)
                LIMIT %s
            """, (player_id, player_id, player_id, top_k))
            
            similar_players = cur.fetchall()

        return [dict(p) for p in similar_players]


class SavedSearchManager:
    """Manage saved searches"""

    def __init__(self, search_engine: PlayerSearchEngine):
        self.search_engine = search_engine

    def save_search(self, user_id: str, search_name: str, filters: Dict, alert_frequency: str = 'weekly') -> int:
        """Save a search for future alerts"""
        try:
            query_vector = self.search_engine.query_builder.build_from_filters(filters)

            with self.search_engine.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO saved_searches 
                    (user_id, search_name, filters, query_embedding, alert_frequency)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id;
                """, (
                    user_id,
                    search_name,
                    json.dumps(filters),
                    query_vector.tolist(),
                    alert_frequency
                ))
                search_id = cur.fetchone()[0]
                self.search_engine.conn.commit()

            return search_id
        except Exception as e:
            logger.error(f"Error saving search for user {user_id}: {e}", exc_info=True)
            self.search_engine.conn.rollback()
            raise

    def get_new_matches(self, saved_search_id: int) -> List[Dict]:
        """Get new matches for a saved search"""
        with self.search_engine.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT filters, query_embedding, last_alerted_at
                FROM saved_searches 
                WHERE id = %s
            """, (saved_search_id,))
            
            saved_search = cur.fetchone()
            if not saved_search:
                return []

            last_alerted = saved_search['last_alerted_at'] or datetime(2000, 1, 1)
            
            # Update last alerted timestamp
            cur.execute("""
                UPDATE saved_searches 
                SET last_alerted_at = NOW()
                WHERE id = %s
            """, (saved_search_id,))
            self.search_engine.conn.commit()

        return []
    
    def get_saved_searches(self, user_id: str) -> List[Dict]:
        """Get all saved searches for a user"""
        with self.search_engine.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, search_name, filters, alert_frequency, 
                       last_alerted_at, created_at
                FROM saved_searches
                WHERE user_id = %s
                ORDER BY created_at DESC;
            """, (user_id,))

            saved_searches = cur.fetchall()
            return saved_searches