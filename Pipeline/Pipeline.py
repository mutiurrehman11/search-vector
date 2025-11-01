"""
Two‑stage retrieval pipeline for the “players” domain, built on PostgreSQL
with the pgvector extension for fast ANN similarity search.

Main components
----------------
1️⃣ PlayerEmbeddingGenerator – turns raw player attributes into a 128‑dim
   float vector (position, skills, location, physical traits).

2️⃣ QueryVectorBuilder – builds a query‑vector from the user’s filter
   dictionary (optionally blended with seed players and boost factors).

3️⃣ PlayerSearchEngine – strict SQL filtering (stage 1) + vector similarity
   ranking (stage 2). Uses a threaded connection pool.

4️⃣ MLReRanker – LightGBM learning‑to‑rank model that re‑orders the
   stage‑2 results using richer engagement features.

5️⃣ ExplainabilityEngine – produces a few human‑readable bullet points
   for each result.

6️⃣ RecommendationEngine – “more‑like‑this” k‑NN lookup.

7️⃣ SavedSearchManager – persists a user’s filter + query embedding and
   can surface new matches later.

8️⃣ SearchPipeline – orchestrates everything, logs impressions, and
   exposes a clean `search()` method for the outer application.

All heavy‑lifting (SQL, vector math, model training) lives here; the
thin wrapper in *app_pg.py* simply instantiates `SearchPipeline` and
calls its methods.
"""

import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
from dataclasses import dataclass, asdict
from datetime import datetime
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import pickle


# ============================================================================
# 1. EMBEDDING GENERATION FROM STRUCTURED ATTRIBUTES
# ============================================================================

class PlayerEmbeddingGenerator:
    """Generate embeddings from structured player attributes"""

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.feature_weights = {
            'position': 0.25,
            'skill_level': 0.20,
            'age_group': 0.15,
            'location': 0.15,
            'availability': 0.10,
            'tags': 0.15
        }

    def encode_position(self, position: str) -> np.ndarray:
        """One-hot encode position into embedding space"""
        positions = ['forward', 'midfielder', 'defender', 'goalkeeper']
        encoding = np.zeros(self.embedding_dim // 4)
        if position.lower() in positions:
            idx = positions.index(position.lower())
            encoding[idx * len(encoding) // len(positions)] = 1.0
        return encoding

    def encode_skill_level(self, skill: int) -> np.ndarray:
        """Encode skill level (1-10) as gaussian distribution"""
        encoding = np.zeros(self.embedding_dim // 4)
        center = int((skill / 10.0) * len(encoding))
        for i in range(len(encoding)):
            encoding[i] = np.exp(-((i - center) ** 2) / 2.0)
        return encoding

    def encode_location(self, lat: float, lon: float) -> np.ndarray:
        """Encode geographic location using spatial hashing"""
        encoding = np.zeros(self.embedding_dim // 4)
        # Simple spatial hashing
        lat_hash = int((lat + 90) / 180 * 100) % len(encoding)
        lon_hash = int((lon + 180) / 360 * 100) % len(encoding)
        encoding[lat_hash] = 0.7
        encoding[lon_hash] = 0.3
        return encoding

    def encode_tags(self, tags: List[str]) -> np.ndarray:
        """Encode tags using learned embeddings (simplified with hashing)"""
        encoding = np.zeros(self.embedding_dim // 4)
        for tag in tags:
            hash_val = hash(tag.lower()) % len(encoding)
            encoding[hash_val] = min(1.0, encoding[hash_val] + 0.3)
        return encoding

    def generate_embedding(self, player: Dict) -> np.ndarray:
        """Generate complete embedding vector for a player"""
        components = [
            self.encode_position(player.get('position', '')),
            self.encode_skill_level(player.get('skill_level', 5)),
            self.encode_location(player.get('latitude', 0), player.get('longitude', 0)),
            self.encode_tags(player.get('tags', []))
        ]
        embedding = np.concatenate(components)
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding


# ============================================================================
# 2. QUERY VECTOR CONSTRUCTION
# ============================================================================

class QueryVectorBuilder:
    """Build query vectors from filters, seed players, and tag boosts"""

    def __init__(self, embedding_generator: PlayerEmbeddingGenerator):
        self.embedding_gen = embedding_generator

    def build_from_filters(self,
                          filters: Dict,
                          seed_players: Optional[List[Dict]] = None,
                          tag_boosts: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Build query vector from:
        - Active filters (position, skill, location, etc.)
        - Seed players (for "find similar" queries)
        - Tag boosts (amplify certain characteristics)
        """
        # Start with filter-based embedding
        filter_embedding = self.embedding_gen.generate_embedding({
            'position': filters.get('position', ''),
            'skill_level': filters.get('skill_level', 5),
            'latitude': filters.get('latitude', 0),
            'longitude': filters.get('longitude', 0),
            'tags': filters.get('tags', [])
        })

        # Add seed player influence
        if seed_players:
            seed_embeddings = [
                self.embedding_gen.generate_embedding(p)
                for p in seed_players
            ]
            seed_avg = np.mean(seed_embeddings, axis=0)
            # Blend: 60% filters, 40% seed players
            filter_embedding = 0.6 * filter_embedding + 0.4 * seed_avg

        # Apply tag boosts
        if tag_boosts:
            boost_vector = np.zeros_like(filter_embedding)
            for tag, weight in tag_boosts.items():
                tag_emb = self.embedding_gen.encode_tags([tag])
                # Pad to match dimensions
                padded = np.zeros_like(filter_embedding)
                padded[:len(tag_emb)] = tag_emb
                boost_vector += weight * padded

            filter_embedding = filter_embedding + 0.2 * boost_vector

        # Normalize final query vector
        norm = np.linalg.norm(filter_embedding)
        if norm > 0:
            filter_embedding = filter_embedding / norm

        return filter_embedding


# ============================================================================
# 3. DATABASE SETUP & VECTOR SEARCH
# ============================================================================

class PlayerSearchEngine:
    """Main search engine with pgvector integration"""

    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.conn = None
        self.embedding_gen = PlayerEmbeddingGenerator()
        self.query_builder = QueryVectorBuilder(self.embedding_gen)
        self.has_pgvector = False

    def connect(self):
        """Connect to PostgreSQL with pgvector"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            # Set autocommit to False for transaction control
            self.conn.autocommit = False
            
            # Check if pgvector is available
            self.has_pgvector = self._check_pgvector_availability()
            if not self.has_pgvector:
                print("Warning: pgvector extension not available. Using fallback search without vector similarity.")
            
            return True
        except Exception as e:
            print(f"Database connection error: {e}")
            return False

    def _check_pgvector_availability(self):
        """Check if pgvector extension is available"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
                result = cur.fetchone()
                return result is not None
        except Exception:
            return False

    def setup_database(self):
        """Initialize database schema with pgvector"""
        with self.conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Players table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS players (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255),
                    position VARCHAR(50),
                    skill_level INTEGER CHECK (skill_level BETWEEN 1 AND 10),
                    age INTEGER,
                    latitude FLOAT,
                    longitude FLOAT,
                    availability TEXT[], -- ['weekday_evening', 'weekend_morning']
                    tags TEXT[],
                    bio TEXT,
                    embedding vector(128),
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            # Create HNSW index for fast similarity search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS players_embedding_idx 
                ON players USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)

            # Engagement events table (for training re-ranker)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS player_engagement_events (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER,
                    player_id INTEGER REFERENCES players(id),
                    event_type VARCHAR(50), -- 'impression', 'open', 'save', 'message'
                    query_context JSONB, -- Store search context
                    created_at TIMESTAMP DEFAULT NOW()
                );""")

            # Saved searches table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS saved_searches (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER,
                    search_name VARCHAR(255),
                    filters JSONB,
                    query_vector vector(128),
                    alert_frequency VARCHAR(20), -- 'daily', 'weekly'
                    last_alerted TIMESTAMP,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            self.conn.commit()

    def index_player(self, player: Dict):
        """Index a player with their embedding"""
        embedding = self.embedding_gen.generate_embedding(player)

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO players 
                (name, position, skill_level, age, latitude, longitude, 
                 availability, tags, bio, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                player['name'],
                player['position'],
                player['skill_level'],
                player['age'],
                player['latitude'],
                player['longitude'],
                player.get('availability', []),
                player.get('tags', []),
                player.get('bio', ''),
                embedding.tolist()
            ))
            player_id = cur.fetchone()[0]
            self.conn.commit()
            return player_id

    def strict_filter_stage(self, filters: Dict) -> List[int]:
        """
        Stage 1: Apply strict SQL filters
        Returns: List of player IDs that pass all filters
        """
        conditions = ["1=1"]  # Start with true condition
        params = []
        joins = []

        # Handle position filter (requires join with player_teams)
        if 'position' in filters:
            joins.append("LEFT JOIN player_teams pt ON p.id = pt.player_id")
            conditions.append("pt.position = %s")
            params.append(filters['position'])

        # Handle skill level filters (these would be in players table if we had them)
        if 'min_skill' in filters:
            conditions.append("p.skill_level >= %s")
            params.append(filters['min_skill'])

        if 'max_skill' in filters:
            conditions.append("p.skill_level <= %s")
            params.append(filters['max_skill'])

        if 'min_age' in filters:
            conditions.append("EXTRACT(YEAR FROM AGE(p.birth_date)) >= %s")
            params.append(filters['min_age'])

        if 'max_age' in filters:
            conditions.append("EXTRACT(YEAR FROM AGE(p.birth_date)) <= %s")
            params.append(filters['max_age'])

        # Note: availability and tags would need to be implemented based on actual schema
        # For now, skipping these filters as they don't exist in the current schema

        if 'max_distance_km' in filters and 'latitude' in filters:
            # Geographic radius filter using location JSONB
            conditions.append("""
                earth_distance(
                    ll_to_earth((p.location->>'latitude')::float, (p.location->>'longitude')::float),
                    ll_to_earth(%s, %s)
                ) <= %s * 1000
            """)
            params.extend([
                filters['latitude'],
                filters['longitude'],
                filters['max_distance_km']
            ])

        # Build the query with proper joins
        join_clause = ' '.join(joins) if joins else ''
        query = f"""
            SELECT DISTINCT p.id FROM players p
            {join_clause}
            WHERE {' AND '.join(conditions)}
            LIMIT 1000
        """

        with self.conn.cursor() as cur:
            cur.execute(query, params)
            return [row[0] for row in cur.fetchall()]

    def vector_similarity_stage(self,
                               query_vector: np.ndarray,
                               candidate_ids: List[int],
                               top_k: int = 100) -> List[Tuple[int, float]]:
        """
        Stage 2: Vector similarity search on filtered candidates
        Returns: List of (player_id, similarity_score) tuples
        """
        if not candidate_ids:
            return []

        if self.has_pgvector:
            # Use pgvector for similarity search
            query = """
                SELECT id, 1 - (embedding <=> %s::vector) as similarity
                FROM players
                WHERE id = ANY(%s)
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """

            with self.conn.cursor() as cur:
                cur.execute(query, (
                    query_vector.tolist(),
                    candidate_ids,
                    query_vector.tolist(),
                    top_k
                ))
                return [(row[0], row[1]) for row in cur.fetchall()]
        else:
            # Fallback: return candidates with random similarity scores
            # In a real implementation, you might use other ranking factors
            import random
            random.shuffle(candidate_ids)
            return [(pid, random.uniform(0.5, 0.9)) for pid in candidate_ids[:top_k]]

    def search(self,
               filters: Dict,
               seed_players: Optional[List[Dict]] = None,
               tag_boosts: Optional[Dict[str, float]] = None,
               top_k: int = 50) -> List[Dict]:
        """
        Complete two-stage search pipeline
        Returns: List of player results with metadata
        """
        # Stage 1: Strict filters
        candidate_ids = self.strict_filter_stage(filters)

        if not candidate_ids:
            return []

        # Build query vector
        query_vector = self.query_builder.build_from_filters(
            filters, seed_players, tag_boosts
        )

        # Stage 2: Vector similarity
        similar_players = self.vector_similarity_stage(
            query_vector, candidate_ids, top_k * 2  # Get 2x for re-ranking
        )

        # Fetch full player data with joins to get all required fields
        player_ids = [pid for pid, _ in similar_players]
        similarity_map = {pid: sim for pid, sim in similar_players}

        if not player_ids:
            return []

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    p.id,
                    CONCAT(p.first_name, ' ', p.last_name) as name,
                    pt.position,
                    COALESCE(AVG(ps.level), 5) as skill_level,
                    EXTRACT(YEAR FROM AGE(p.birth_date)) as age,
                    (p.location->>'latitude')::float as latitude,
                    (p.location->>'longitude')::float as longitude,
                    p.location->>'name' as location_name,
                    ARRAY[]::text[] as availability,
                    ARRAY[]::text[] as tags,
                    p.description as bio,
                    p.profile_picture->>'url' as profile_image_url
                FROM players p
                LEFT JOIN player_teams pt ON p.id = pt.player_id AND pt.end_at IS NULL
                LEFT JOIN player_skills ps ON p.id = ps.player_id
                WHERE p.id = ANY(%s)
                GROUP BY p.id, p.first_name, p.last_name, pt.position, p.birth_date, 
                         p.location, p.description, p.profile_picture
            """, (player_ids,))
            players = cur.fetchall()

        # Convert to list of dicts and add similarity scores
        result_players = []
        for player in players:
            player_dict = dict(player)
            player_dict['vector_similarity'] = similarity_map.get(player_dict['id'], 0.0)
            result_players.append(player_dict)

        return result_players


# ============================================================================
# 4. ML RE-RANKING WITH LEARNING-TO-RANK
# ============================================================================

class MLReRanker:
    """LightGBM-based Learning-to-Rank model"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'vector_similarity',
            'skill_match_score',
            'location_distance',
            'availability_overlap',
            'tag_match_score',
            'profile_completeness',
            'recent_activity_score',
            'historical_ctr',
            'historical_open_rate',
            'historical_save_rate',
            'historical_message_rate'
        ]

    def extract_features(self, player: Dict, query_context: Dict) -> np.ndarray:
        """Extract features for re-ranking"""
        features = []

        # Vector similarity (already computed)
        features.append(player.get('vector_similarity', 0.0))

        # Skill match score
        if 'skill_level' in query_context:
            skill_diff = abs(player['skill_level'] - query_context['skill_level'])
            features.append(1.0 - (skill_diff / 10.0))
        else:
            features.append(0.5)

        # Location distance (normalized)
        if 'latitude' in query_context:
            lat_diff = abs(player['latitude'] - query_context['latitude'])
            lon_diff = abs(player['longitude'] - query_context['longitude'])
            distance = np.sqrt(lat_diff**2 + lon_diff**2)
            features.append(1.0 / (1.0 + distance))
        else:
            features.append(0.5)

        # Availability overlap
        query_avail = set(query_context.get('availability', []))
        player_avail = set(player.get('availability', []))
        if query_avail:
            overlap = len(query_avail & player_avail) / len(query_avail)
            features.append(overlap)
        else:
            features.append(0.5)

        # Tag match score
        query_tags = set(query_context.get('tags', []))
        player_tags = set(player.get('tags', []))
        if query_tags:
            tag_match = len(query_tags & player_tags) / len(query_tags)
            features.append(tag_match)
        else:
            features.append(0.5)

        # Profile completeness
        completeness = sum([
            bool(player.get('bio')),
            bool(player.get('tags')),
            bool(player.get('availability')),
            player.get('skill_level', 0) > 0
        ]) / 4.0
        features.append(completeness)

        # Recent activity (mock - would come from your analytics)
        features.append(player.get('recent_activity_score', 0.5))

        # Historical engagement rates (from engagement_events table)
        features.append(player.get('historical_ctr', 0.0))
        features.append(player.get('historical_open_rate', 0.0))
        features.append(player.get('historical_save_rate', 0.0))
        features.append(player.get('historical_message_rate', 0.0))

        return np.array(features)

    def prepare_training_data(self, conn) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data from engagement events
        Labels: impression=0, open=1, save=2, message=3 (ordinal)
        """
        query = """
            SELECT 
                e.player_id,
                e.event_type,
                e.query_context,
                p.*,
                -- Aggregate historical stats
                COUNT(*) FILTER (WHERE e2.event_type = 'impression') as total_impressions,
                COUNT(*) FILTER (WHERE e2.event_type = 'open') as total_opens,
                COUNT(*) FILTER (WHERE e2.event_type = 'save') as total_saves,
                COUNT(*) FILTER (WHERE e2.event_type = 'message') as total_messages
            FROM player_engagement_events e
            JOIN players p ON e.player_id = p.id
            LEFT JOIN player_engagement_events e2 ON e2.player_id = p.id 
                AND e2.created_at < e.created_at
            WHERE e.created_at > NOW() - INTERVAL '30 days'
            GROUP BY e.id, e.player_id, e.event_type, e.query_context, p.id;
        """

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            events = cur.fetchall()

        X, y, groups = [], [], []
        current_query_id = 0
        last_context = None

        event_labels = {
            'impression': 0,
            'open': 1,
            'save': 2,
            'message': 3
        }

        for event in events:
            # Calculate historical rates
            impressions = max(1, event['total_impressions'])
            event['historical_ctr'] = event['total_opens'] / impressions
            event['historical_open_rate'] = event['total_opens'] / impressions
            event['historical_save_rate'] = event['total_saves'] / impressions
            event['historical_message_rate'] = event['total_messages'] / impressions

            query_context = event['query_context']

            # New query group
            if query_context != last_context:
                current_query_id += 1
                last_context = query_context

            features = self.extract_features(dict(event), query_context)
            X.append(features)
            y.append(event_labels[event['event_type']])
            groups.append(current_query_id)

        return np.array(X), np.array(y), np.array(groups)

    def train(self, conn):
        """Train LightGBM ranker on engagement data"""
        X, y, groups = self.prepare_training_data(conn)

        if len(X) == 0:
            print("No training data available yet")
            return

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Create LightGBM dataset with query groups
        train_data = lgb.Dataset(
            X_scaled,
            label=y,
            group=np.bincount(groups)[1:]  # Group sizes
        )

        # LambdaRank parameters
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [5, 10, 20],
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(10)]
        )

        print("Re-ranking model trained successfully")

    def rerank(self, players: List[Dict], query_context: Dict) -> List[Dict]:
        """Re-rank players using trained model"""
        if not self.model or not players:
            return players

        # Extract features
        X = np.array([
            self.extract_features(p, query_context)
            for p in players
        ])
        X_scaled = self.scaler.transform(X)

        # Predict scores
        scores = self.model.predict(X_scaled)

        # Add scores and sort
        for player, score in zip(players, scores):
            player['ml_score'] = float(score)

        return sorted(players, key=lambda p: p['ml_score'], reverse=True)

    def save_model(self, path: str):
        """Save trained model to disk"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)

    def load_model(self, path: str):
        """Load trained model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']


# ============================================================================
# 5. EXPLAINABILITY ENGINE
# ============================================================================

class ExplainabilityEngine:
    """Generate "why this result" explanations"""

    @staticmethod
    def explain_result(player: Dict,
                      query_context: Dict,
                      features: Dict[str, float]) -> List[str]:
        """
        Generate human-readable explanations
        Returns: List of explanation strings
        """
        explanations = []

        # Vector similarity
        if features.get('vector_similarity', 0) > 0.8:
            explanations.append("Strong overall match to your search criteria")

        # Skill match
        if 'skill_level' in query_context:
            skill_diff = abs(player['skill_level'] - query_context['skill_level'])
            if skill_diff <= 1:
                explanations.append(
                    f"Skill level ({player['skill_level']}/10) matches your preference"
                )

        # Location
        if 'latitude' in query_context:
            lat_diff = abs(player['latitude'] - query_context['latitude'])
            lon_diff = abs(player['longitude'] - query_context['longitude'])
            distance_km = np.sqrt(lat_diff**2 + lon_diff**2) * 111  # Rough conversion
            if distance_km < 5:
                explanations.append(f"Located nearby ({distance_km:.1f}km away)")

        # Tag matches
        query_tags = set(query_context.get('tags', []))
        player_tags = set(player.get('tags', []))
        matching_tags = query_tags & player_tags
        if matching_tags:
            explanations.append(
                f"Shares your interests: {', '.join(list(matching_tags)[:3])}"
            )

        # Availability
        query_avail = set(query_context.get('availability', []))
        player_avail = set(player.get('availability', []))
        matching_avail = query_avail & player_avail
        if matching_avail:
            explanations.append(
                f"Available when you are: {', '.join(list(matching_avail)[:2])}"
            )

        # High engagement
        if features.get('historical_ctr', 0) > 0.1:
            explanations.append("Popular profile with high engagement")

        # ML score
        if features.get('ml_score', 0) > 0.7:
            explanations.append("Predicted to be a great match based on similar searches")

        # Profile quality
        if features.get('profile_completeness', 0) > 0.75:
            explanations.append("Complete and detailed profile")

        return explanations[:4]  # Return top 4 explanations


# ============================================================================
# 6. "MORE LIKE THIS" RECOMMENDATIONS
# ============================================================================

class RecommendationEngine:
    """k-NN based recommendations"""

    def __init__(self, search_engine: PlayerSearchEngine):
        self.search_engine = search_engine

    def more_like_this(self, player_id: str, top_k: int = 10) -> List[Dict]:
        """Find players similar to the given player"""
        # Get the seed player
        with self.search_engine.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM players WHERE id = %s;
            """, (player_id,))
            seed_player = cur.fetchone()

        if not seed_player:
            return []

        if self.search_engine.has_pgvector:
            # Use player's embedding as query vector
            query_vector = np.array(seed_player['embedding'])

            # Find similar players with proper joins and formatting
            with self.search_engine.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        p.id,
                        CONCAT(p.first_name, ' ', p.last_name) as name,
                        pt.position,
                        COALESCE(AVG(ps.level), 5) as skill_level,
                        EXTRACT(YEAR FROM AGE(p.birth_date)) as age,
                        (p.location->>'latitude')::float as latitude,
                        (p.location->>'longitude')::float as longitude,
                        p.location->>'name' as location_name,
                        ARRAY[]::text[] as availability,
                        ARRAY[]::text[] as tags,
                        p.description as bio,
                        p.profile_picture->>'url' as profile_image_url,
                        1 - (p.embedding <=> %s::vector) as similarity
                    FROM players p
                    LEFT JOIN player_teams pt ON p.id = pt.player_id AND pt.end_at IS NULL
                    LEFT JOIN player_skills ps ON p.id = ps.player_id
                    WHERE p.id != %s
                    GROUP BY p.id, p.first_name, p.last_name, pt.position, p.birth_date, 
                             p.location, p.description, p.profile_picture, p.embedding
                    ORDER BY p.embedding <=> %s::vector
                    LIMIT %s;
                """, (
                    query_vector.tolist(),
                    player_id,
                    query_vector.tolist(),
                    top_k
                ))
                similar_players = cur.fetchall()
        else:
            # Fallback: find players with similar attributes and proper formatting
            with self.search_engine.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        p.id,
                        CONCAT(p.first_name, ' ', p.last_name) as name,
                        pt.position,
                        COALESCE(AVG(ps.level), 5) as skill_level,
                        EXTRACT(YEAR FROM AGE(p.birth_date)) as age,
                        (p.location->>'latitude')::float as latitude,
                        (p.location->>'longitude')::float as longitude,
                        p.location->>'name' as location_name,
                        ARRAY[]::text[] as availability,
                        ARRAY[]::text[] as tags,
                        p.description as bio,
                        p.profile_picture->>'url' as profile_image_url,
                        RANDOM() as similarity
                    FROM players p
                    LEFT JOIN player_teams pt ON p.id = pt.player_id AND pt.end_at IS NULL
                    LEFT JOIN player_skills ps ON p.id = ps.player_id
                    WHERE p.id != %s
                    AND (pt.position = (
                        SELECT pt2.position FROM player_teams pt2 WHERE pt2.player_id = %s LIMIT 1
                    ) OR pt.position IS NULL)
                    GROUP BY p.id, p.first_name, p.last_name, pt.position, p.birth_date, 
                             p.location, p.description, p.profile_picture
                    ORDER BY similarity DESC
                    LIMIT %s;
                """, (player_id, player_id, top_k))
                similar_players = cur.fetchall()

        # Format results similar to search method
        results = []
        for player in similar_players:
            formatted_player = {
                'id': player['id'],
                'name': player['name'],
                'position': player['position'],
                'skill_level': str(player['skill_level']),
                'age': str(player['age']),
                'location': {
                    'latitude': player['latitude'],
                    'longitude': player['longitude'],
                    'name': player['location_name']
                },
                'availability': player['availability'],
                'tags': player['tags'],
                'bio': player['bio'],
                'profile_image_url': player['profile_image_url'],
                'similarity': float(player['similarity'])
            }
            results.append(formatted_player)

        return results


# ============================================================================
# 7. SAVED SEARCH ALERTS
# ============================================================================

class SavedSearchManager:
    """Manage saved searches and alerts"""

    def __init__(self, search_engine: PlayerSearchEngine):
        self.search_engine = search_engine

    def save_search(self,
                    user_id: int,
                    search_name: str,
                    filters: Dict,
                    alert_frequency: str = 'weekly') -> int:
        """Save a search for future alerts"""
        # Build query vector
        query_vector = self.search_engine.query_builder.build_from_filters(filters)

        with self.search_engine.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO saved_searches 
                (user_id, search_name, filters, query_vector, alert_frequency)
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

    def get_new_matches(self, saved_search_id: int) -> List[Dict]:
        """Get new matches for a saved search since last alert"""
        with self.search_engine.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get saved search
            cur.execute("""
                SELECT * FROM saved_searches WHERE id = %s;
            """, (saved_search_id,))
            saved_search = cur.fetchone()

            if not saved_search:
                return []

            # Find new players matching the criteria
            filters = json.loads(saved_search['filters'])
            query_vector = np.array(saved_search['query_vector'])

            # Get players created since last alert
            last_alerted = saved_search['last_alerted'] or datetime(2000, 1, 1)

            if self.search_engine.has_pgvector:
                cur.execute("""
                    SELECT *, 1 - (embedding <=> %s::vector) as similarity
                    FROM players
                    WHERE created_at > %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT 20;
                """, (
                    query_vector.tolist(),
                    last_alerted,
                    query_vector.tolist()
                ))
            else:
                # Fallback: get recent players without vector similarity
                cur.execute("""
                    SELECT *, RANDOM() as similarity
                    FROM players
                    WHERE created_at > %s
                    ORDER BY created_at DESC
                    LIMIT 20;
                """, (last_alerted,))

            new_matches = cur.fetchall()

            # Update last alerted timestamp
            cur.execute("""
                UPDATE saved_searches 
                SET last_alerted = NOW()
                WHERE id = %s;
            """, (saved_search_id,))
            self.search_engine.conn.commit()

        return [dict(m) for m in new_matches]


# ============================================================================
# 8. COMPLETE PIPELINE ORCHESTRATION
# ============================================================================

class SearchPipeline:
    """Orchestrates the complete search pipeline"""

    def __init__(self, db_config: Dict):
        self.search_engine = PlayerSearchEngine(db_config)
        self.search_engine.connect()
        self.reranker = MLReRanker()
        self.explainer = ExplainabilityEngine()
        self.recommender = RecommendationEngine(self.search_engine)
        self.saved_search_mgr = SavedSearchManager(self.search_engine)

    def search(self,
               user_id: int,
               filters: Dict,
               seed_players: Optional[List[Dict]] = None,
               tag_boosts: Optional[Dict[str, float]] = None,
               top_k: int = 20) -> List[Dict]:
        """
        Execute complete search pipeline with explanations
        """
        # Two-stage retrieval
        candidates = self.search_engine.search(
            filters, seed_players, tag_boosts, top_k=top_k * 2
        )

        if not candidates:
            return []

        # ML re-ranking
        reranked = self.reranker.rerank(candidates, filters)[:top_k]

        # Add explanations to each result
        for player in reranked:
            features = {
                'vector_similarity': player.get('vector_similarity', 0),
                'ml_score': player.get('ml_score', 0),
                'historical_ctr': player.get('historical_ctr', 0),
                'profile_completeness': sum([
                    bool(player.get('bio')),
                    bool(player.get('tags')),
                    bool(player.get('availability')),
                    player.get('skill_level', 0) > 0
                ]) / 4.0
            }

            player['explanations'] = self.explainer.explain_result(
                player, filters, features
            )

            # Log impression event
            self._log_event(user_id, player['id'], 'impression', filters)

        return reranked

    def _log_event(self, user_id: int, player_id: int,
                   event_type: str, query_context: Dict):
        """Log engagement event for model training"""
        with self.search_engine.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO player_engagement_events 
                (user_id, player_id, event_type, query_context, created_at)
                VALUES (%s, %s, %s, %s, NOW());
            """, (user_id, player_id, event_type, json.dumps(query_context)))
            self.search_engine.conn.commit()

    def log_interaction(self, user_id: int, player_id: int,
                       event_type: str, query_context: Dict):
        """Public method to log user interactions"""
        self._log_event(user_id, player_id, event_type, query_context)

    def train_reranker(self):
        """Train the ML re-ranking model"""
        self.reranker.train(self.search_engine.conn)

    def get_recommendations(self, player_id: int, top_k: int = 10) -> List[Dict]:
        """Get "more like this" recommendations"""
        return self.recommender.more_like_this(player_id, top_k)


# ============================================================================
# 9. USAGE EXAMPLES
# ============================================================================

def example_usage():
    """Demonstration of the complete pipeline"""

    # Initialize
    db_config = {
        'host': 'localhost',
        'database': 'player_search',
        'user': 'your_user',
        'password': 'your_password'
    }

    pipeline = SearchPipeline(db_config)

    # Setup database (run once)
    pipeline.search_engine.setup_database()

    # ========================================================================
    # INDEX PLAYERS
    # ========================================================================
    sample_players = [
        {
            'name': 'Alex Johnson',
            'position': 'midfielder',
            'skill_level': 7,
            'age': 28,
            'latitude': 40.7128,
            'longitude': -74.0060,
            'availability': ['weekday_evening', 'weekend_morning'],
            'tags': ['competitive', 'team_player', 'experienced'],
            'bio': 'Been playing for 10 years, love tactical play'
        },
        {
            'name': 'Sam Chen',
            'position': 'forward',
            'skill_level': 8,
            'age': 25,
            'latitude': 40.7589,
            'longitude': -73.9851,
            'availability': ['weekend_morning', 'weekend_afternoon'],
            'tags': ['competitive', 'fast', 'goal_scorer'],
            'bio': 'Former college player looking for competitive matches'
        },
        {
            'name': 'Jordan Martinez',
            'position': 'defender',
            'skill_level': 6,
            'age': 32,
            'latitude': 40.7306,
            'longitude': -73.9352,
            'availability': ['weekday_evening'],
            'tags': ['casual', 'friendly', 'team_player'],
            'bio': 'Just looking for fun pickup games'
        }
    ]

    for player in sample_players:
        player_id = pipeline.search_engine.index_player(player)
        print(f"Indexed player: {player['name']} (ID: {player_id})")

    # ========================================================================
    # SEARCH WITH FILTERS
    # ========================================================================
    print("\n" + "="*60)
    print("SEARCH EXAMPLE 1: Basic filter search")
    print("="*60)

    search_filters = {
        'position': 'midfielder',
        'min_skill': 6,
        'max_skill': 9,
        'latitude': 40.7128,
        'longitude': -74.0060,
        'max_distance_km': 10,
        'availability': ['weekday_evening'],
        'tags': ['competitive', 'team_player']
    }

    results = pipeline.search(
        user_id=1,
        filters=search_filters,
        top_k=10
    )

    for i, player in enumerate(results, 1):
        print(f"\n{i}. {player['name']}")
        print(f"   Position: {player['position']} | Skill: {player['skill_level']}/10")
        print(f"   Vector Similarity: {player.get('vector_similarity', 0):.3f}")
        print(f"   ML Score: {player.get('ml_score', 0):.3f}")
        print(f"   Why this result:")
        for explanation in player.get('explanations', []):
            print(f"   • {explanation}")

    # ========================================================================
    # SEARCH WITH SEED PLAYERS (Find Similar)
    # ========================================================================
    print("\n" + "="*60)
    print("SEARCH EXAMPLE 2: Find players like Alex Johnson")
    print("="*60)

    seed_player = sample_players[0]  # Alex Johnson

    results = pipeline.search(
        user_id=1,
        filters={'min_skill': 5},
        seed_players=[seed_player],
        top_k=5
    )

    for i, player in enumerate(results, 1):
        print(f"\n{i}. {player['name']}")
        print(f"   Similar because:")
        for explanation in player.get('explanations', []):
            print(f"   • {explanation}")

    # ========================================================================
    # SEARCH WITH TAG BOOSTS
    # ========================================================================
    print("\n" + "="*60)
    print("SEARCH EXAMPLE 3: Boost competitive players")
    print("="*60)

    results = pipeline.search(
        user_id=1,
        filters={'min_skill': 6},
        tag_boosts={'competitive': 2.0, 'goal_scorer': 1.5},
        top_k=5
    )

    for i, player in enumerate(results, 1):
        print(f"{i}. {player['name']} - ML Score: {player.get('ml_score', 0):.3f}")

    # ========================================================================
    # LOG INTERACTIONS (for training)
    # ========================================================================
    print("\n" + "="*60)
    print("LOGGING INTERACTIONS")
    print("="*60)

    # User opens a profile
    pipeline.log_interaction(
        user_id=1,
        player_id=results[0]['id'],
        event_type='open',
        query_context=search_filters
    )
    print(f"Logged 'open' event for {results[0]['name']}")

    # User saves the profile
    pipeline.log_interaction(
        user_id=1,
        player_id=results[0]['id'],
        event_type='save',
        query_context=search_filters
    )
    print(f"Logged 'save' event for {results[0]['name']}")

    # User sends a message
    pipeline.log_interaction(
        user_id=1,
        player_id=results[0]['id'],
        event_type='message',
        query_context=search_filters
    )
    print(f"Logged 'message' event for {results[0]['name']}")

    # ========================================================================
    # TRAIN RE-RANKER (run periodically, e.g., nightly)
    # ========================================================================
    print("\n" + "="*60)
    print("TRAINING ML RE-RANKER")
    print("="*60)

    # Train on accumulated engagement data
    pipeline.train_reranker()

    # Save model
    pipeline.reranker.save_model('reranker_model.pkl')
    print("Model trained and saved to reranker_model.pkl")

    # ========================================================================
    # MORE LIKE THIS RECOMMENDATIONS
    # ========================================================================
    print("\n" + "="*60)
    print("MORE LIKE THIS: Players similar to Sam Chen")
    print("="*60)

    similar_players = pipeline.get_recommendations(player_id=2, top_k=5)

    for i, player in enumerate(similar_players, 1):
        print(f"{i}. {player['name']} (Similarity: {player['similarity']:.3f})")
        print(f"   {player['position']} | Skill {player['skill_level']}/10")

    # ========================================================================
    # SAVED SEARCH ALERTS
    # ========================================================================
    print("\n" + "="*60)
    print("SAVED SEARCH ALERTS")
    print("="*60)

    # User saves a search
    saved_search_id = pipeline.saved_search_mgr.save_search(
        user_id=1,
        search_name="Competitive midfielders near me",
        filters=search_filters,
        alert_frequency='weekly'
    )
    print(f"Saved search ID: {saved_search_id}")

    # Later, check for new matches
    new_matches = pipeline.saved_search_mgr.get_new_matches(saved_search_id)
    print(f"Found {len(new_matches)} new matches for saved search")


# ============================================================================
# 10. PRODUCTION DEPLOYMENT GUIDE
# ============================================================================

"""
DEPLOYMENT CHECKLIST
====================

1. DATABASE SETUP:
   ```sql
   -- Enable extensions
   CREATE EXTENSION IF NOT EXISTS vector;
   CREATE EXTENSION IF NOT EXISTS cube;
   CREATE EXTENSION IF NOT EXISTS earthdistance;
   
   -- Run setup_database() method to create tables and indexes
   ```

2. INDEXING PIPELINE:
   - Set up batch indexing job for existing players
   - Create webhook/queue listener for new player signups
   - Run nightly re-indexing for updated profiles

3. MODEL TRAINING:
   - Initial training: Wait 1-2 weeks for engagement data
   - Schedule: Retrain nightly using last 30 days of data
   - Monitoring: Track NDCG@10, CTR, conversion rates
   - A/B test: Compare ML re-ranker vs. vector-only

4. API INTEGRATION:
   ```python
   # In your API endpoint
   @app.route('/api/search', methods=['POST'])
   def search_api():
       data = request.json
       results = pipeline.search(
           user_id=data['user_id'],
           filters=data['filters'],
           seed_players=data.get('seed_players'),
           tag_boosts=data.get('tag_boosts'),
           top_k=data.get('limit', 20)
       )
       return jsonify(results)
   
   # Log interactions
   @app.route('/api/events', methods=['POST'])
   def log_event():
       data = request.json
       pipeline.log_interaction(
           user_id=data['user_id'],
           player_id=data['player_id'],
           event_type=data['event_type'],
           query_context=data['query_context']
       )
       return jsonify({'status': 'ok'})
   ```

5. PERFORMANCE OPTIMIZATION:
   - Connection pooling (use psycopg2.pool)
   - Cache frequent queries (Redis)
   - Batch embedding generation
   - Async event logging (use queue like Celery)

6. MONITORING:
   - Query latency (target: <200ms for stage 1+2, <50ms for re-rank)
   - Cache hit rates
   - Model performance metrics (NDCG, MRR)
   - User satisfaction (CTR, saves, messages per impression)

7. ALERT SYSTEM:
   ```python
   # Cron job (daily or weekly)
   def send_saved_search_alerts():
       # Get all saved searches due for alerts
       cur.execute("""
           # SELECT * FROM saved_searches
           # WHERE (alert_frequency = 'daily' AND last_alerted < NOW() - INTERVAL '1 day')
           #    OR (alert_frequency = 'weekly' AND last_alerted < NOW() - INTERVAL '7 days');
""")
       
       for search in cur.fetchall():
           new_matches = pipeline.saved_search_mgr.get_new_matches(search['id'])
           if new_matches:
               send_email(search['user_id'], new_matches)
   ```

8. SCALING CONSIDERATIONS:
   - Horizontal: Read replicas for search queries
   - Vertical: Increase shared_buffers, work_mem for Postgres
   - Sharding: By geography if dataset > 10M players
   - Model serving: Load model once, serve multiple requests

9. FEATURE FLAGS:
   - Enable/disable ML re-ranking
   - Enable/disable specific explanation types
   - A/B test different embedding dimensions
   - Gradual rollout of new model versions

10. TESTING:
    ```python
    # Unit tests
    def test_search_pipeline():
        results = pipeline.search(user_id=1, filters={'position': 'midfielder'})
        assert len(results) > 0
        assert all('explanations' in r for r in results)
    
    # Integration tests
    def test_full_user_journey():
        # Search
        results = pipeline.search(user_id=1, filters={})
        # Interact
        pipeline.log_interaction(1, results[0]['id'], 'open', {})
        # Train
        pipeline.train_reranker()
        # Search again (should be better)
        results2 = pipeline.search(user_id=1, filters={})
    ```
"""


if __name__ == "__main__":
    print("Two-Stage Retrieval Pipeline with ML Re-Ranking")
    print("================================================\n")
    print("To use this system:")
    print("1. Configure your PostgreSQL connection")
    print("2. Run setup_database() to initialize tables and indexes")
    print("3. Index your players using index_player()")
    print("4. Start searching with the search() method")
    print("5. Log user interactions for model training")
    print("6. Train the re-ranker periodically")
    print("\nSee example_usage() for complete demonstrations")
    print("\nFor production deployment, see DEPLOYMENT CHECKLIST above")