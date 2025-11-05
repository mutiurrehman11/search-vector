import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CompatPlayerSearchEngine:
    
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.conn = None
        self.pgvector_available = False
        self.connect()
        self._check_pgvector_availability()
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def _check_pgvector_availability(self):
        """Check if pgvector extension is available or custom vector functions exist"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
                result = cur.fetchone()
                if result is not None:
                    self.pgvector_available = True
                    logger.info("pgvector extension available")
                    return
                
                cur.execute("""
                    SELECT 1 FROM pg_proc 
                    WHERE proname = 'vector_cosine_similarity' 
                    AND pronargs = 2
                """)
                result = cur.fetchone()
                if result is not None:
                    self.pgvector_available = True
                    logger.info("Custom vector functions available")
                    return
                
                self.pgvector_available = False
                logger.warning("Neither pgvector extension nor custom vector functions available")
                
        except Exception as e:
            self.pgvector_available = False
            logger.warning(f"Error checking vector availability: {e}")
    
    def search_players_by_filters(self, filters: Dict, limit: int = 50, offset: int = 0) -> List[Dict]:
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                where_conditions = ["p.status = 'active'", "p.deleted_at IS NULL"]
                params = []
                
                # Position filter
                if filters.get('position') and filters['position'] != 'any':
                    # Check in player_teams table for position
                    where_conditions.append("""
                        EXISTS (
                            SELECT 1 FROM player_teams pt 
                            WHERE pt.player_id = p.id 
                            AND pt.position = %s
                        )
                    """)
                    params.append(filters['position'])
                
                # Age filters
                if filters.get('min_age'):
                    where_conditions.append("EXTRACT(YEAR FROM AGE(p.birth_date)) >= %s")
                    params.append(filters['min_age'])
                
                if filters.get('max_age'):
                    where_conditions.append("EXTRACT(YEAR FROM AGE(p.birth_date)) <= %s")
                    params.append(filters['max_age'])
                
                # Gender filter
                if filters.get('gender'):
                    where_conditions.append("p.gender = %s")
                    params.append(filters['gender'])
                
                # Skill level filters (check average skill level)
                if filters.get('min_skill') or filters.get('max_skill'):
                    skill_subquery = """
                        (SELECT AVG(ps.level) FROM player_skills ps WHERE ps.player_id = p.id)
                    """
                    if filters.get('min_skill'):
                        where_conditions.append(f"{skill_subquery} >= %s")
                        params.append(filters['min_skill'])
                    
                    if filters.get('max_skill'):
                        where_conditions.append(f"{skill_subquery} <= %s")
                        params.append(filters['max_skill'])
                
                # Location-based search (if lat/lng provided)
                if filters.get('latitude') and filters.get('longitude') and filters.get('max_distance_km'):
                    # Use earth distance calculation
                    where_conditions.append("""
                        earth_distance(
                            ll_to_earth(
                                CAST(p.location->>'lat' AS FLOAT), 
                                CAST(p.location->>'lng' AS FLOAT)
                            ),
                            ll_to_earth(%s, %s)
                        ) <= %s * 1000
                    """)
                    params.extend([filters['latitude'], filters['longitude'], filters['max_distance_km']])
                
                # Build the main query
                query = f"""
                    SELECT 
                        p.id,
                        p.first_name,
                        p.last_name,
                        p.location,
                        p.birth_date,
                        p.height,
                        p.weight,
                        p.gender,
                        p.status,
                        EXTRACT(YEAR FROM AGE(p.birth_date)) as age,
                        (SELECT AVG(ps.level) FROM player_skills ps WHERE ps.player_id = p.id) as avg_skill_level,
                        (SELECT array_agg(pt.position) FROM player_teams pt WHERE pt.player_id = p.id) as positions,
                        (SELECT json_object_agg(ps.skill, ps.level) FROM player_skills ps WHERE ps.player_id = p.id) as skills
                    FROM players p
                    WHERE {' AND '.join(where_conditions)}
                    ORDER BY p.created_at DESC
                    LIMIT %s OFFSET %s
                """
                
                params.extend([limit, offset])
                
                cur.execute(query, params)
                results = cur.fetchall()
                
                # Convert to list of dicts
                players = []
                for row in results:
                    player = dict(row)
                    # Parse JSON fields
                    if player['location']:
                        player['location'] = json.loads(player['location']) if isinstance(player['location'], str) else player['location']
                    if player['skills']:
                        player['skills'] = json.loads(player['skills']) if isinstance(player['skills'], str) else player['skills']
                    
                    players.append(player)
                
                logger.info(f"Found {len(players)} players matching filters")
                return players
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def vector_similarity_search(self, query_vector: np.ndarray, candidate_ids: List[str] = None, top_k: int = 50) -> List[Dict]:
        """Perform vector similarity search"""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                if self.pgvector_available:
                    # Check if we have pgvector extension or custom functions
                    cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
                    has_pgvector_ext = cur.fetchone() is not None
                    
                    query_vector_list = query_vector.tolist()
                    
                    if has_pgvector_ext:
                        # Use pgvector extension
                        query_vector_str = '[' + ','.join(map(str, query_vector)) + ']'
                        
                        if candidate_ids:
                            # Search within candidate set
                            placeholders = ','.join(['%s'] * len(candidate_ids))
                            query = f"""
                                SELECT 
                                    p.id, p.first_name, p.last_name, p.location, p.birth_date,
                                    pe.embedding <=> %s::vector as similarity_score
                                FROM players p
                                JOIN player_embeddings pe ON pe.player_id = p.id
                                WHERE p.id IN ({placeholders})
                                ORDER BY pe.embedding <=> %s::vector
                                LIMIT %s
                            """
                            params = [query_vector_str] + candidate_ids + [query_vector_str, top_k]
                        else:
                            # Search all players
                            query = """
                                SELECT 
                                    p.id, p.first_name, p.last_name, p.location, p.birth_date,
                                    pe.embedding <=> %s::vector as similarity_score
                                FROM players p
                                JOIN player_embeddings pe ON pe.player_id = p.id
                                WHERE p.status = 'active' AND p.deleted_at IS NULL
                                ORDER BY pe.embedding <=> %s::vector
                                LIMIT %s
                            """
                            params = [query_vector_str, query_vector_str, top_k]
                        
                        cur.execute(query, params)
                        
                    else:
                        # Use custom vector functions with JSONB embeddings
                        if candidate_ids:
                            placeholders = ','.join(['%s'] * len(candidate_ids))
                            query = f"""
                                SELECT 
                                    p.id, p.first_name, p.last_name, p.location, p.birth_date,
                                    vector_cosine_distance(%s, 
                                        ARRAY(SELECT jsonb_array_elements_text(pef.embedding)::float)
                                    ) as similarity_score
                                FROM players p
                                JOIN player_embeddings_fallback pef ON pef.player_id = p.id
                                WHERE p.id IN ({placeholders})
                                ORDER BY vector_cosine_distance(%s, 
                                    ARRAY(SELECT jsonb_array_elements_text(pef.embedding)::float)
                                )
                                LIMIT %s
                            """
                            params = [query_vector_list] + candidate_ids + [query_vector_list, top_k]
                        else:
                            query = """
                                SELECT 
                                    p.id, p.first_name, p.last_name, p.location, p.birth_date,
                                    vector_cosine_distance(%s, 
                                        ARRAY(SELECT jsonb_array_elements_text(pef.embedding)::float)
                                    ) as similarity_score
                                FROM players p
                                JOIN player_embeddings_fallback pef ON pef.player_id = p.id
                                WHERE p.status = 'active' AND p.deleted_at IS NULL
                                ORDER BY vector_cosine_distance(%s, 
                                    ARRAY(SELECT jsonb_array_elements_text(pef.embedding)::float)
                                )
                                LIMIT %s
                            """
                            params = [query_vector_list, query_vector_list, top_k]
                        
                        cur.execute(query, params)
                    
                else:
                    # Fallback: use JSONB embeddings with manual similarity calculation
                    if candidate_ids:
                        placeholders = ','.join(['%s'] * len(candidate_ids))
                        query = f"""
                            SELECT 
                                p.id, p.first_name, p.last_name, p.location, p.birth_date,
                                pef.embedding
                            FROM players p
                            JOIN player_embeddings_fallback pef ON pef.player_id = p.id
                            WHERE p.id IN ({placeholders})
                        """
                        params = candidate_ids
                    else:
                        query = """
                            SELECT 
                                p.id, p.first_name, p.last_name, p.location, p.birth_date,
                                pef.embedding
                            FROM players p
                            JOIN player_embeddings_fallback pef ON pef.player_id = p.id
                            WHERE p.status = 'active' AND p.deleted_at IS NULL
                        """
                        params = []
                    
                    cur.execute(query, params)
                    results = cur.fetchall()
                    
                    # Calculate similarities manually
                    scored_results = []
                    for row in results:
                        embedding = json.loads(row['embedding']) if isinstance(row['embedding'], str) else row['embedding']
                        embedding_array = np.array(embedding)
                        
                        # Cosine similarity
                        similarity = np.dot(query_vector, embedding_array) / (
                            np.linalg.norm(query_vector) * np.linalg.norm(embedding_array)
                        )
                        
                        result_dict = dict(row)
                        result_dict['similarity_score'] = 1 - similarity  # Convert to distance
                        scored_results.append(result_dict)
                    
                    # Sort by similarity and limit
                    scored_results.sort(key=lambda x: x['similarity_score'])
                    results = scored_results[:top_k]
                
                # Convert results to list of dicts
                players = []
                for row in results:
                    player = dict(row)
                    if 'embedding' in player:
                        del player['embedding']  # Remove embedding from results
                    players.append(player)
                
                logger.info(f"Vector similarity search returned {len(players)} results")
                return players
                
        except Exception as e:
            logger.error(f"Vector similarity search failed: {e}")
            return []
    
    def get_player_embedding(self, player_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific player"""
        try:
            with self.conn.cursor() as cur:
                if self.pgvector_available:
                    cur.execute("SELECT embedding FROM player_embeddings WHERE player_id = %s", (player_id,))
                else:
                    cur.execute("SELECT embedding FROM player_embeddings_fallback WHERE player_id = %s", (player_id,))
                
                result = cur.fetchone()
                if result:
                    if self.pgvector_available:
                        # pgvector returns as string, parse it
                        embedding_str = result[0]
                        # Parse vector string format: [1.0,2.0,3.0]
                        embedding_list = json.loads(embedding_str.replace('[', '[').replace(']', ']'))
                        return np.array(embedding_list)
                    else:
                        # JSONB format
                        embedding = json.loads(result[0]) if isinstance(result[0], str) else result[0]
                        return np.array(embedding)
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get embedding for player {player_id}: {e}")
            return None
    
    def store_player_embedding(self, player_id: str, embedding: np.ndarray):
        """Store embedding for a player"""
        try:
            with self.conn.cursor() as cur:
                if self.pgvector_available:
                    embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                    cur.execute("""
                        INSERT INTO player_embeddings (player_id, embedding, updated_at)
                        VALUES (%s, %s::vector, %s)
                        ON CONFLICT (player_id) DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            updated_at = EXCLUDED.updated_at
                    """, (player_id, embedding_str, datetime.now()))
                else:
                    embedding_json = json.dumps(embedding.tolist())
                    cur.execute("""
                        INSERT INTO player_embeddings_fallback (player_id, embedding, updated_at)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (player_id) DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            updated_at = EXCLUDED.updated_at
                    """, (player_id, embedding_json, datetime.now()))
                
                self.conn.commit()
                logger.debug(f"Stored embedding for player {player_id}")
                
        except Exception as e:
            logger.error(f"Failed to store embedding for player {player_id}: {e}")
            self.conn.rollback()