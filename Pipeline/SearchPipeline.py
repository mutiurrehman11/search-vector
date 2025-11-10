"""
Corrected SearchPipeline - Fixed undefined variables and missing methods
"""

import numpy as np
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
import time

from .EmbeddingGenerator import PlayerEmbeddingGenerator
from .Pipeline import PlayerSearchEngine, ExplainabilityEngine, RecommendationEngine
from .Pipeline import SavedSearchManager
from .MLReRanker import MLReRanker

logger = logging.getLogger(__name__)


class SearchPipeline:
    """Enhanced search pipeline with ML re-ranking and recommendations"""
    
    def __init__(self, db_config: Dict, model_path: str = "models/reranker.pkl"):
        self.db_config = db_config
        self.config = db_config
        
        self.search_engine = PlayerSearchEngine(db_config)
        if not self.search_engine.connect():
            logger.error("Failed to connect to the database.")
            raise ConnectionError("Database connection failed.")
        
        self.embedding_generator = PlayerEmbeddingGenerator()
        
        # Initialize ML components
        self.ml_reranker = MLReRanker()
        try:
            self.ml_reranker.load_model("models/reranker.pkl")
            logger.info("Successfully loaded ML re-ranking model.")
        except FileNotFoundError:
            logger.warning("Reranker model not found. Proceeding without ML re-ranking.")
        except Exception as e:
            logger.error(f"Error loading reranker model: {e}")
        
        self.recommendation_engine = RecommendationEngine(self.search_engine)
        self.saved_search_mgr = SavedSearchManager(self.search_engine)

        self.max_candidates = 1000
        self.default_limit = 20
        
        logger.info("SearchPipeline initialized")

    def train_ml_model(self) -> Dict:
        """Train the ML re-ranking model using engagement data"""
        start_time = time.time()
        
        try:
            logger.info("Starting ML model training...")
            
            success = self.ml_reranker.train(self.search_engine.conn)
            
            if success:
                import os
                os.makedirs("models", exist_ok=True)
                self.ml_reranker.save_model("models/reranker.pkl")
                
                total_time = time.time() - start_time
                logger.info(f"ML model training completed successfully in {total_time:.2f}s")
                
                return {
                    'success': True,
                    'message': 'ML model trained and saved successfully',
                    'metadata': {
                        'training_time_s': total_time,
                        'model_path': 'models/reranker.pkl'
                    }
                }
            else:
                return {
                    'success': False,
                    'error': 'Model training failed - insufficient data or other error',
                    'metadata': {
                        'training_time_s': time.time() - start_time
                    }
                }
                
        except Exception as e:
            logger.error(f"ML model training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'metadata': {
                    'training_time_s': time.time() - start_time
                }
            }

    def search_players(self, filters: Dict, limit: int = None, offset: int = 0) -> Dict:
        """Enhanced search with ML re-ranking"""
        start_time = time.time()
        limit = limit or self.default_limit
        
        try:
            # Stage 1: Filter-based candidate retrieval
            logger.info("Stage 1: Filter-based candidate retrieval")
            filter_start = time.time()
            
            candidate_ids = self.search_engine.strict_filter_stage(filters)
            
            if not candidate_ids:
                return {
                    'results': [],
                    'total_count': 0,
                    'metadata': {
                        'search_type': 'filter_only',
                        'candidates_found': 0,
                        'reranked': False,
                        'ml_reranked': False
                    },
                    'telemetry': {
                        'total_time_ms': (time.time() - start_time) * 1000,
                        'filter_time_ms': (time.time() - filter_start) * 1000,
                        'vector_time_ms': 0,
                        'rerank_time_ms': 0
                    }
                }

            candidates = self.search_engine.get_players_by_ids(candidate_ids)
            filter_time = time.time() - filter_start
            logger.info(f"Filter search returned {len(candidates)} candidates in {filter_time:.3f}s")
            
            vector_time = 0
            rerank_time = 0
            reranked = False
            ml_reranked = False
            
            # Stage 2: Vector similarity (if beneficial)
            if len(candidates) > limit and self._should_use_vector_search(filters):
                logger.info("Stage 2: Vector similarity reranking")
                vector_start = time.time()
                
                query_vector = self.search_engine.query_builder.build_from_filters(filters)
                top_k_vector = int(self.config.get('vector_search_top_k', 100))
                
                # Get vector similarity results
                vector_results = self.search_engine.vector_similarity_stage(
                    query_vector,
                    [c['id'] for c in candidates],
                    top_k=min(len(candidates), top_k_vector * 2)
                )
                
                if vector_results:
                    # Merge filter and vector results
                    candidates = self._merge_filter_and_vector_results(candidates, vector_results)
                    reranked = True
                
                vector_time = time.time() - vector_start
                logger.info(f"Vector reranking completed in {vector_time:.3f}s")
            
            # Stage 3: ML Re-ranking (if model available)
            if self.ml_reranker.model and candidates and len(candidates) > 1:
                logger.info("Stage 3: ML re-ranking")
                rerank_start = time.time()
                
                query_context = {
                    'min_skill': filters.get('min_skill'),
                    'max_skill': filters.get('max_skill'),
                    'tags': filters.get('tags', []),
                    'location': filters.get('location'),
                    'positions': filters.get('positions', [])
                }
                
                candidates = self.ml_reranker.rerank(candidates, query_context)
                ml_reranked = True
                
                rerank_time = time.time() - rerank_start
                logger.info(f"ML re-ranking completed in {rerank_time:.3f}s")
            
            # Final result preparation
            final_results = candidates[offset:offset + limit]
            
            # Add computed fields
            for result in final_results:
                try:
                    result['match_score'] = self._calculate_match_score(result, filters)
                    result['distance_km'] = self._calculate_distance(result, filters)
                except Exception as e:
                    logger.error(f"Error calculating fields for player {result.get('id', 'unknown')}: {e}")
                    result['match_score'] = 0.5
                    result['distance_km'] = None
            
            total_time = time.time() - start_time
            
            return {
                'results': final_results,
                'total_count': len(candidates),
                'metadata': {
                    'search_type': 'ml_reranked' if ml_reranked else ('hybrid' if reranked else 'filter_only'),
                    'candidates_found': len(candidates),
                    'reranked': reranked,
                    'ml_reranked': ml_reranked,
                    'pgvector_available': self.search_engine.pgvector_available
                },
                'telemetry': {
                    'total_time_ms': total_time * 1000,
                    'filter_time_ms': filter_time * 1000,
                    'vector_time_ms': vector_time * 1000,
                    'rerank_time_ms': rerank_time * 1000
                }
            }
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'results': [],
                'total_count': 0,
                'error': str(e),
                'metadata': {
                    'search_type': 'failed',
                    'candidates_found': 0,
                    'reranked': False
                },
                'telemetry': {
                    'total_time_ms': (time.time() - start_time) * 1000,
                    'filter_time_ms': 0,
                    'vector_time_ms': 0,
                    'rerank_time_ms': 0
                }
            }
    
    def get_recommendations(self, player_id: str, limit: int = 20) -> Dict:
        """Get player recommendations using similarity"""
        start_time = time.time()
        
        try:
            logger.info(f"Getting recommendations for player {player_id}")
            recommendations = self.recommendation_engine.more_like_this(player_id, limit)
            
            if not recommendations:
                logger.info(f"No recommendations found for player {player_id}")
                return {
                    'results': [],
                    'total_count': 0,
                    'metadata': {
                        'recommendation_type': 'similarity_based',
                        'reason': 'no_similar_players_found'
                    }
                }
            
            # Enrich with full player details
            player_ids = [r['id'] for r in recommendations]
            enriched = self.search_engine.get_players_by_ids(player_ids)
            
            # Merge similarity scores
            similarity_map = {r['id']: r.get('similarity', 0) for r in recommendations}
            for player in enriched:
                player['similarity_score'] = similarity_map.get(player['id'], 0)
                player['recommendation_reason'] = 'Similar playing style and attributes'
            
            total_time = time.time() - start_time
            
            return {
                'results': enriched,
                'total_count': len(enriched),
                'metadata': {
                    'recommendation_type': 'similarity_based',
                    'target_player_id': player_id,
                    'pgvector_available': self.search_engine.pgvector_available
                },
                'telemetry': {
                    'total_time_ms': total_time * 1000
                }
            }
            
        except Exception as e:
            logger.error(f"Recommendations failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'results': [],
                'total_count': 0,
                'error': str(e),
                'metadata': {'recommendation_type': 'failed'}
            }
    
    def generate_and_store_embeddings(self, batch_size: int = 100) -> Dict:
        """Generate embeddings for all players and store them"""
        start_time = time.time()
        
        try:
            players = self._get_players_needing_embeddings(batch_size)
            
            if not players:
                logger.info("No players need embedding updates")
                return {
                    'processed': 0,
                    'success': 0,
                    'errors': 0,
                    'message': 'No players need embedding updates'
                }
            
            processed = 0
            success = 0
            errors = 0
            
            for player in players:
                try:
                    embedding = self.embedding_generator.generate_embedding(player)
                    self.search_engine.store_player_embedding(player['id'], embedding)
                    
                    success += 1
                    processed += 1
                    
                    if processed % 10 == 0:
                        logger.info(f"Processed {processed}/{len(players)} embeddings")
                        
                except Exception as e:
                    logger.error(f"Failed to process player {player.get('id', 'unknown')}: {e}")
                    errors += 1
                    processed += 1
            
            total_time = time.time() - start_time
            
            logger.info(f"Embedding generation completed: {success} success, {errors} errors in {total_time:.2f}s")
            
            return {
                'processed': processed,
                'success': success,
                'errors': errors,
                'time_seconds': total_time,
                'message': f'Successfully processed {success}/{processed} players'
            }
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return {
                'processed': 0,
                'success': 0,
                'errors': 1,
                'error': str(e)
            }
    
    def _should_use_vector_search(self, filters: Dict) -> bool:
        """Determine if vector search should be used"""
        vector_beneficial_filters = [
            'position', 'min_skill', 'max_skill', 'latitude', 'longitude'
        ]
        
        return any(filters.get(f) for f in vector_beneficial_filters)
    
    def _merge_filter_and_vector_results(self, filter_results: List[Dict], vector_results: List[Dict]) -> List[Dict]:
        """Merge filter-based and vector similarity results"""
        # Create lookup for vector scores
        vector_scores = {r['id']: r.get('similarity_score', 0) for r in vector_results}
        
        # Add similarity scores to filter results
        for result in filter_results:
            result['similarity_score'] = vector_scores.get(result['id'], 0.0)
        
        # Sort by similarity score
        filter_results.sort(key=lambda x: x.get('similarity_score', 0.0), reverse=True)
        
        return filter_results
    
    def _calculate_match_score(self, player: Dict, filters: Dict) -> float:
        """Calculate match score for the player"""
        score = 0.0
        factors = 0
        
        # Position match
        if filters.get('position') and filters['position'] != 'any':
            positions = player.get('positions', [])
            if positions and filters['position'] in positions:
                score += 1.0
            factors += 1
        
        # Skill level match
        avg_skill = float(player.get('avg_skill_level', 0))
        min_skill = filters.get('min_skill', 0)
        max_skill = filters.get('max_skill', 100)
        
        if min_skill <= avg_skill <= max_skill:
            skill_range = max_skill - min_skill
            if skill_range > 0:
                center = (min_skill + max_skill) / 2
                distance_from_center = abs(avg_skill - center)
                skill_score = 1.0 - (distance_from_center / (skill_range / 2))
                score += max(skill_score, 0.5)
            else:
                score += 1.0
        factors += 1
        
        # Age match
        age = int(player.get('age', 0))
        min_age = filters.get('min_age', 0)
        max_age = filters.get('max_age', 100)
        
        if min_age <= age <= max_age:
            score += 1.0
        factors += 1
        
        return score / factors if factors > 0 else 0.5
    
    def _calculate_distance(self, player: Dict, filters: Dict) -> Optional[float]:
        """Calculate distance between player and search location"""
        if not (filters.get('latitude') and filters.get('longitude')):
            return None
        
        player_location = player.get('location')
        if not player_location:
            return None
        
        # Parse location if it's a string
        if isinstance(player_location, str):
            try:
                player_location = json.loads(player_location)
            except:
                return None
        
        if not (player_location.get('latitude') and player_location.get('longitude')):
            return None
        
        from math import radians, cos, sin, asin, sqrt
        
        lat1, lon1 = radians(float(filters['latitude'])), radians(float(filters['longitude']))
        lat2, lon2 = radians(float(player_location['latitude'])), radians(float(player_location['longitude']))
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371
        
        return c * r
    
    def _get_players_needing_embeddings(self, limit: int) -> List[Dict]:
        """Get players that need embedding generation/updates"""
        try:
            with self.search_engine.conn.cursor() as cur:
                query = """
                    SELECT 
                        p.id, p.first_name, p.last_name, p.location, p.birth_date,
                        p.height, p.weight, p.gender, p.status,
                        EXTRACT(YEAR FROM AGE(p.birth_date)) as age,
                        (SELECT AVG(ps.level) FROM player_skills ps WHERE ps.player_id = p.id) as avg_skill_level,
                        (SELECT array_agg(pt.position) FROM player_teams pt WHERE pt.player_id = p.id AND pt.end_at IS NULL) as positions,
                        (SELECT json_object_agg(ps.skill, ps.level) FROM player_skills ps WHERE ps.player_id = p.id) as skills
                    FROM players p
                    WHERE p.status = 'active' 
                    AND p.deleted_at IS NULL
                    AND p.embedding IS NULL
                    ORDER BY p.created_at DESC
                    LIMIT %s
                """
                
                cur.execute(query, (limit,))
                results = cur.fetchall()
                
                players = []
                for row in results:
                    location_json = row[3]
                    if isinstance(location_json, str):
                        try:
                            location_json = json.loads(location_json)
                        except:
                            location_json = {}
                    
                    player = {
                        'id': row[0],
                        'first_name': row[1],
                        'last_name': row[2],
                        'location': location_json,
                        'birth_date': row[4],
                        'height': row[5],
                        'weight': row[6],
                        'gender': row[7],
                        'status': row[8],
                        'age': int(row[9]) if row[9] is not None else None,
                        'avg_skill_level': float(row[10]) if row[10] is not None else 50.0,
                        'positions': row[11] or [],
                        'skills': row[12] if row[12] else {}
                    }
                    players.append(player)
                
                return players
                
        except Exception as e:
            logger.error(f"Failed to get players needing embeddings: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def log_interaction(self, user_id, player_id, event_type, query_context):
        """Log user interactions"""
        try:
            with self.search_engine.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO player_engagement_events 
                    (user_id, player_id, event_type, query_context, created_at)
                    VALUES (%s, %s, %s, %s, NOW());
                """, (user_id, player_id, event_type, json.dumps(query_context)))
                self.search_engine.conn.commit()
                logger.info(f"Logged {event_type} event for user {user_id}, player {player_id}")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
