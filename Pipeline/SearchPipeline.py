import numpy as np
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time

from .PipelineCompat import CompatPlayerSearchEngine
from .EmbeddingGenerator import CompatPlayerEmbeddingGenerator
from .QueryBuilder import QueryVectorBuilder
from .MLReRanker import MLReRanker
from .RecommendationEngine import RecommendationEngine

logger = logging.getLogger(__name__)


class SearchPipeline:
    """Enhanced search pipeline with ML re-ranking and recommendations"""
    
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        
        self.search_engine = CompatPlayerSearchEngine(db_config)
        self.embedding_generator = CompatPlayerEmbeddingGenerator()
        self.query_builder = QueryVectorBuilder()
        
        # Initialize ML components
        self.ml_reranker = MLReRanker(model_path="models/reranker.pkl")
        self.recommendation_engine = RecommendationEngine()

        self.max_candidates = 1000
        self.default_limit = 20
        
        # Build recommendation index on startup
        self._initialize_ml_components()
        
        logger.info("SearchPipeline initialized")

    def _initialize_ml_components(self):
        """Initialize ML components with existing data"""
        try:
            # Build recommendation index
            self.recommendation_engine.build_player_index(self.search_engine.conn)
            logger.info("ML components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ML components: {e}")
            # Continue without ML features if initialization fails

    def train_ml_model(self) -> Dict:
        """Train the ML re-ranking model using engagement data"""
        start_time = time.time()
        
        try:
            logger.info("Starting ML model training...")
            
            # Train the re-ranking model
            success = self.ml_reranker.train(self.search_engine.conn)
            
            if success:
                # Save the trained model
                import os
                os.makedirs("models", exist_ok=True)
                self.ml_reranker.save_model("models/reranker.pkl")
                
                # Refresh recommendation index
                self.recommendation_engine.refresh_index(self.search_engine.conn)
                
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
            
            candidates = self.search_engine.search_players_by_filters(
                filters, 
                limit=min(self.max_candidates, limit * 10),
                offset=offset
            )
            
            filter_time = time.time() - filter_start
            logger.info(f"Filter search returned {len(candidates)} candidates in {filter_time:.3f}s")
            
            if not candidates:
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
                        'filter_time_ms': filter_time * 1000,
                        'vector_time_ms': 0,
                        'rerank_time_ms': 0
                    }
                }
            vector_time = 0
            rerank_time = 0
            reranked = False
            ml_reranked = False
            
            if len(candidates) > limit and self._should_use_vector_search(filters):
                logger.info("Stage 2: Vector similarity reranking")
                vector_start = time.time()
                
                query_vector = self.query_builder.build_query_vector(filters)
                
                # Get candidate IDs
                candidate_ids = [c['id'] for c in candidates]
                
                # Perform vector similarity search on candidates
                vector_results = self.search_engine.vector_similarity_search(
                    query_vector, 
                    candidate_ids=candidate_ids,
                    top_k=limit * 2
                )
                
                if vector_results:
                    # Merge filter results with vector scores
                    candidates = self._merge_filter_and_vector_results(candidates, vector_results)
                    reranked = True
                
                vector_time = time.time() - vector_start
                logger.info(f"Vector reranking completed in {vector_time:.3f}s")
                
                # Stage 3: ML Re-ranking (if model is available)
                if self.ml_reranker.model and len(candidates) > 1:
                    logger.info("Stage 3: ML re-ranking")
                    rerank_start = time.time()
                    
                    # Prepare query context for ML features
                    query_context = {
                        'min_skill': filters.get('min_skill'),
                        'max_skill': filters.get('max_skill'),
                        'tags': filters.get('tags', []),
                        'location': filters.get('location'),
                        'positions': filters.get('positions', [])
                    }
                    
                    # Apply ML re-ranking
                    candidates = self.ml_reranker.rerank(candidates, query_context)
                    ml_reranked = True
                    
                    rerank_time = time.time() - rerank_start
                    logger.info(f"ML re-ranking completed in {rerank_time:.3f}s")
            
            # Final result preparation
            final_results = candidates[:limit]
            
            # Add computed fields
            for result in final_results:
                try:
                    result['match_score'] = self._calculate_match_score(result, filters)
                    result['distance_km'] = self._calculate_distance(result, filters)
                except Exception as e:
                    logger.error(f"Error calculating fields for player {result.get('id', 'unknown')}: {e}")
                    logger.error(f"Player location: {result.get('location', {})}")
                    logger.error(f"Filters: {filters}")
                    result['match_score'] = 0.5
                    result['distance_km'] = None
            
            total_time = time.time() - start_time
            
            return {
                'results': final_results,
                'total_count': len(candidates),
                'metadata': {
                    'search_type': 'hybrid' if reranked else 'filter_only',
                    'candidates_found': len(candidates),
                    'reranked': reranked,
                    'ml_reranked': ml_reranked,
                    'pgvector_available': self.search_engine._check_pgvector_availability()
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
        """Get player recommendations using ML-based similarity"""
        start_time = time.time()
        
        try:
            # Use ML recommendation engine if available
            if self.recommendation_engine.is_fitted:
                logger.info(f"Getting ML-based recommendations for player {player_id}")
                recommendations = self.recommendation_engine.get_similar_players(player_id, limit)
                
                if recommendations:
                    # Enrich with player details
                    enriched_recommendations = []
                    for rec in recommendations:
                        player_details = self.search_engine.get_player_by_id(rec['player_id'])
                        if player_details:
                            player_details.update({
                                'similarity_score': rec['similarity_score'],
                                'recommendation_reason': f'Similar profile (rank #{rec["rank"]})',
                                'rank': rec['rank']
                            })
                            enriched_recommendations.append(player_details)
                    
                    total_time = time.time() - start_time
                    return {
                        'results': enriched_recommendations,
                        'total_count': len(enriched_recommendations),
                        'metadata': {
                            'recommendation_type': 'ml_similarity',
                            'algorithm': 'k_nearest_neighbors'
                        },
                        'telemetry': {
                            'total_time_ms': total_time * 1000
                        }
                    }
            
            # Fallback to vector similarity if ML engine not available
            logger.info(f"Using vector similarity fallback for player {player_id}")
            target_embedding = self.search_engine.get_player_embedding(player_id)
            
            if target_embedding is None:
                logger.warning(f"No embedding found for player {player_id}")
                return {
                    'results': [],
                    'total_count': 0,
                    'error': 'Player embedding not found',
                    'metadata': {'recommendation_type': 'failed'}
                }
            
            # Perform vector similarity search
            similar_players = self.search_engine.vector_similarity_search(
                target_embedding,
                top_k=limit + 1  # +1 to exclude the target player
            )
            
            # Remove the target player from results
            recommendations = [p for p in similar_players if p['id'] != player_id][:limit]
            
            # Add recommendation scores
            for i, rec in enumerate(recommendations):
                rec['similarity_score'] = rec.get('similarity_score', 0)
                rec['recommendation_reason'] = 'Similar playing style and attributes'
                rec['rank'] = i + 1
            
            total_time = time.time() - start_time
            
            return {
                'results': recommendations,
                'total_count': len(recommendations),
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
            return {
                'results': [],
                'total_count': 0,
                'error': str(e),
                'metadata': {'recommendation_type': 'failed'}
            }
    
    def get_personalized_recommendations(self, user_id: str, limit: int = 20) -> Dict:
        """Get personalized recommendations based on user's interaction history"""
        start_time = time.time()
        
        try:
            if not self.recommendation_engine.is_fitted:
                logger.warning("Recommendation engine not available")
                return {
                    'results': [],
                    'total_count': 0,
                    'error': 'Recommendation engine not initialized',
                    'metadata': {'recommendation_type': 'failed'}
                }
            
            logger.info(f"Getting personalized recommendations for user {user_id}")
            recommendations = self.recommendation_engine.get_recommendations_for_user(
                self.search_engine.conn, user_id, limit
            )
            
            if not recommendations:
                logger.info(f"No personalized recommendations found for user {user_id}")
                return {
                    'results': [],
                    'total_count': 0,
                    'metadata': {
                        'recommendation_type': 'personalized',
                        'reason': 'no_interaction_history'
                    }
                }
            
            # Enrich with player details
            enriched_recommendations = []
            for rec in recommendations:
                player_details = self.search_engine.get_player_by_id(rec['player_id'])
                if player_details:
                    player_details.update({
                        'similarity_score': rec['similarity_score'],
                        'recommendation_reason': rec.get('reason', 'Based on your activity'),
                        'rank': rec['rank'],
                        'recommendation_count': rec.get('recommendation_count', 1)
                    })
                    enriched_recommendations.append(player_details)
            
            total_time = time.time() - start_time
            return {
                'results': enriched_recommendations,
                'total_count': len(enriched_recommendations),
                'metadata': {
                    'recommendation_type': 'personalized',
                    'algorithm': 'collaborative_filtering'
                },
                'telemetry': {
                    'total_time_ms': total_time * 1000
                }
            }
            
        except Exception as e:
            logger.error(f"Personalized recommendation failed: {e}")
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
            # Get all players without embeddings or with outdated embeddings
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
                    # Generate embedding
                    embedding = self.embedding_generator.generate_embedding(player)
                    
                    # Store embedding
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
        """Determine if vector search should be used based on filters"""
        # Use vector search if we have specific preferences that benefit from similarity
        vector_beneficial_filters = [
            'position', 'min_skill', 'max_skill', 'latitude', 'longitude'
        ]
        
        return any(filters.get(f) for f in vector_beneficial_filters)
    
    def _merge_filter_and_vector_results(self, filter_results: List[Dict], vector_results: List[Dict]) -> List[Dict]:
        """Merge filter-based and vector similarity results"""
        # Create lookup for vector scores
        vector_scores = {r['id']: r.get('similarity_score', 1.0) for r in vector_results}
        
        # Add vector scores to filter results and sort by similarity
        for result in filter_results:
            result['similarity_score'] = vector_scores.get(result['id'], 1.0)
        
        # Sort by similarity score (lower is better for distance-based similarity)
        filter_results.sort(key=lambda x: x.get('similarity_score', 1.0))
        
        return filter_results
    
    def _calculate_match_score(self, player: Dict, filters: Dict) -> float:
        """Calculate a match score for the player based on filters"""
        score = 0.0
        factors = 0
        
        # Position match
        if filters.get('position') and filters['position'] != 'any':
            positions = player.get('positions', [])
            if positions and filters['position'] in positions:
                score += 1.0
            factors += 1
        
        # Skill level match
        avg_skill = player.get('avg_skill_level', 0)
        min_skill = filters.get('min_skill', 0)
        max_skill = filters.get('max_skill', 100)
        
        if min_skill <= avg_skill <= max_skill:
            # Perfect match gets full score
            skill_range = max_skill - min_skill
            if skill_range > 0:
                # Closer to center of range gets higher score
                center = (min_skill + max_skill) / 2
                distance_from_center = abs(avg_skill - center)
                skill_score = 1.0 - (distance_from_center / (skill_range / 2))
                score += max(skill_score, 0.5)  # Minimum 0.5 for being in range
            else:
                score += 1.0
        factors += 1
        
        # Age match
        age = player.get('age', 0)
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
        
        player_location = player.get('location', {})
        if not (player_location.get('latitude') and player_location.get('longitude')):
            return None
        
        # Simple haversine distance calculation
        from math import radians, cos, sin, asin, sqrt
        
        # Convert to float to handle Decimal types from database
        lat1, lon1 = radians(float(filters['latitude'])), radians(float(filters['longitude']))
        lat2, lon2 = radians(float(player_location['latitude'])), radians(float(player_location['longitude']))
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Earth's radius in kilometers
        
        return c * r
    
    def _get_players_needing_embeddings(self, limit: int) -> List[Dict]:
        """Get players that need embedding generation/updates"""
        try:
            with self.search_engine.conn.cursor() as cur:
                # Get players without embeddings or with old embeddings
                if self.search_engine.pgvector_available:
                    query = """
                        SELECT 
                            p.id, p.first_name, p.last_name, p.location, p.birth_date,
                            p.height, p.weight, p.gender, p.status,
                            EXTRACT(YEAR FROM AGE(p.birth_date)) as age,
                            (SELECT AVG(ps.level) FROM player_skills ps WHERE ps.player_id = p.id) as avg_skill_level,
                            (SELECT array_agg(pt.position) FROM player_teams pt WHERE pt.player_id = p.id) as positions,
                            (SELECT json_object_agg(ps.skill, ps.level) FROM player_skills ps WHERE ps.player_id = p.id) as skills
                        FROM players p
                        LEFT JOIN player_embeddings pe ON pe.player_id = p.id
                        WHERE p.status = 'active' AND p.deleted_at IS NULL
                        AND (pe.player_id IS NULL OR pe.updated_at < p.updated_at)
                        ORDER BY p.created_at DESC
                        LIMIT %s
                    """
                else:
                    query = """
                        SELECT 
                            p.id, p.first_name, p.last_name, p.location, p.birth_date,
                            p.height, p.weight, p.gender, p.status,
                            EXTRACT(YEAR FROM AGE(p.birth_date)) as age,
                            (SELECT AVG(ps.level) FROM player_skills ps WHERE ps.player_id = p.id) as avg_skill_level,
                            (SELECT array_agg(pt.position) FROM player_teams pt WHERE pt.player_id = p.id) as positions,
                            (SELECT json_object_agg(ps.skill, ps.level) FROM player_skills ps WHERE ps.player_id = p.id) as skills
                        FROM players p
                        LEFT JOIN player_embeddings_fallback pef ON pef.player_id = p.id
                        WHERE p.status = 'active' AND p.deleted_at IS NULL
                        AND (pef.player_id IS NULL OR pef.updated_at < p.updated_at)
                        ORDER BY p.created_at DESC
                        LIMIT %s
                    """
                
                cur.execute(query, (limit,))
                results = cur.fetchall()
                
                # Convert to list of dicts
                players = []
                for row in results:
                    player = {
                        'id': row[0],
                        'first_name': row[1],
                        'last_name': row[2],
                        'location': json.loads(row[3]) if row[3] else {},
                        'birth_date': row[4],
                        'height': row[5],
                        'weight': row[6],
                        'gender': row[7],
                        'status': row[8],
                        'age': row[9],
                        'avg_skill_level': row[10] or 50,
                        'positions': row[11] or [],
                        'skills': json.loads(row[12]) if row[12] else {}
                    }
                    players.append(player)
                
                return players
                
        except Exception as e:
            logger.error(f"Failed to get players needing embeddings: {e}")
            return []
    
    def _log_event(self, user_id: int, player_id: str, event_type: str, query_context: Dict):
        """Log engagement event for model training"""
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

    def log_interaction(self, user_id: int, player_id: str, event_type: str, query_context: Dict):
        """Public method to log user interactions"""
        self._log_event(user_id, player_id, event_type, query_context)