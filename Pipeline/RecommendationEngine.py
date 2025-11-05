import numpy as np
import json
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """k-NN based recommendation engine for "more like this" functionality"""

    def __init__(self):
        self.knn_model = NearestNeighbors(
            n_neighbors=20,
            metric='cosine',
            algorithm='brute'
        )
        self.scaler = StandardScaler()
        self.player_features = {}
        self.player_embeddings = {}
        self.is_fitted = False

    def extract_player_features(self, player: Dict) -> List[float]:
        """Extract numerical features from player profile"""
        features = []
        
        # Age (normalized to 0-1, assuming age range 16-40)
        age = player.get('age', 25)
        normalized_age = max(0, min(1, (age - 16) / 24))
        features.append(normalized_age)
        
        # Skills (average and individual skill levels)
        skills = player.get('skills', {})
        if skills:
            skill_values = list(skills.values())
            avg_skill = np.mean(skill_values)
            max_skill = max(skill_values)
            min_skill = min(skill_values)
            skill_variance = np.var(skill_values)
            
            features.extend([
                avg_skill / 10.0,  # Normalize to 0-1
                max_skill / 10.0,
                min_skill / 10.0,
                skill_variance / 25.0  # Normalize variance
            ])
        else:
            features.extend([0.5, 0.5, 0.5, 0.0])  # Default neutral values
        
        # Position diversity (number of positions)
        positions = player.get('positions', [])
        position_diversity = min(1.0, len(positions) / 5.0)  # Normalize to 0-1
        features.append(position_diversity)
        
        # Profile completeness
        completeness_factors = [
            bool(player.get('first_name')),
            bool(player.get('last_name')),
            bool(player.get('skills')),
            bool(player.get('location')),
            bool(player.get('positions')),
            player.get('age', 0) > 0,
        ]
        completeness = sum(completeness_factors) / len(completeness_factors)
        features.append(completeness)
        
        # Location features (if available)
        location = player.get('location', {})
        if location and 'coordinates' in location:
            # Use normalized coordinates as features
            lat = location['coordinates'].get('lat', 0) / 90.0  # Normalize latitude
            lng = location['coordinates'].get('lng', 0) / 180.0  # Normalize longitude
            features.extend([lat, lng])
        else:
            features.extend([0.0, 0.0])  # Default location
        
        return features

    def build_player_index(self, conn):
        """Build k-NN index from all players in database"""
        logger.info("Building player recommendation index...")
        
        try:
            with conn.cursor() as cur:
                # Get all players with their data
                cur.execute("""
                    SELECT 
                        p.id,
                        p.first_name,
                        p.last_name,
                        p.location,
                        p.birth_date,
                        EXTRACT(YEAR FROM AGE(p.birth_date)) as age,
                        (SELECT json_object_agg(ps.skill, ps.level) 
                         FROM player_skills ps WHERE ps.player_id = p.id) as skills,
                        (SELECT array_agg(pt.position) 
                         FROM player_teams pt WHERE pt.player_id = p.id) as positions,
                        p.embedding
                    FROM players p
                    WHERE p.embedding IS NOT NULL
                    ORDER BY p.id
                """)
                
                rows = cur.fetchall()
                
                if not rows:
                    logger.warning("No players with embeddings found")
                    return False
                
                player_ids = []
                features_list = []
                embeddings_list = []
                
                for row in rows:
                    player_id = row[0]
                    
                    # Create player dict
                    player = {
                        'id': player_id,
                        'first_name': row[1],
                        'last_name': row[2],
                        'location': json.loads(row[3]) if row[3] else {},
                        'age': row[5],
                        'skills': json.loads(row[6]) if row[6] else {},
                        'positions': row[7] or []
                    }
                    
                    # Extract features
                    features = self.extract_player_features(player)
                    
                    # Get embedding
                    embedding = row[8]
                    if embedding:
                        embedding_array = np.array(embedding)
                    else:
                        continue  # Skip players without embeddings
                    
                    player_ids.append(player_id)
                    features_list.append(features)
                    embeddings_list.append(embedding_array)
                    
                    # Store for later use
                    self.player_features[player_id] = features
                    self.player_embeddings[player_id] = embedding_array
                
                if not features_list:
                    logger.warning("No valid player features extracted")
                    return False
                
                # Combine features and embeddings
                features_array = np.array(features_list)
                embeddings_array = np.array(embeddings_list)
                
                # Standardize features
                features_scaled = self.scaler.fit_transform(features_array)
                
                # Combine scaled features with embeddings (weighted)
                feature_weight = 0.3
                embedding_weight = 0.7
                
                combined_features = np.hstack([
                    features_scaled * feature_weight,
                    embeddings_array * embedding_weight
                ])
                
                # Fit k-NN model
                self.knn_model.fit(combined_features)
                self.player_ids = player_ids
                self.combined_features = combined_features
                self.is_fitted = True
                
                logger.info(f"Built recommendation index for {len(player_ids)} players")
                return True
                
        except Exception as e:
            logger.error(f"Error building player index: {e}")
            return False

    def get_similar_players(self, player_id: str, n_recommendations: int = 10) -> List[Dict]:
        """Get players similar to the given player"""
        if not self.is_fitted:
            logger.warning("Recommendation index not built yet")
            return []
        
        if player_id not in self.player_features:
            logger.warning(f"Player {player_id} not found in index")
            return []
        
        try:
            # Get player's combined features
            player_idx = self.player_ids.index(player_id)
            player_features = self.combined_features[player_idx].reshape(1, -1)
            
            # Find similar players
            distances, indices = self.knn_model.kneighbors(
                player_features,
                n_neighbors=n_recommendations + 1  # +1 to exclude self
            )
            
            recommendations = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == player_idx:  # Skip self
                    continue
                
                similar_player_id = self.player_ids[idx]
                similarity_score = 1 - distance  # Convert distance to similarity
                
                recommendations.append({
                    'player_id': similar_player_id,
                    'similarity_score': float(similarity_score),
                    'rank': len(recommendations) + 1
                })
                
                if len(recommendations) >= n_recommendations:
                    break
            
            logger.info(f"Found {len(recommendations)} similar players for {player_id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting similar players: {e}")
            return []

    def get_recommendations_for_user(self, conn, user_id: str, n_recommendations: int = 10) -> List[Dict]:
        """Get personalized recommendations based on user's interaction history"""
        if not self.is_fitted:
            logger.warning("Recommendation index not built yet")
            return []
        
        try:
            with conn.cursor() as cur:
                # Get user's interaction history
                cur.execute("""
                    SELECT 
                        pee.player_id,
                        pee.event_type,
                        COUNT(*) as interaction_count,
                        MAX(pee.created_at) as last_interaction
                    FROM player_engagement_events pee
                    WHERE pee.user_id = %s
                        AND pee.created_at > NOW() - INTERVAL '30 days'
                    GROUP BY pee.player_id, pee.event_type
                    ORDER BY last_interaction DESC
                """, (user_id,))
                
                interactions = cur.fetchall()
                
                if not interactions:
                    # No interaction history, return popular players
                    return self._get_popular_players(conn, n_recommendations)
                
                # Weight interactions by type and recency
                interaction_weights = {
                    'impression': 1,
                    'profile_view': 2,
                    'follow': 4,
                    'save_to_playlist': 6,
                    'message': 8
                }
                
                player_scores = {}
                for row in interactions:
                    player_id = row[0]
                    event_type = row[1]
                    count = row[2]
                    
                    weight = interaction_weights.get(event_type, 1)
                    score = weight * count
                    
                    if player_id in player_scores:
                        player_scores[player_id] += score
                    else:
                        player_scores[player_id] = score
                
                # Get top interacted players
                top_players = sorted(
                    player_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]  # Top 5 most interacted players
                
                # Get similar players for each top player
                all_recommendations = {}
                for player_id, _ in top_players:
                    similar_players = self.get_similar_players(player_id, n_recommendations)
                    
                    for rec in similar_players:
                        rec_player_id = rec['player_id']
                        
                        # Skip if user already interacted with this player
                        if rec_player_id in player_scores:
                            continue
                        
                        if rec_player_id in all_recommendations:
                            # Boost score if recommended by multiple players
                            all_recommendations[rec_player_id]['similarity_score'] += rec['similarity_score'] * 0.5
                            all_recommendations[rec_player_id]['recommendation_count'] += 1
                        else:
                            all_recommendations[rec_player_id] = {
                                'player_id': rec_player_id,
                                'similarity_score': rec['similarity_score'],
                                'recommendation_count': 1
                            }
                
                # Sort by combined score
                final_recommendations = sorted(
                    all_recommendations.values(),
                    key=lambda x: x['similarity_score'] * x['recommendation_count'],
                    reverse=True
                )[:n_recommendations]
                
                # Add rank
                for i, rec in enumerate(final_recommendations):
                    rec['rank'] = i + 1
                
                logger.info(f"Generated {len(final_recommendations)} personalized recommendations for user {user_id}")
                return final_recommendations
                
        except Exception as e:
            logger.error(f"Error getting user recommendations: {e}")
            return self._get_popular_players(conn, n_recommendations)

    def _get_popular_players(self, conn, n_recommendations: int) -> List[Dict]:
        """Fallback: get popular players based on engagement"""
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        pee.player_id,
                        COUNT(*) as total_interactions,
                        COUNT(DISTINCT pee.user_id) as unique_users
                    FROM player_engagement_events pee
                    WHERE pee.created_at > NOW() - INTERVAL '7 days'
                    GROUP BY pee.player_id
                    ORDER BY total_interactions DESC, unique_users DESC
                    LIMIT %s
                """, (n_recommendations,))
                
                rows = cur.fetchall()
                
                recommendations = []
                for i, row in enumerate(rows):
                    recommendations.append({
                        'player_id': row[0],
                        'similarity_score': 1.0 - (i * 0.1),  # Decreasing score
                        'rank': i + 1,
                        'recommendation_count': 1,
                        'reason': 'popular'
                    })
                
                return recommendations
                
        except Exception as e:
            logger.error(f"Error getting popular players: {e}")
            return []

    def refresh_index(self, conn):
        """Refresh the recommendation index with latest data"""
        logger.info("Refreshing recommendation index...")
        return self.build_player_index(conn)