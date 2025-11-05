import numpy as np
import json
import logging
from typing import Dict, List, Optional
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import pickle
import os

logger = logging.getLogger(__name__)


class MLReRanker:
    """LightGBM-based Learning-to-Rank model for player search re-ranking"""

    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path
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
        
        # Load existing model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            logger.info(f"Loaded ML re-ranking model from {model_path}")

    def extract_features(self, player: Dict, query_context: Dict) -> List[float]:
        """Extract features for ML re-ranking"""
        features = []
        
        # Vector similarity (from previous stage)
        features.append(player.get('similarity_score', 0.0))
        
        # Skill match score
        skill_match = self._calculate_skill_match(player, query_context)
        features.append(skill_match)
        
        # Location distance (normalized)
        distance = player.get('distance_km', 1000)  # Default to far if unknown
        normalized_distance = max(0, 1 - (distance / 100))  # Normalize to 0-1
        features.append(normalized_distance)
        
        # Availability overlap (placeholder - would need real availability data)
        availability_overlap = 0.5  # Default neutral score
        features.append(availability_overlap)
        
        # Tag match score
        tag_match = self._calculate_tag_match(player, query_context)
        features.append(tag_match)
        
        # Profile completeness
        completeness = self._calculate_profile_completeness(player)
        features.append(completeness)
        
        # Recent activity score (placeholder)
        recent_activity = 0.5  # Default neutral score
        features.append(recent_activity)
        
        # Historical engagement rates (placeholders - would come from database)
        features.extend([0.1, 0.05, 0.02, 0.01])  # CTR, open, save, message rates
        
        return features

    def _calculate_skill_match(self, player: Dict, query_context: Dict) -> float:
        """Calculate how well player skills match query requirements"""
        player_skills = player.get('skills', {})
        if not player_skills:
            return 0.5
        
        # Simple average skill level match
        avg_skill = np.mean(list(player_skills.values()))
        
        # Check if query has skill requirements
        min_skill = query_context.get('min_skill', 0)
        max_skill = query_context.get('max_skill', 10)
        
        if min_skill <= avg_skill <= max_skill:
            return min(1.0, avg_skill / 10.0)
        else:
            return 0.3  # Penalty for not matching skill range
    
    def _calculate_tag_match(self, player: Dict, query_context: Dict) -> float:
        """Calculate tag matching score"""
        player_tags = set(player.get('tags', []))
        query_tags = set(query_context.get('tags', []))
        
        if not query_tags:
            return 0.5  # Neutral if no tag preferences
        
        if not player_tags:
            return 0.3  # Slight penalty for no tags
        
        # Jaccard similarity
        intersection = len(player_tags & query_tags)
        union = len(player_tags | query_tags)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_profile_completeness(self, player: Dict) -> float:
        """Calculate profile completeness score"""
        completeness_factors = [
            bool(player.get('first_name')),
            bool(player.get('last_name')),
            bool(player.get('skills')),
            bool(player.get('location')),
            bool(player.get('positions')),
            player.get('age', 0) > 0,
            len(player.get('skills', {})) >= 3,
        ]
        
        return sum(completeness_factors) / len(completeness_factors)

    def prepare_training_data(self, conn):
        """Prepare training data from engagement events"""
        try:
            with conn.cursor() as cur:
                # Get engagement data with search context
                cur.execute("""
                    SELECT 
                        pee.user_id,
                        pee.player_id,
                        pee.event_type,
                        pee.query_context,
                        p.first_name,
                        p.last_name,
                        p.location,
                        p.birth_date,
                        EXTRACT(YEAR FROM AGE(p.birth_date)) as age,
                        (SELECT json_object_agg(ps.skill, ps.level) 
                         FROM player_skills ps WHERE ps.player_id = p.id) as skills,
                        (SELECT array_agg(pt.position) 
                         FROM player_teams pt WHERE pt.player_id = p.id) as positions
                    FROM player_engagement_events pee
                    JOIN players p ON p.id = pee.player_id
                    WHERE pee.created_at > NOW() - INTERVAL '30 days'
                    ORDER BY pee.user_id, pee.created_at
                """)
                
                rows = cur.fetchall()
                
                if not rows:
                    logger.warning("No training data available")
                    return np.array([]), np.array([]), np.array([])
                
                X, y, groups = [], [], []
                current_group = 0
                current_user = None
                
                for row in rows:
                    user_id = row[0]
                    if user_id != current_user:
                        current_group += 1
                        current_user = user_id
                    
                    # Create player dict
                    player = {
                        'id': row[1],
                        'first_name': row[4],
                        'last_name': row[5],
                        'location': json.loads(row[6]) if row[6] else {},
                        'age': row[8],
                        'skills': json.loads(row[9]) if row[9] else {},
                        'positions': row[10] or []
                    }
                    
                    # Query context
                    query_context = json.loads(row[3]) if row[3] else {}
                    
                    # Extract features
                    features = self.extract_features(player, query_context)
                    X.append(features)
                    
                    # Create engagement score (higher for more engaged events)
                    event_type = row[2]
                    engagement_scores = {
                        'impression': 0,
                        'profile_view': 1,
                        'follow': 2,
                        'save_to_playlist': 3,
                        'message': 4
                    }
                    y.append(engagement_scores.get(event_type, 0))
                    groups.append(current_group)
                
                return np.array(X), np.array(y), np.array(groups)
                
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return np.array([]), np.array([]), np.array([])

    def train(self, conn):
        """Train LightGBM ranker on engagement data"""
        logger.info("Starting ML re-ranker training...")
        
        X, y, groups = self.prepare_training_data(conn)
        
        if len(X) == 0:
            logger.warning("No training data available yet")
            return False
        
        try:
            # Standardize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Create LightGBM dataset with query groups
            unique_groups, group_counts = np.unique(groups, return_counts=True)
            train_data = lgb.Dataset(
                X_scaled,
                label=y,
                group=group_counts
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
                'verbose': -1,
                'force_row_wise': True
            }
            
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(10)]
            )
            
            logger.info("ML re-ranking model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False

    def rerank(self, players: List[Dict], query_context: Dict) -> List[Dict]:
        """Re-rank players using trained model"""
        if not self.model or not players:
            return players
        
        try:
            # Extract features for all players
            X = np.array([
                self.extract_features(p, query_context)
                for p in players
            ])
            
            # Standardize features
            X_scaled = self.scaler.transform(X)
            
            # Predict scores
            scores = self.model.predict(X_scaled)
            
            # Add scores and sort
            for player, score in zip(players, scores):
                player['ml_score'] = float(score)
            
            # Sort by ML score (higher is better)
            reranked = sorted(players, key=lambda p: p['ml_score'], reverse=True)
            
            logger.info(f"Re-ranked {len(players)} players using ML model")
            return reranked
            
        except Exception as e:
            logger.error(f"Error in ML re-ranking: {e}")
            return players

    def save_model(self, path: str):
        """Save the trained model and scaler"""
        if not self.model:
            logger.warning("No model to save")
            return
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }
            
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def load_model(self, path: str):
        """Load a trained model and scaler"""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data.get('feature_names', self.feature_names)
            
            logger.info(f"Model loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False