import numpy as np
import json
import hashlib
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class QueryVectorBuilder:
    """Build query vectors from search filters"""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        
        # Position mappings (same as in EmbeddingGenerator)
        self.positions = [
            'goalkeeper', 'defender', 'midfielder', 'forward',
            'center-back', 'full-back', 'wing-back', 'defensive-midfielder',
            'central-midfielder', 'attacking-midfielder', 'winger', 'striker'
        ]
    
    def build_query_vector(self, filters: Dict) -> np.ndarray:
        """Build query vector from search filters"""
        try:
            embedding_parts = []
            
            # 1. Position encoding (16 dims)
            position_encoding = self._encode_position_filter(filters.get('position'))
            embedding_parts.append(position_encoding)
            
            # 2. Skill level encoding (32 dims)
            skill_encoding = self._encode_skill_filters(filters)
            embedding_parts.append(skill_encoding)
            
            # 3. Physical attributes (16 dims) - neutral for query
            physical_encoding = np.zeros(16)
            embedding_parts.append(physical_encoding)
            
            # 4. Location encoding (24 dims)
            location_encoding = self._encode_location_filter(filters)
            embedding_parts.append(location_encoding)
            
            # 5. Age encoding (8 dims)
            age_encoding = self._encode_age_filter(filters)
            embedding_parts.append(age_encoding)
            
            # 6. Experience/Performance encoding (16 dims)
            experience_encoding = self._encode_experience_filter(filters)
            embedding_parts.append(experience_encoding)
            
            # 7. Availability/Status encoding (8 dims)
            availability_encoding = self._encode_availability_filter(filters)
            embedding_parts.append(availability_encoding)
            
            # 8. Gender encoding (8 dims)
            gender_encoding = self._encode_gender_filter(filters.get('gender'))
            embedding_parts.append(gender_encoding)
            
            # Concatenate all parts
            full_embedding = np.concatenate(embedding_parts)
            
            # Ensure exact dimension
            if len(full_embedding) > self.embedding_dim:
                full_embedding = full_embedding[:self.embedding_dim]
            elif len(full_embedding) < self.embedding_dim:
                padding = np.zeros(self.embedding_dim - len(full_embedding))
                full_embedding = np.concatenate([full_embedding, padding])
            
            # Normalize
            norm = np.linalg.norm(full_embedding)
            if norm > 0:
                full_embedding = full_embedding / norm
            else:
                # If all zeros, create a small random vector
                full_embedding = np.random.normal(0, 0.01, self.embedding_dim)
                full_embedding = full_embedding / np.linalg.norm(full_embedding)
            
            logger.debug(f"Built query vector with shape: {full_embedding.shape}")
            return full_embedding
            
        except Exception as e:
            logger.error(f"Failed to build query vector: {e}")
            # Return random normalized embedding as fallback
            random_embedding = np.random.normal(0, 0.1, self.embedding_dim)
            return random_embedding / np.linalg.norm(random_embedding)
    
    def _encode_position_filter(self, position: Optional[str]) -> np.ndarray:
        """Encode position filter (16 dims)"""
        encoding = np.zeros(16)
        
        if not position or position == 'any':
            return encoding
        
        position = position.lower()
        
        # One-hot encoding for specific position
        for i, pos in enumerate(self.positions[:12]):
            if pos == position:
                encoding[i] = 1.0
                break
        
        # Broader category encoding
        if position in ['goalkeeper']:
            encoding[12] = 1.0
        elif position in ['defender', 'center-back', 'full-back', 'wing-back']:
            encoding[13] = 1.0
        elif position in ['midfielder', 'defensive-midfielder', 'central-midfielder', 'attacking-midfielder']:
            encoding[14] = 1.0
        elif position in ['forward', 'winger', 'striker']:
            encoding[15] = 1.0
        
        return encoding
    
    def _encode_skill_filters(self, filters: Dict) -> np.ndarray:
        """Encode skill level filters (32 dims)"""
        encoding = np.zeros(32)
        
        min_skill = filters.get('min_skill', 0)
        max_skill = filters.get('max_skill', 100)
        
        if min_skill > 0 or max_skill < 100:
            # Target skill range
            target_skill = (min_skill + max_skill) / 2.0
            skill_range = max_skill - min_skill
            
            # Encode target skill level
            encoding[16] = target_skill / 100.0
            encoding[17] = max_skill / 100.0
            encoding[18] = min_skill / 100.0
            encoding[19] = skill_range / 100.0
            
            # Skill preference indicators
            if target_skill >= 80:
                encoding[20] = 1.0  # High skill preference
            elif target_skill >= 60:
                encoding[21] = 1.0  # Medium skill preference
            else:
                encoding[22] = 1.0  # Entry level preference
        
        return encoding
    
    def _encode_location_filter(self, filters: Dict) -> np.ndarray:
        """Encode location-based filters (24 dims)"""
        encoding = np.zeros(24)
        
        lat = filters.get('latitude')
        lng = filters.get('longitude')
        max_distance = filters.get('max_distance_km')
        
        if lat is not None and lng is not None:
            # Normalize coordinates
            encoding[0] = (lat + 90) / 180.0  # Normalize -90 to 90 -> 0 to 1
            encoding[1] = (lng + 180) / 360.0  # Normalize -180 to 180 -> 0 to 1
            
            # Distance preference
            if max_distance:
                # Normalize distance (0-1000km range)
                encoding[2] = min(max_distance / 1000.0, 1.0)
            
            # Geographic regions
            if 35 <= lat <= 70 and -10 <= lng <= 40:  # Europe
                encoding[3] = 1.0
            elif 25 <= lat <= 50 and -125 <= lng <= -65:  # North America
                encoding[4] = 1.0
            elif -35 <= lat <= 15 and -80 <= lng <= -35:  # South America
                encoding[5] = 1.0
            elif -35 <= lat <= 35 and 10 <= lng <= 50:  # Africa
                encoding[6] = 1.0
            elif 10 <= lat <= 55 and 70 <= lng <= 140:  # Asia
                encoding[7] = 1.0
            elif -45 <= lat <= -10 and 110 <= lng <= 155:  # Australia
                encoding[8] = 1.0
        
        return encoding
    
    def _encode_age_filter(self, filters: Dict) -> np.ndarray:
        """Encode age filters (8 dims)"""
        encoding = np.zeros(8)
        
        min_age = filters.get('min_age', 16)
        max_age = filters.get('max_age', 40)
        
        if min_age > 16 or max_age < 40:
            # Target age range
            target_age = (min_age + max_age) / 2.0
            age_range = max_age - min_age
            
            # Normalize target age
            encoding[0] = min(max((target_age - 16) / 24.0, 0), 1)
            
            # Age category preferences
            if target_age < 20:
                encoding[1] = 1.0  # Youth preference
            elif target_age < 25:
                encoding[2] = 1.0  # Young preference
            elif target_age < 30:
                encoding[3] = 1.0  # Prime preference
            elif target_age < 35:
                encoding[4] = 1.0  # Experienced preference
            else:
                encoding[5] = 1.0  # Veteran preference
            
            # Age range encoding
            encoding[6] = np.sin(2 * np.pi * target_age / 40.0)
            encoding[7] = np.cos(2 * np.pi * target_age / 40.0)
        
        return encoding
    
    def _encode_experience_filter(self, filters: Dict) -> np.ndarray:
        """Encode experience preferences (16 dims)"""
        encoding = np.zeros(16)
        
        # Use skill level as experience proxy
        min_skill = filters.get('min_skill', 0)
        max_skill = filters.get('max_skill', 100)
        
        if min_skill > 0 or max_skill < 100:
            target_skill = (min_skill + max_skill) / 2.0
            encoding[0] = target_skill / 100.0
            
            # Experience level preferences
            if target_skill >= 90:
                encoding[1] = 1.0  # Elite
            elif target_skill >= 80:
                encoding[2] = 1.0  # Professional
            elif target_skill >= 70:
                encoding[3] = 1.0  # Semi-professional
            elif target_skill >= 60:
                encoding[4] = 1.0  # Amateur+
            else:
                encoding[5] = 1.0  # Amateur
        
        return encoding
    
    def _encode_availability_filter(self, filters: Dict) -> np.ndarray:
        """Encode availability preferences (8 dims)"""
        encoding = np.zeros(8)
        
        # Default to active players
        availability = filters.get('availability', 'active')
        
        if availability == 'active':
            encoding[0] = 1.0
        elif availability == 'any':
            # No specific preference
            pass
        
        return encoding
    
    def _encode_gender_filter(self, gender: Optional[str]) -> np.ndarray:
        """Encode gender filter (8 dims)"""
        encoding = np.zeros(8)
        
        if gender:
            gender = gender.lower()
            
            if gender == 'male' or gender == 'm':
                encoding[0] = 1.0
            elif gender == 'female' or gender == 'f':
                encoding[1] = 1.0
            else:
                encoding[2] = 1.0  # Other/Any
        
        return encoding
    
    def build_recommendation_vector(self, seed_player_ids: List[str], player_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Build query vector for recommendations based on seed players"""
        try:
            if not seed_player_ids or not player_embeddings:
                # Return neutral vector
                return np.random.normal(0, 0.01, self.embedding_dim)
            
            # Get embeddings for seed players
            seed_embeddings = []
            for player_id in seed_player_ids:
                if player_id in player_embeddings:
                    seed_embeddings.append(player_embeddings[player_id])
            
            if not seed_embeddings:
                return np.random.normal(0, 0.01, self.embedding_dim)
            
            # Average the seed embeddings
            avg_embedding = np.mean(seed_embeddings, axis=0)
            
            # Normalize
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm
            
            logger.debug(f"Built recommendation vector from {len(seed_embeddings)} seed players")
            return avg_embedding
            
        except Exception as e:
            logger.error(f"Failed to build recommendation vector: {e}")
            return np.random.normal(0, 0.01, self.embedding_dim)