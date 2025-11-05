"""
Player Embedding Generator - Compatible with current database schema
Generates 128-dimensional embeddings from player attributes
"""

import numpy as np
import json
import hashlib
from typing import Dict, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CompatPlayerEmbeddingGenerator:
    """Generate player embeddings from structured attributes"""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        
        # Define position mappings
        self.positions = [
            'goalkeeper', 'defender', 'midfielder', 'forward',
            'center-back', 'full-back', 'wing-back', 'defensive-midfielder',
            'central-midfielder', 'attacking-midfielder', 'winger', 'striker'
        ]
        
        # Skill categories
        self.skill_categories = [
            'technical', 'physical', 'mental', 'tactical',
            'shooting', 'passing', 'dribbling', 'defending',
            'goalkeeping', 'crossing', 'finishing', 'pace'
        ]
    
    def generate_embedding(self, player_data: Dict) -> np.ndarray:
        """Generate embedding from player data"""
        try:
            embedding_parts = []
            
            # 1. Position encoding (16 dims)
            position_encoding = self._encode_positions(player_data.get('positions', []))
            embedding_parts.append(position_encoding)
            
            # 2. Skill level encoding (32 dims)
            skill_encoding = self._encode_skills(player_data.get('skills', {}))
            embedding_parts.append(skill_encoding)
            
            # 3. Physical attributes (16 dims)
            physical_encoding = self._encode_physical_attributes(player_data)
            embedding_parts.append(physical_encoding)
            
            # 4. Location encoding (24 dims)
            location_encoding = self._encode_location(player_data.get('location', {}))
            embedding_parts.append(location_encoding)
            
            # 5. Age encoding (8 dims)
            age_encoding = self._encode_age(player_data.get('age', 0))
            embedding_parts.append(age_encoding)
            
            # 6. Experience/Performance encoding (16 dims)
            experience_encoding = self._encode_experience(player_data)
            embedding_parts.append(experience_encoding)
            
            # 7. Availability/Status encoding (8 dims)
            availability_encoding = self._encode_availability(player_data)
            embedding_parts.append(availability_encoding)
            
            # 8. Gender encoding (8 dims)
            gender_encoding = self._encode_gender(player_data.get('gender', 'unknown'))
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
            
            logger.debug(f"Generated embedding with shape: {full_embedding.shape}")
            return full_embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return random normalized embedding as fallback
            random_embedding = np.random.normal(0, 0.1, self.embedding_dim)
            return random_embedding / np.linalg.norm(random_embedding)
    
    def _encode_positions(self, positions: List[str]) -> np.ndarray:
        """Encode player positions (16 dims)"""
        encoding = np.zeros(16)
        
        if not positions:
            return encoding
        
        # One-hot encoding for primary positions
        for i, pos in enumerate(self.positions[:12]):  # Use first 12 positions
            if pos in positions:
                encoding[i] = 1.0
        
        # Additional position features
        if 'goalkeeper' in positions:
            encoding[12] = 1.0
        if any(pos in ['defender', 'center-back', 'full-back', 'wing-back'] for pos in positions):
            encoding[13] = 1.0
        if any(pos in ['midfielder', 'defensive-midfielder', 'central-midfielder', 'attacking-midfielder'] for pos in positions):
            encoding[14] = 1.0
        if any(pos in ['forward', 'winger', 'striker'] for pos in positions):
            encoding[15] = 1.0
        
        return encoding
    
    def _encode_skills(self, skills: Dict[str, float]) -> np.ndarray:
        """Encode player skills (32 dims)"""
        encoding = np.zeros(32)
        
        if not skills:
            return encoding
        
        # Map known skills to dimensions
        skill_mapping = {
            'technical': 0, 'physical': 1, 'mental': 2, 'tactical': 3,
            'shooting': 4, 'passing': 5, 'dribbling': 6, 'defending': 7,
            'goalkeeping': 8, 'crossing': 9, 'finishing': 10, 'pace': 11,
            'strength': 12, 'stamina': 13, 'agility': 14, 'balance': 15
        }
        
        # Encode known skills
        for skill, level in skills.items():
            if skill.lower() in skill_mapping:
                idx = skill_mapping[skill.lower()]
                encoding[idx] = min(level / 100.0, 1.0)  # Normalize to 0-1
        
        # Calculate aggregate metrics
        if skills:
            avg_skill = np.mean(list(skills.values())) / 100.0
            max_skill = max(skills.values()) / 100.0
            min_skill = min(skills.values()) / 100.0
            skill_variance = np.var(list(skills.values())) / 10000.0
            
            encoding[16] = avg_skill
            encoding[17] = max_skill
            encoding[18] = min_skill
            encoding[19] = skill_variance
            
            # Skill distribution features
            high_skills = sum(1 for v in skills.values() if v >= 80) / len(skills)
            low_skills = sum(1 for v in skills.values() if v <= 40) / len(skills)
            
            encoding[20] = high_skills
            encoding[21] = low_skills
        
        return encoding
    
    def _encode_physical_attributes(self, player_data: Dict) -> np.ndarray:
        """Encode physical attributes (16 dims)"""
        encoding = np.zeros(16)
        
        # Height encoding (normalized)
        height = player_data.get('height', 175)  # Default 175cm
        encoding[0] = min(max((height - 150) / 50.0, 0), 1)  # Normalize 150-200cm to 0-1
        
        # Weight encoding (normalized)
        weight = player_data.get('weight', 70)  # Default 70kg
        encoding[1] = min(max((weight - 50) / 50.0, 0), 1)  # Normalize 50-100kg to 0-1
        
        # BMI calculation
        if height > 0:
            bmi = weight / ((height / 100) ** 2)
            encoding[2] = min(max((bmi - 18) / 10.0, 0), 1)  # Normalize BMI 18-28 to 0-1
        
        # Physical build categories
        if height >= 185:
            encoding[3] = 1.0  # Tall
        elif height <= 170:
            encoding[4] = 1.0  # Short
        else:
            encoding[5] = 1.0  # Average height
        
        if weight >= 80:
            encoding[6] = 1.0  # Heavy
        elif weight <= 65:
            encoding[7] = 1.0  # Light
        else:
            encoding[8] = 1.0  # Average weight
        
        return encoding
    
    def _encode_location(self, location: Dict) -> np.ndarray:
        """Encode location information (24 dims)"""
        encoding = np.zeros(24)
        
        if not location:
            return encoding
        
        # Latitude/Longitude encoding
        lat = location.get('lat', 0)
        lng = location.get('lng', 0)
        
        # Normalize coordinates
        encoding[0] = (lat + 90) / 180.0  # Normalize -90 to 90 -> 0 to 1
        encoding[1] = (lng + 180) / 360.0  # Normalize -180 to 180 -> 0 to 1
        
        # Geographic regions (simplified)
        if -90 <= lat <= 90 and -180 <= lng <= 180:
            # Continental regions
            if 35 <= lat <= 70 and -10 <= lng <= 40:  # Europe
                encoding[2] = 1.0
            elif 25 <= lat <= 50 and -125 <= lng <= -65:  # North America
                encoding[3] = 1.0
            elif -35 <= lat <= 15 and -80 <= lng <= -35:  # South America
                encoding[4] = 1.0
            elif -35 <= lat <= 35 and 10 <= lng <= 50:  # Africa
                encoding[5] = 1.0
            elif 10 <= lat <= 55 and 70 <= lng <= 140:  # Asia
                encoding[6] = 1.0
            elif -45 <= lat <= -10 and 110 <= lng <= 155:  # Australia
                encoding[7] = 1.0
        
        # City/Country information if available
        city = location.get('city', '').lower()
        country = location.get('country', '').lower()
        
        # Hash-based encoding for city/country
        if city:
            city_hash = int(hashlib.md5(city.encode()).hexdigest()[:8], 16)
            encoding[8:16] = [(city_hash >> i) & 1 for i in range(8)]
        
        if country:
            country_hash = int(hashlib.md5(country.encode()).hexdigest()[:8], 16)
            encoding[16:24] = [(country_hash >> i) & 1 for i in range(8)]
        
        return encoding
    
    def _encode_age(self, age: int) -> np.ndarray:
        """Encode age information (8 dims)"""
        encoding = np.zeros(8)
        
        # Normalize age
        normalized_age = min(max((age - 16) / 24.0, 0), 1)  # Normalize 16-40 to 0-1
        encoding[0] = normalized_age
        
        # Age categories
        if age < 18:
            encoding[1] = 1.0  # Youth
        elif age < 23:
            encoding[2] = 1.0  # Young
        elif age < 30:
            encoding[3] = 1.0  # Prime
        elif age < 35:
            encoding[4] = 1.0  # Experienced
        else:
            encoding[5] = 1.0  # Veteran
        
        # Age-related features
        encoding[6] = np.sin(2 * np.pi * age / 40.0)  # Cyclical encoding
        encoding[7] = np.cos(2 * np.pi * age / 40.0)
        
        return encoding
    
    def _encode_experience(self, player_data: Dict) -> np.ndarray:
        """Encode experience and performance metrics (16 dims)"""
        encoding = np.zeros(16)
        
        # Average skill level as experience proxy
        avg_skill = player_data.get('avg_skill_level', 50)
        encoding[0] = min(avg_skill / 100.0, 1.0)
        
        # Performance categories based on skill level
        if avg_skill >= 90:
            encoding[1] = 1.0  # Elite
        elif avg_skill >= 80:
            encoding[2] = 1.0  # Professional
        elif avg_skill >= 70:
            encoding[3] = 1.0  # Semi-professional
        elif avg_skill >= 60:
            encoding[4] = 1.0  # Amateur+
        else:
            encoding[5] = 1.0  # Amateur
        
        # Additional performance indicators could be added here
        # For now, use skill-based proxies
        
        return encoding
    
    def _encode_availability(self, player_data: Dict) -> np.ndarray:
        """Encode availability and status (8 dims)"""
        encoding = np.zeros(8)
        
        status = player_data.get('status', 'active').lower()
        
        if status == 'active':
            encoding[0] = 1.0
        elif status == 'inactive':
            encoding[1] = 1.0
        elif status == 'injured':
            encoding[2] = 1.0
        elif status == 'suspended':
            encoding[3] = 1.0
        
        # Additional availability features could be added
        # For now, keep simple
        
        return encoding
    
    def _encode_gender(self, gender: str) -> np.ndarray:
        """Encode gender information (8 dims)"""
        encoding = np.zeros(8)
        
        gender = gender.lower()
        
        if gender == 'male' or gender == 'm':
            encoding[0] = 1.0
        elif gender == 'female' or gender == 'f':
            encoding[1] = 1.0
        else:
            encoding[2] = 1.0  # Other/Unknown
        
        return encoding