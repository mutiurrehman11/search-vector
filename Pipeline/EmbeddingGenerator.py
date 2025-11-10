import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from pgvector.psycopg2 import register_vector
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
import logging

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
