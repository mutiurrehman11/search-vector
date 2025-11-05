#!/usr/bin/env python3
"""
Insert sample player data for testing the search system
"""
import psycopg2
import json
import os
from dotenv import load_dotenv
from datetime import datetime, date
import random
import string

# Load environment variables
load_dotenv()

def generate_player_id():
    """Generate a 26-character player ID (similar to ULID format)"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=26))

def insert_sample_data():
    """Insert sample player data for testing"""
    try:
        # Database configuration
        db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'database': os.getenv('DB_NAME', 'player_search'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password'),
            'port': os.getenv('DB_PORT', 5432)
        }
        
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        print("🏈 Inserting sample player data...")
        
        # Sample players data
        sample_players = [
            {
                'first_name': 'Lionel',
                'last_name': 'Messi',
                'position': 'forward',
                'skill_level': 10,
                'age': 36,
                'location': {'lat': 25.7617, 'lng': -80.1918, 'city': 'Miami', 'country': 'USA'},
                'availability': ['weekends', 'evenings'],
                'tags': ['experienced', 'playmaker', 'left-footed'],
                'height': 1.70,
                'weight': 72.0,
                'gender': 'male',
                'status': 'active'
            },
            {
                'first_name': 'Cristiano',
                'last_name': 'Ronaldo',
                'position': 'forward',
                'skill_level': 10,
                'age': 39,
                'location': {'lat': 24.4539, 'lng': 54.3773, 'city': 'Abu Dhabi', 'country': 'UAE'},
                'availability': ['weekends'],
                'tags': ['experienced', 'striker', 'right-footed'],
                'height': 1.87,
                'weight': 84.0,
                'gender': 'male',
                'status': 'active'
            },
            {
                'first_name': 'Kylian',
                'last_name': 'Mbappé',
                'position': 'forward',
                'skill_level': 9,
                'age': 25,
                'location': {'lat': 48.8566, 'lng': 2.3522, 'city': 'Paris', 'country': 'France'},
                'availability': ['weekdays', 'weekends'],
                'tags': ['fast', 'young', 'striker'],
                'height': 1.78,
                'weight': 73.0,
                'gender': 'male',
                'status': 'active'
            },
            {
                'first_name': 'Virgil',
                'last_name': 'van Dijk',
                'position': 'defender',
                'skill_level': 9,
                'age': 32,
                'location': {'lat': 53.4084, 'lng': -2.9916, 'city': 'Liverpool', 'country': 'UK'},
                'availability': ['weekends'],
                'tags': ['experienced', 'captain', 'tall'],
                'height': 1.93,
                'weight': 92.0,
                'gender': 'male',
                'status': 'active'
            },
            {
                'first_name': 'Kevin',
                'last_name': 'De Bruyne',
                'position': 'midfielder',
                'skill_level': 9,
                'age': 32,
                'location': {'lat': 53.4808, 'lng': -2.2426, 'city': 'Manchester', 'country': 'UK'},
                'availability': ['evenings', 'weekends'],
                'tags': ['playmaker', 'experienced', 'creative'],
                'height': 1.81,
                'weight': 70.0,
                'gender': 'male',
                'status': 'active'
            },
            {
                'first_name': 'Alisson',
                'last_name': 'Becker',
                'position': 'goalkeeper',
                'skill_level': 9,
                'age': 31,
                'location': {'lat': 53.4084, 'lng': -2.9916, 'city': 'Liverpool', 'country': 'UK'},
                'availability': ['weekends'],
                'tags': ['experienced', 'reliable', 'tall'],
                'height': 1.91,
                'weight': 91.0,
                'gender': 'male',
                'status': 'active'
            },
            {
                'first_name': 'Alex',
                'last_name': 'Morgan',
                'position': 'forward',
                'skill_level': 8,
                'age': 34,
                'location': {'lat': 32.7157, 'lng': -117.1611, 'city': 'San Diego', 'country': 'USA'},
                'availability': ['weekends', 'evenings'],
                'tags': ['experienced', 'striker', 'leader'],
                'height': 1.70,
                'weight': 61.0,
                'gender': 'female',
                'status': 'active'
            },
            {
                'first_name': 'Megan',
                'last_name': 'Rapinoe',
                'position': 'midfielder',
                'skill_level': 8,
                'age': 38,
                'location': {'lat': 47.6062, 'lng': -122.3321, 'city': 'Seattle', 'country': 'USA'},
                'availability': ['weekends'],
                'tags': ['experienced', 'captain', 'creative'],
                'height': 1.68,
                'weight': 61.0,
                'gender': 'female',
                'status': 'active'
            }
        ]
        
        # Insert players
        for player_data in sample_players:
            player_id = generate_player_id()
            
            # Calculate birth_date from age
            current_year = datetime.now().year
            birth_year = current_year - player_data['age']
            birth_date = date(birth_year, 1, 1)
            
            # Insert player
            cursor.execute("""
                INSERT INTO players (
                    id, first_name, last_name, location, birth_date, 
                    height, weight, gender, status, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (
                player_id,
                player_data['first_name'],
                player_data['last_name'],
                json.dumps(player_data['location']),
                birth_date,
                player_data['height'],
                player_data['weight'],
                player_data['gender'],
                player_data['status'],
                datetime.now(),
                datetime.now()
            ))
            
            # Insert player skills
            skills = ['dribbling', 'passing', 'shooting', 'speed', 'stamina', 'strength']
            for skill in skills:
                # Generate skill level based on position and overall skill
                if player_data['position'] == 'goalkeeper':
                    if skill in ['dribbling', 'shooting']:
                        level = max(1, player_data['skill_level'] - 3)
                    else:
                        level = player_data['skill_level']
                elif player_data['position'] == 'defender':
                    if skill in ['shooting']:
                        level = max(1, player_data['skill_level'] - 2)
                    else:
                        level = player_data['skill_level']
                else:
                    level = player_data['skill_level']
                
                cursor.execute("""
                    INSERT INTO player_skills (player_id, skill, level, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s)
                """, (player_id, skill, level, datetime.now(), datetime.now()))
            
            # Generate and insert sample embedding (128-dimensional)
            embedding = [random.uniform(-1, 1) for _ in range(128)]
            
            cursor.execute("""
                INSERT INTO player_embeddings_fallback (player_id, embedding, updated_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (player_id) DO UPDATE SET 
                    embedding = EXCLUDED.embedding,
                    updated_at = EXCLUDED.updated_at
            """, (player_id, json.dumps(embedding), datetime.now()))
            
            # Insert engagement stats
            cursor.execute("""
                INSERT INTO player_engagement_stats (
                    player_id, ctr, recent_activity_score, follow_rate, message_rate, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (player_id) DO NOTHING
            """, (
                player_id,
                random.uniform(0.01, 0.15),  # CTR
                random.uniform(0.1, 1.0),    # Activity score
                random.uniform(0.05, 0.3),   # Follow rate
                random.uniform(0.02, 0.1),   # Message rate
                datetime.now()
            ))
            
            print(f"✅ Inserted player: {player_data['first_name']} {player_data['last_name']} ({player_data['position']})")
        
        # Insert some sample engagement events
        cursor.execute("SELECT id FROM players LIMIT 5")
        player_ids = [row[0] for row in cursor.fetchall()]
        
        event_types = ['impression', 'profile_view', 'follow', 'message']
        for i in range(20):  # 20 sample events
            cursor.execute("""
                INSERT INTO player_engagement_events (
                    user_id, player_id, event_type, query_context, created_at
                ) VALUES (%s, %s, %s, %s, %s)
            """, (
                generate_player_id()[:26],  # Sample user ID
                random.choice(player_ids),
                random.choice(event_types),
                json.dumps({'search_query': 'sample query', 'filters': {}}),
                datetime.now()
            ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"\n🎉 Successfully inserted {len(sample_players)} players with skills, embeddings, and engagement data!")
        print("📊 Sample data includes:")
        print("   - Players from different positions (forward, midfielder, defender, goalkeeper)")
        print("   - Various skill levels and attributes")
        print("   - Geographic diversity (USA, UK, France, UAE)")
        print("   - Both male and female players")
        print("   - Sample engagement events and statistics")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to insert sample data: {e}")
        return False

if __name__ == "__main__":
    success = insert_sample_data()
    if not success:
        exit(1)