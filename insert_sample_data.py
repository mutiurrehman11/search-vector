import psycopg2
from psycopg2.extras import RealDictCursor
import json
import random
from datetime import datetime, date, timedelta
from faker import Faker
import numpy as np
import os
from dotenv import load_dotenv
import string

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', '127.0.0.1'),
    'database': os.getenv('DB_NAME', 'player_search'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'muti12345'),
    'port': int(os.getenv('DB_PORT', 5432))
}

fake = Faker()

def generate_ulid():
    """Generate a ULID-like ID (26 characters)"""
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choices(chars, k=26))

def generate_player_id():
    """Generate a 26-character ID (ULID-like format)"""
    import string
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=26))

def generate_location_json():
    """Generate location JSON with coordinates"""
    return {
        "latitude": round(random.uniform(-90, 90), 6),
        "longitude": round(random.uniform(-180, 180), 6),
        "city": fake.city(),
        "state": fake.state(),
        "country": fake.country()
    }

def generate_contacts_json():
    """Generate contacts JSON"""
    return {
        "email": fake.email(),
        "phone": fake.phone_number(),
        "social_media": {
            "instagram": f"@{fake.user_name()}",
            "twitter": f"@{fake.user_name()}"
        }
    }

def generate_profile_picture_json():
    """Generate profile picture JSON"""
    return {
        "url": f"https://example.com/profiles/{fake.uuid4()}.jpg",
        "thumbnail": f"https://example.com/thumbnails/{fake.uuid4()}.jpg"
    }

def generate_pictures_json():
    """Generate pictures array JSON"""
    return [
        {
            "url": f"https://example.com/gallery/{fake.uuid4()}.jpg",
            "caption": fake.sentence()
        } for _ in range(random.randint(1, 5))
    ]

def generate_vector_embedding():
    """Generate a random 128-dimensional vector for embeddings"""
    return np.random.normal(0, 1, 128).tolist()

def insert_sample_data():
    """Insert sample data into all tables"""
    
    try:
        # Connect to database
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        print("Connected to database successfully!")
        
        # Store generated IDs for foreign key references
        player_ids = []
        chat_room_ids = []
        playlist_ids = []
        
        # 1. Insert Countries
        print("Inserting countries...")
        countries_data = [
            ('United States', 'USA'),
            ('Canada', 'CAN'),
            ('United Kingdom', 'GBR'),
            ('Germany', 'DEU'),
            ('France', 'FRA'),
            ('Spain', 'ESP'),
            ('Italy', 'ITA'),
            ('Brazil', 'BRA'),
            ('Argentina', 'ARG'),
            ('Mexico', 'MEX')
        ]
        
        for name, code in countries_data:
            cur.execute(
                "INSERT INTO countries (name, code) VALUES (%s, %s) ON CONFLICT (code) DO NOTHING",
                (name, code)
            )
        
        # Get country IDs
        cur.execute("SELECT id, name FROM countries")
        countries = {row['name']: row['id'] for row in cur.fetchall()}
        
        # 2. Insert States
        print("Inserting states...")
        states_data = []
        for country_name, country_id in countries.items():
            for i in range(5):  # 5 states per country
                state_name = fake.state()
                state_code = fake.state_abbr()
                states_data.append((state_name, state_code, country_id))
        
        for name, code, country_id in states_data:
            cur.execute(
                "INSERT INTO states (name, code, country_id) VALUES (%s, %s, %s)",
                (name, code, country_id)
            )
        
        # Get state IDs
        cur.execute("SELECT id, name FROM states")
        states = {row['name']: row['id'] for row in cur.fetchall()}
        
        # 3. Insert Cities
        print("Inserting cities...")
        cities_data = []
        for state_name, state_id in list(states.items())[:20]:  # Limit to first 20 states
            for i in range(5):  # 5 cities per state
                city_name = fake.city()
                cities_data.append((city_name, state_id))
        
        for name, state_id in cities_data:
            cur.execute(
                "INSERT INTO cities (name, state_id) VALUES (%s, %s)",
                (name, state_id)
            )
        
        # 4. Insert Skills
        print("Inserting skills...")
        skills_data = [
            ('Dribbling', 'Ball control and movement skills', 'Technical'),
            ('Passing', 'Ability to pass the ball accurately', 'Technical'),
            ('Shooting', 'Goal scoring ability', 'Technical'),
            ('Speed', 'Running speed and acceleration', 'Physical'),
            ('Stamina', 'Endurance and fitness level', 'Physical'),
            ('Strength', 'Physical power and body strength', 'Physical'),
            ('Defending', 'Defensive positioning and tackling', 'Tactical'),
            ('Leadership', 'Team leadership and communication', 'Mental'),
            ('Vision', 'Game reading and anticipation', 'Mental'),
            ('Crossing', 'Ability to deliver crosses', 'Technical')
        ]
        
        for name, description, category in skills_data:
            cur.execute(
                "INSERT INTO skills (name, description, category) VALUES (%s, %s, %s)",
                (name, description, category)
            )
        
        # 5. Insert Categories
        print("Inserting categories...")
        categories_data = [
            ('Youth', 'Players under 18 years old'),
            ('Amateur', 'Non-professional players'),
            ('Semi-Professional', 'Part-time professional players'),
            ('Professional', 'Full-time professional players'),
            ('Veteran', 'Experienced players over 35')
        ]
        
        for name, description in categories_data:
            cur.execute(
                "INSERT INTO categories (name, description) VALUES (%s, %s)",
                (name, description)
            )
        
        # 6. Insert Divisions
        print("Inserting divisions...")
        divisions_data = [
            ('Premier League', 1),
            ('Championship', 2),
            ('League One', 3),
            ('League Two', 4),
            ('National League', 5)
        ]
        
        for name, level in divisions_data:
            cur.execute(
                "INSERT INTO divisions (name, level) VALUES (%s, %s)",
                (name, level)
            )
        
        # 7. Insert Soccer Positions
        print("Inserting soccer positions...")
        positions_data = [
            ('Goalkeeper', 'GK', 'Primary goalkeeper position'),
            ('Center Back', 'CB', 'Central defensive position'),
            ('Left Back', 'LB', 'Left defensive position'),
            ('Right Back', 'RB', 'Right defensive position'),
            ('Defensive Midfielder', 'CDM', 'Defensive midfield position'),
            ('Central Midfielder', 'CM', 'Central midfield position'),
            ('Attacking Midfielder', 'CAM', 'Attacking midfield position'),
            ('Left Winger', 'LW', 'Left wing position'),
            ('Right Winger', 'RW', 'Right wing position'),
            ('Striker', 'ST', 'Central attacking position')
        ]
        
        for name, code, description in positions_data:
            cur.execute(
                "INSERT INTO soccer_positions (name, code, description) VALUES (%s, %s, %s)",
                (name, code, description)
            )
        
        # 8. Insert Chat Rooms
        print("Inserting chat rooms...")
        for i in range(100):
            name = fake.catch_phrase()
            slug = name.lower().replace(' ', '-').replace(',', '').replace('.', '')
            
            cur.execute("""
                INSERT INTO chat_rooms (name, slug, description, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s) RETURNING id
            """, (
                name,
                slug,
                fake.text(max_nb_chars=100),
                fake.date_time_this_year(),
                fake.date_time_this_year()
            ))
            result = cur.fetchone()
            chat_room_id = result['id']
            chat_room_ids.append(chat_room_id)
        print("✅ Chat rooms inserted successfully!")
        
        # 9. Insert Playlists
        print("Inserting playlists...")
        for i in range(100):
            cur.execute("""
                INSERT INTO playlists (name, description, is_public, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s) RETURNING id
            """, (
                fake.catch_phrase(),
                fake.text(max_nb_chars=100),
                fake.boolean(),
                fake.date_time_this_year(),
                fake.date_time_this_year()
            ))
            result = cur.fetchone()
            playlist_id = result['id']
            playlist_ids.append(playlist_id)
        print("✅ Playlists inserted successfully!")
        
        # 10. Insert Players (100 records)
        print("Inserting players...")
        positions = ['forward', 'midfielder', 'defender', 'goalkeeper']
        statuses = ['active', 'inactive', 'pending']
        genders = ['male', 'female', 'other']
        
        for i in range(100):
            player_id = generate_ulid()
            player_ids.append(player_id)
            
            first_name = fake.first_name()
            last_name = fake.last_name()
            contacts = generate_contacts_json()
            description = fake.text(max_nb_chars=200)
            playing_since = fake.date_between(start_date='-10y', end_date='today')
            height = round(random.uniform(1.5, 2.1), 2)  # meters
            weight = round(random.uniform(50, 120), 1)  # kg
            location = generate_location_json()
            birth_date = fake.date_between(start_date='-40y', end_date='-16y')
            document = fake.ssn()
            profile_picture = generate_profile_picture_json()
            pictures = generate_pictures_json()
            gender = random.choice(genders)
            status = random.choice(statuses)
            user_id = generate_ulid()
            
            cur.execute("""
                INSERT INTO players (
                    id, first_name, last_name, contacts, description, playing_since,
                    height, weight, location, birth_date, document, profile_picture,
                    pictures, gender, status, user_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                player_id, first_name, last_name, json.dumps(contacts), description,
                playing_since, height, weight, json.dumps(location), birth_date,
                document, json.dumps(profile_picture), json.dumps(pictures),
                gender, status, user_id
            ))
        
        # 11. Insert Player Skills
        print("Inserting player skills...")
        skills = ['dribbling', 'passing', 'shooting', 'speed', 'stamina', 'strength']
        
        for player_id in player_ids:
            # Each player gets 3-6 random skills
            num_skills = random.randint(3, 6)
            player_skills = random.sample(skills, num_skills)
            
            for skill in player_skills:
                cur.execute("""
                    INSERT INTO player_skills (player_id, skill, level, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    player_id,
                    skill,
                    random.randint(1, 10),
                    fake.date_time_this_year(),
                    fake.date_time_this_year()
                ))
        print("✅ Player skills inserted successfully!")
        
        # 12. Insert Player Teams
        print("Inserting player teams...")
        for player_id in player_ids[:50]:  # Half the players have team associations
            cur.execute("""
                INSERT INTO player_teams (player_id, playable_type, playable_id, overview, 
                                        start_at, end_at, position, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                player_id,
                random.choice(['team', 'club', 'academy']),
                generate_ulid(),  # playable_id
                fake.text(max_nb_chars=100),
                fake.date_time_between(start_date='-2y', end_date='now'),
                fake.date_time_between(start_date='now', end_date='+1y') if fake.boolean() else None,
                random.choice(['forward', 'midfielder', 'defender', 'goalkeeper']),
                fake.date_time_this_year(),
                fake.date_time_this_year()
            ))
        print("✅ Player teams inserted successfully!")
        
        # 13. Insert Player Improvements
        print("Inserting player improvements...")
        for player_id in player_ids:
            cur.execute("""
                INSERT INTO player_improvements (player_id, assessment_date, skill_score, physical_score, mental_score, overall_score, improvement_status, notes, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                player_id,
                fake.date_this_year(),
                round(fake.random.uniform(1.0, 10.0), 2),
                round(fake.random.uniform(1.0, 10.0), 2),
                round(fake.random.uniform(1.0, 10.0), 2),
                round(fake.random.uniform(1.0, 10.0), 2),
                fake.random_element(elements=('Improving', 'Stable', 'Declining', 'Excellent')),
                fake.text(max_nb_chars=100),
                fake.date_time_this_year(),
                fake.date_time_this_year()
            ))
        print("✅ Player improvements inserted successfully!")

        # 14. Insert Player Goals
        print("Inserting player goals...")
        for player_id in player_ids:
            cur.execute("""
                INSERT INTO player_goals (player_id, title, description, target_date, status, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                player_id,
                fake.catch_phrase(),
                fake.text(max_nb_chars=200),
                fake.date_between(start_date='today', end_date='+1y'),
                fake.random_element(elements=('active', 'completed', 'paused', 'cancelled')),
                fake.date_time_this_year(),
                fake.date_time_this_year()
            ))
        print("✅ Player goals inserted successfully!")

        # 15. Insert Player Training Sessions
        print("Inserting player training sessions...")
        for player_id in player_ids:
            cur.execute("""
                INSERT INTO player_training_sessions (player_id, session_date, duration_minutes, intensity, notes, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                player_id,
                fake.date_this_year(),
                fake.random_int(min=30, max=180),
                fake.random_element(elements=('Low', 'Medium', 'High', 'Very High')),
                fake.text(max_nb_chars=150),
                fake.date_time_this_year(),
                fake.date_time_this_year()
            ))
        print("✅ Player training sessions inserted successfully!")
        
        # 16. Insert Views
        print("Inserting views...")
        for i in range(200):
            cur.execute("""
                INSERT INTO views (ip, viewable_type, viewable_id, viewerable_type, viewerable_id, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                fake.ipv4(),
                'player',
                fake.random_element(elements=player_ids),
                'player',
                fake.random_element(elements=player_ids),
                fake.date_time_this_year(),
                fake.date_time_this_year()
            ))
        print("✅ Views inserted successfully!")

        # 17. Insert Follows
        print("Inserting follows...")
        for i in range(150):
            cur.execute("""
                INSERT INTO follows (followerable_type, followerable_id, followedable_type, followedable_id, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                'player',
                fake.random_element(elements=player_ids),
                'player',
                fake.random_element(elements=player_ids),
                fake.date_time_this_year(),
                fake.date_time_this_year()
            ))
        print("✅ Follows inserted successfully!")

        # 18. Insert Chat Messages
        print("Inserting chat messages...")
        for i in range(200):
            cur.execute("""
                INSERT INTO chat_messages (chat_room_id, sender_type, sender_id, message, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                fake.random_element(elements=chat_room_ids),
                'player',
                fake.random_element(elements=player_ids),
                fake.text(max_nb_chars=200),
                fake.date_time_this_year(),
                fake.date_time_this_year()
            ))
        print("✅ Chat messages inserted successfully!")
        
        # 19. Insert Player Embeddings (if pgvector is available)
        print("Inserting player embeddings...")
        try:
            # Create a savepoint before attempting embeddings
            cur.execute("SAVEPOINT embedding_savepoint")
            for player_id in player_ids:
                embedding = generate_vector_embedding()
                cur.execute("""
                    INSERT INTO player_embeddings (player_id, embedding, created_at, updated_at)
                    VALUES (%s, %s, %s, %s)
                """, (player_id, embedding.tolist(), fake.date_time_this_year(), fake.date_time_this_year()))
            print("✅ Player embeddings inserted successfully!")
        except Exception as e:
            print(f"Warning: Could not insert embeddings (pgvector not available): {e}")
            # Rollback only to the savepoint, not the entire transaction
            cur.execute("ROLLBACK TO SAVEPOINT embedding_savepoint")
            print("Continuing with other tables...")

        # 20. Insert Player Engagement Stats
        print("Inserting player engagement stats...")
        for player_id in player_ids:
            cur.execute("""
                INSERT INTO player_engagement_stats (player_id, ctr, recent_activity_score, follow_rate, message_rate, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                player_id,
                round(random.uniform(0.0, 1.0), 4),
                round(random.uniform(0.0, 100.0), 2),
                round(random.uniform(0.0, 1.0), 4),
                round(random.uniform(0.0, 1.0), 4),
                fake.date_time_this_year()
            ))
        print("✅ Player engagement stats inserted successfully!")
        
        # 21. Insert Player Engagement Events
        print("Inserting player engagement events...")
        for i in range(200):
            cur.execute("""
                INSERT INTO player_engagement_events (user_id, player_id, event_type, query_context, created_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                fake.random_element(elements=player_ids),  # user_id
                fake.random_element(elements=player_ids),  # player_id
                fake.random_element(elements=('impression', 'profile_view', 'follow', 'message', 'save_to_playlist')),
                json.dumps({
                    'search_query': fake.word(),
                    'filters': {'position': fake.word()},
                    'page': fake.random_int(min=1, max=5)
                }),
                fake.date_time_this_year()
            ))
        print("✅ Player engagement events inserted successfully!")

        # 22. Insert Saved Searches
        print("Inserting saved searches...")
        try:
            # Create a savepoint before attempting saved searches
            cur.execute("SAVEPOINT saved_searches_savepoint")
            for i in range(100):
                cur.execute("""
                    INSERT INTO saved_searches (user_id, search_name, filters, query_embedding, alert_frequency, last_alerted_at, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    random.choice(player_ids),  # user_id
                    fake.catch_phrase(),
                    json.dumps({
                        'position': random.choice(["forward", "midfielder", "defender", "goalkeeper"]),
                        'age_range': [18, 30],
                        'location': fake.city()
                    }),
                    generate_vector_embedding(),
                    fake.random_element(elements=('daily', 'weekly', 'monthly')),
                    fake.date_time_this_month(),
                    fake.date_time_this_year(),
                    fake.date_time_this_year()
                ))
            print("✅ Saved searches inserted successfully!")
        except Exception as e:
            # Rollback only to the savepoint, not the entire transaction
            cur.execute("ROLLBACK TO SAVEPOINT saved_searches_savepoint")
            print(f"Warning: Could not insert saved searches (pgvector not available): {e}")
            print("Continuing with other tables...")
        
        # Commit all changes
        conn.commit()
        print("🎉 All sample data inserted successfully!")
        
    except Exception as e:
        print(f"❌ Error inserting sample data: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    print("🚀 Starting sample data insertion...")
    insert_sample_data()
    print("🎉 Sample data insertion script completed!")