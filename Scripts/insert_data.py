"""
Insert Sample Data Script
Inserts realistic sample data for testing ML model training and embeddings
"""

import psycopg2
import os
import random
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import string

load_dotenv()

def generate_ulid():
    """Generate a simple ULID-like ID (26 characters)"""
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(26))

def get_db_connection():
    """Get database connection"""
    db_uri = os.getenv('DATABASE_URL')
    
    if db_uri:
        return psycopg2.connect(db_uri)
    else:
        return psycopg2.connect(
            host='167.86.115.58',
            port=5432,
            dbname='prospects_dev',
            user='devuser',
            password='testdev123'
        )

def insert_sample_players(conn, count=100):
    """Insert sample players"""
    print(f"Inserting {count} sample players...")
    
    first_names = ['John', 'Michael', 'David', 'James', 'Robert', 'William', 'Carlos', 'Luis', 'Diego', 'Pedro',
                   'Emma', 'Sophia', 'Olivia', 'Isabella', 'Mia', 'Charlotte', 'Amelia', 'Harper', 'Evelyn', 'Abigail']
    
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Martinez', 'Rodriguez', 'Lopez', 'Gonzalez',
                  'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Thompson', 'White']
    
    positions = ['forward', 'midfielder', 'defender', 'goalkeeper']
    
    genders = ['male', 'female']
    
    # US cities with coordinates
    cities = [
        {'name': 'New York', 'lat': 40.7128, 'lng': -74.0060, 'country': 'USA'},
        {'name': 'Los Angeles', 'lat': 34.0522, 'lng': -118.2437, 'country': 'USA'},
        {'name': 'Chicago', 'lat': 41.8781, 'lng': -87.6298, 'country': 'USA'},
        {'name': 'Houston', 'lat': 29.7604, 'lng': -95.3698, 'country': 'USA'},
        {'name': 'Miami', 'lat': 25.7617, 'lng': -80.1918, 'country': 'USA'},
        {'name': 'London', 'lat': 51.5074, 'lng': -0.1278, 'country': 'UK'},
        {'name': 'Madrid', 'lat': 40.4168, 'lng': -3.7038, 'country': 'Spain'},
        {'name': 'Barcelona', 'lat': 41.3851, 'lng': 2.1734, 'country': 'Spain'},
        {'name': 'Paris', 'lat': 48.8566, 'lng': 2.3522, 'country': 'France'},
        {'name': 'Berlin', 'lat': 52.5200, 'lng': 13.4050, 'country': 'Germany'},
    ]
    
    skills_list = ['dribbling', 'passing', 'shooting', 'speed', 'stamina', 'strength']
    
    player_ids = []
    
    with conn.cursor() as cur:
        for i in range(count):
            player_id = generate_ulid()
            player_ids.append(player_id)
            
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            
            city = random.choice(cities)
            location = json.dumps({
                'name': city['name'],
                'latitude': city['lat'],
                'longitude': city['lng'],
                'country': city['country']
            })
            
            age = random.randint(16, 35)
            birth_date = datetime.now() - timedelta(days=age*365 + random.randint(0, 365))
            
            height = random.randint(160, 195)
            weight = random.randint(60, 95)
            
            gender = random.choice(genders)
            
            description = f"Experienced {random.choice(positions)} player with {age-13} years of experience. Known for strong technical skills and game intelligence."
            
            tags = json.dumps(random.sample(['competitive', 'team-player', 'leader', 'technical', 'physical', 'tactical'], k=random.randint(2, 4)))
            
            availability = json.dumps(random.sample(['weekday_morning', 'weekday_afternoon', 'weekday_evening', 'weekend_morning', 'weekend_afternoon'], k=random.randint(2, 4)))
            
            profile_picture = json.dumps({
                'url': f'https://randomuser.me/api/portraits/{"men" if gender == "male" else "women"}/{i}.jpg'
            })
            
            # Insert player
            cur.execute("""
                INSERT INTO players 
                (id, first_name, last_name, location, birth_date, height, weight, gender, 
                 status, description, tags, availability, profile_picture, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                player_id, first_name, last_name, location, birth_date.date(), 
                height, weight, gender, 'active', description, tags, availability, 
                profile_picture, datetime.now(), datetime.now()
            ))
            
            # Insert player skills
            for skill in skills_list:
                level = random.randint(40, 95)
                cur.execute("""
                    INSERT INTO player_skills (player_id, skill, level, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s)
                """, (player_id, skill, level, datetime.now(), datetime.now()))
            
            # Insert player position
            position = random.choice(positions)
            playable_id = generate_ulid()
            cur.execute("""
                INSERT INTO player_teams 
                (player_id, playable_type, playable_id, position, start_at, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                player_id, 'Team', playable_id, position, 
                datetime.now() - timedelta(days=random.randint(30, 365)),
                datetime.now(), datetime.now()
            ))
        
        conn.commit()
    
    print(f"✓ Inserted {count} players with skills and positions\n")
    return player_ids

def insert_engagement_events(conn, player_ids, user_count=20):
    """Insert sample engagement events for ML training"""
    print(f"Inserting engagement events for {user_count} users...")
    
    event_types = ['impression', 'profile_view', 'follow', 'message', 'save_to_playlist']
    event_weights = [50, 30, 10, 5, 5]  # More impressions, fewer conversions
    
    positions = ['forward', 'midfielder', 'defender', 'goalkeeper']
    
    total_events = 0
    
    with conn.cursor() as cur:
        # Generate events for each user
        for user_num in range(user_count):
            user_id = generate_ulid()
            
            # Each user has 20-50 interactions
            num_events = random.randint(20, 50)
            
            for _ in range(num_events):
                player_id = random.choice(player_ids)
                event_type = random.choices(event_types, weights=event_weights)[0]
                
                # Generate query context
                query_context = {
                    'position': random.choice(positions),
                    'min_skill': random.randint(40, 70),
                    'max_skill': random.randint(70, 95),
                    'min_age': random.randint(16, 25),
                    'max_age': random.randint(25, 35),
                    'tags': random.sample(['competitive', 'team-player', 'technical'], k=random.randint(1, 2))
                }
                
                created_at = datetime.now() - timedelta(days=random.randint(0, 30))
                
                cur.execute("""
                    INSERT INTO player_engagement_events 
                    (user_id, player_id, event_type, query_context, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    user_id, player_id, event_type, 
                    json.dumps(query_context), created_at
                ))
                
                total_events += 1
        
        conn.commit()
    
    print(f"✓ Inserted {total_events} engagement events\n")

def insert_engagement_stats(conn, player_ids):
    """Insert engagement statistics for players"""
    print("Inserting engagement statistics...")
    
    with conn.cursor() as cur:
        for player_id in player_ids:
            # Generate realistic engagement stats
            ctr = random.uniform(0.05, 0.25)
            recent_activity = random.uniform(0.3, 0.9)
            follow_rate = random.uniform(0.01, 0.10)
            message_rate = random.uniform(0.005, 0.05)
            
            cur.execute("""
                INSERT INTO player_engagement_stats 
                (player_id, ctr, recent_activity_score, follow_rate, message_rate, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                player_id, ctr, recent_activity, follow_rate, message_rate, datetime.now()
            ))
        
        conn.commit()
    
    print(f"✓ Inserted engagement stats for {len(player_ids)} players\n")

def verify_data(conn):
    """Verify inserted data"""
    print("Verifying inserted data...")
    
    with conn.cursor() as cur:
        # Count players
        cur.execute("SELECT COUNT(*) FROM players;")
        player_count = cur.fetchone()[0]
        print(f"  Players: {player_count}")
        
        # Count skills
        cur.execute("SELECT COUNT(*) FROM player_skills;")
        skills_count = cur.fetchone()[0]
        print(f"  Player skills: {skills_count}")
        
        # Count positions
        cur.execute("SELECT COUNT(*) FROM player_teams;")
        teams_count = cur.fetchone()[0]
        print(f"  Player teams: {teams_count}")
        
        # Count engagement events
        cur.execute("SELECT COUNT(*) FROM player_engagement_events;")
        events_count = cur.fetchone()[0]
        print(f"  Engagement events: {events_count}")
        
        # Count engagement stats
        cur.execute("SELECT COUNT(*) FROM player_engagement_stats;")
        stats_count = cur.fetchone()[0]
        print(f"  Engagement stats: {stats_count}")
        
        # Check event type distribution
        cur.execute("""
            SELECT event_type, COUNT(*) 
            FROM player_engagement_events 
            GROUP BY event_type 
            ORDER BY COUNT(*) DESC;
        """)
        print("\n  Event type distribution:")
        for row in cur.fetchall():
            print(f"    {row[0]}: {row[1]}")
    
    print("\n✓ Data verification complete\n")

def main():
    """Main data insertion function"""
    print("=" * 60)
    print("Sample Data Insertion")
    print("=" * 60)
    print()
    
    try:
        # Connect to database
        conn = get_db_connection()
        print("✓ Connected to database\n")
        
        # Insert players
        player_ids = insert_sample_players(conn, count=100)
        
        # Insert engagement events
        insert_engagement_events(conn, player_ids, user_count=20)
        
        # Insert engagement stats
        insert_engagement_stats(conn, player_ids)
        
        # Verify data
        verify_data(conn)
        
        print("=" * 60)
        print("Sample data insertion completed successfully!")
        print("=" * 60)
        print("\nYou can now:")
        print("1. Generate embeddings: python generate_embeddings.py")
        print("2. Train ML model: python train_model.py")
        print("3. Test search: curl http://localhost:5000/api/v1/search")
        
    except Exception as e:
        print(f"\nData insertion failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

if __name__ == "__main__":
    main()