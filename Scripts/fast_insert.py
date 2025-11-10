import psycopg2
from psycopg2.extras import execute_batch
import json
import random
from datetime import datetime, date
from faker import Faker
import numpy as np
import os
from dotenv import load_dotenv
import string
import argparse
from tqdm import tqdm

load_dotenv()

DB_CONFIG = {
    'host': '167.86.115.58',
    'port': 5432,
    'dbname': 'prospects_dev',
    'user': 'devuser',
    'password': 'testdev123'
}

fake = Faker()

def generate_ulid():
    """Generate a ULID-like ID (26 characters)"""
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choices(chars, k=26))

def generate_vector_embedding():
    """Generate a random 128-dimensional vector for embeddings"""
    return np.random.normal(0, 1, 128).tolist()

def insert_data(scaling_factor=1, truncate=False):
    """Insert sample data into all tables"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        print("Connected to database successfully!")

        if truncate:
            print("Truncating tables...")
            tables = [
                'player_engagement_events', 'player_engagement_stats', 'player_embeddings', 
                'player_goals', 'player_improvements', 'player_skills', 'player_teams', 
                'player_training_sessions', 'players', 'chat_messages', 'chat_rooms', 
                'divisions', 'categories', 'skills', 'playlists', 'follows', 'views', 
                'soccer_positions', 'cities', 'states', 'countries', 'tenants', 'saved_searches'
            ]
            for table in tables:
                cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;")
            conn.commit()
            print("Tables truncated successfully!")

        # Insert Tenants
        print("Inserting tenants...")
        tenants_data = [(generate_ulid(), fake.company(), fake.slug(), fake.text(max_nb_chars=100)) for _ in range(5)]
        execute_batch(cur, "INSERT INTO tenants (id, name, slug, description) VALUES (%s, %s, %s, %s)", tenants_data)
        conn.commit()
        cur.execute("SELECT id FROM tenants")
        tenant_ids = [row[0] for row in cur.fetchall()]
        print("✅ Tenants inserted successfully!")

        # Insert Players
        print("Inserting players...")
        positions = ['FWD', 'MID', 'DEF', 'GK']
        statuses = ['active', 'inactive', 'pending']
        genders = ['male', 'female', 'other']
        players_data = []
        for _ in tqdm(range(100 * scaling_factor), desc="Generating Players"):
            players_data.append((
                generate_ulid(), fake.first_name(), fake.last_name(), 
                json.dumps({'city': fake.city(), 'state': fake.state(), 'country': fake.country()}),
                fake.date_of_birth(minimum_age=16, maximum_age=40),
                random.choice(statuses), random.choice(genders),
                random.choice(tenant_ids), random.choice(positions)
            ))
        
        execute_batch(cur, """
            INSERT INTO players (id, first_name, last_name, location, birth_date, status, gender, tenant_id, position_code)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, players_data)
        conn.commit()
        cur.execute("SELECT id FROM players")
        player_ids = [row[0] for row in cur.fetchall()]
        print("✅ Players inserted successfully!")

        # Insert Player Embeddings
        print("Inserting player embeddings...")
        player_embeddings_data = []
        for player_id in tqdm(player_ids, desc="Generating Embeddings"):
            player_embeddings_data.append((player_id, generate_vector_embedding()))
        
        execute_batch(cur, "INSERT INTO player_embeddings (player_id, embedding) VALUES (%s, %s)", player_embeddings_data)
        conn.commit()
        print("✅ Player embeddings inserted successfully!")

        print("✅ All sample data inserted successfully!")
        
    except Exception as e:
        print(f"❌ Error inserting sample data: {e}")
        import traceback
        traceback.print_exc()
        if 'conn' in locals() and conn:
            conn.rollback()
            
    finally:
        if 'cur' in locals() and cur:
            cur.close()
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Insert sample data into the database.')
    parser.add_argument('--scaling_factor', type=int, default=1, help='Factor to scale the amount of data generated.')
    parser.add_argument('--truncate', action='store_true', help='Truncate tables before inserting data.')
    args = parser.parse_args()

    print("🚀 Starting sample data insertion...")
    insert_data(args.scaling_factor, args.truncate)
    print("🎉 Sample data insertion script completed!")