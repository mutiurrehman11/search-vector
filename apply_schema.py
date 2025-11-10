import psycopg2
import os

def apply_schema():
    db_config = {
        'host': '167.86.115.58',
        'port': 5432,
        'dbname': 'prospects_dev',
        'user': 'devuser',
        'password': 'testdev123'
    }
    conn = psycopg2.connect(**db_config)
    conn.autocommit = True
    cursor = conn.cursor()

    # Drop existing tables in the correct order to avoid dependency issues
    cursor.execute("DROP TABLE IF EXISTS player_embeddings CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS player_engagement_stats CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS player_engagement_events CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS saved_searches CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS player_skills CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS player_teams CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS player_improvements CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS player_goals CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS player_training_sessions CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS views CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS follows CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS chat_messages CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS chat_rooms CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS playlists CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS skills CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS categories CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS divisions CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS soccer_positions CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS cities CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS states CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS countries CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS players CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS tenants CASCADE;")

    with open('Database/schema.sql', 'r') as f:
        schema = f.read()
        cursor.execute(schema)

    conn.close()

if __name__ == '__main__':
    apply_schema()