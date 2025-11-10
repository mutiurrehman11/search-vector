#!/usr/bin/env python3
"""
Create the database schema from the schema.sql file
"""
import psycopg2
import os
from dotenv import load_dotenv
import sys

# Prefer centralized app Config for DB settings if available
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app import Config as AppConfig
except Exception:
    AppConfig = None

# Load environment variables
load_dotenv()

def create_schema():
    """Create the database schema"""
    try:
        # Database configuration: prefer app Config, fallback to env/hardcoded
        if AppConfig is not None:
            if getattr(AppConfig, 'DB_URI', None):
                conn = psycopg2.connect(AppConfig.DB_URI)
            elif getattr(AppConfig, 'DB_CONFIG', None):
                conn = psycopg2.connect(**AppConfig.DB_CONFIG)
            else:
                conn = psycopg2.connect(
                    host=os.getenv('DB_HOST', 'localhost'),
                    port=int(os.getenv('DB_PORT', 5432)),
                    dbname=os.getenv('DB_NAME', 'player_search'),
                    user=os.getenv('DB_USER', 'postgres'),
                    password=os.getenv('DB_PASSWORD', 'password')
                )
        else:
            conn = psycopg2.connect(
                host='localhost',
                port=5432,
                dbname='prospects_dev',
                user='devuser',
                password='testdev123'
            )
        cursor = conn.cursor()

        print("Applying database schema...")
        with open('Database/schema.sql', 'r', encoding='utf-8') as f:
            cursor.execute(f.read())
        print("Schema applied successfully.")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create schema: {e}")
        return False

if __name__ == "__main__":
    success = create_schema()
    if not success:
        exit(1)