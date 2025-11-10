import psycopg2, os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host='167.86.115.58',
    port=5432,
    dbname='prospects_dev',
    user='devuser',
    password='testdev123'
)
cur = conn.cursor()

tables = ['players', 'player_engagement_events', 'saved_searches', 'tenants', 'soccer_positions']

for table in tables:
    print(f"Schema for table: {table}")
    cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}';")
    rows = cur.fetchall()
    if not rows:
        print(f"  Table '{table}' not found.")
    for row in rows:
        print(f"  {row[0]}: {row[1]}")

cur.close()
conn.close()