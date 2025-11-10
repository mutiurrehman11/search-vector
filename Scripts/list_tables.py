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
cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name;")
tables = [r[0] for r in cur.fetchall()]
print('Existing tables:', tables)
cur.close(); conn.close()