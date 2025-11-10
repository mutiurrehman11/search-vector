import psycopg2
import re

# Database connection configuration
db_config = {
    'host': '167.86.115.58',
    'port': 5432,
    'dbname': 'prospects_dev',
    'user': 'devuser',
    'password': 'testdev123'
}

def apply_schema():
    """Connects to the database and applies the schema from a file."""
    conn = None
    try:
        # Connect to the database
        conn = psycopg2.connect(**db_config)
        print("Database connection successful.")

        with conn.cursor() as cur:
            # Drop the players table if it exists
            print("Dropping players table...")
            cur.execute("DROP TABLE IF EXISTS players CASCADE;")
            print("Players table dropped.")

            # Read the schema file
            with open('Database/schema.sql', 'r') as f:
                schema_sql = f.read()

            # Extract the CREATE TABLE players statement
            players_table_sql = re.search(r"CREATE TABLE players \((.*?)\);", schema_sql, re.DOTALL).group(0)

            # Execute the schema SQL
            print("Applying schema...")
            cur.execute(players_table_sql)
            conn.commit()
            print("Schema applied successfully.")

    except psycopg2.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    apply_schema()