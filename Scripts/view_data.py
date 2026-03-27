import psycopg2
import os
import json
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

load_dotenv()

def get_db_connection():
    """Get database connection"""
    db_uri = os.getenv('DATABASE_URL')
    
    if db_uri:
        return psycopg2.connect(db_uri)
    else:
        # Fallback to default credentials if DATABASE_URL is not set
        return psycopg2.connect(
            host='167.86.115.58',
            port=5432,
            dbname='prospects_dev',
            user='devuser',
            password='testdev123'
        )

def view_all_tables(conn):
    """Fetches data from all tables and saves it to a JSON file."""
    print("Fetching data from all tables...")
    
    tables = [
        'player_engagement_events'
    ]

    all_data = {}

    # Iterate through each table and fetch its content
    for table_name in tables:
        print(f"Fetching data from {table_name.upper()}...")
        
        try:
            # Use RealDictCursor to get results as dictionaries
            with conn.cursor(cursor_factory=RealDictCursor) as dict_cur:
                dict_cur.execute(f"SELECT * FROM {table_name};")
                rows = dict_cur.fetchall()
                print(f"  - Fetched {len(rows)} rows from {table_name.upper()}.")
                
                # Convert datetime objects to strings for JSON serialization
                for row in rows:
                    for key, value in row.items():
                        if hasattr(value, 'isoformat'):
                            row[key] = value.isoformat()

                all_data[table_name] = rows

        except psycopg2.Error as e:
            print(f"Error fetching data from {table_name}: {e}")
            conn.rollback() # Rollback the transaction on error

    # Write the data to a JSON file
    output_path = os.path.join(os.path.dirname(__file__), '..', 'database_data.json')
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=4)
    
    print(f"\nData saved to {output_path}")

def main():
    """Main function to view data."""
    conn = None
    try:
        conn = get_db_connection()
        print("✓ Connected to database\n")
        
        view_all_tables(conn)
        
    except psycopg2.Error as e:
        print(f"Database connection failed: {e}")
    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

if __name__ == "__main__":
    main()