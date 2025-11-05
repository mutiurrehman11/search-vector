import os, psycopg2, re

def main():
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    conn.autocommit = True
    cur = conn.cursor()

    with open('schema.sql', 'r', encoding='utf-8') as f:
        sql = f.read()

    # Skip extensions - already created by superuser
    print("Extensions already created by superuser - skipping.")

    # 2. Tables (skip if already exist)
    tables = re.findall(r'CREATE TABLE\s+(\w+)', sql)
    for tbl in tables:
        cur.execute(f"SELECT to_regclass('public.{tbl}');")
        if cur.fetchone()[0] is None:
            # Find the CREATE TABLE statement and add IF NOT EXISTS
            ddl_match = re.search(r'CREATE TABLE\s+' + tbl + r'.*?(?=\n\n|\nCREATE|\Z)', sql, re.S)
            if ddl_match:
                ddl = ddl_match.group(0)
                # Add IF NOT EXISTS after CREATE TABLE
                ddl = ddl.replace('CREATE TABLE', 'CREATE TABLE IF NOT EXISTS', 1)
                try:
                    cur.execute(ddl)
                    print(f"Table {tbl} created.")
                except Exception as e:
                    print(f"Table {tbl} failed: {e}")
        else:
            print(f"Table {tbl} already exists – skipped.")

    # 3. Indexes (skip if already exist)
    indexes = re.findall(r'CREATE\s+(?:UNIQUE\s+)?INDEX\s+(\w+)', sql)
    for idx in indexes:
        cur.execute(f"SELECT to_regclass('public.{idx}');")
        if cur.fetchone()[0] is None:
            # Find the CREATE INDEX statement and add IF NOT EXISTS
            ddl_match = re.search(r'CREATE\s+(?:UNIQUE\s+)?INDEX\s+' + idx + r'.*?(?=;)', sql)
            if ddl_match:
                ddl = ddl_match.group(0) + ';'
                # Add IF NOT EXISTS after CREATE [UNIQUE] INDEX
                ddl = ddl.replace('CREATE INDEX', 'CREATE INDEX IF NOT EXISTS').replace('CREATE UNIQUE INDEX', 'CREATE UNIQUE INDEX IF NOT EXISTS')
                try:
                    cur.execute(ddl)
                    print(f"Index {idx} created.")
                except Exception as e:
                    print(f"Index {idx} failed: {e}")
        else:
            print(f"Index {idx} already exists – skipped.")

    cur.close(); conn.close()
    print("Schema check complete.")

if __name__ == '__main__':
    main()