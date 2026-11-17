"""
db_setup.py
-----------
Run this ONCE to create the database and table for the
Handwriting Recognition project.
"""

import mysql.connector
from config import DB_CONFIG

def setup():
    # Connect without specifying a database first
    conn = mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"]
    )
    cursor = conn.cursor()

    # Create database if not exists
    cursor.execute("CREATE DATABASE IF NOT EXISTS handwriting_db;")
    cursor.execute("USE handwriting_db;")

    # Create predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            predicted   INT            NOT NULL,
            confidence  FLOAT          NOT NULL,
            timestamp   DATETIME       DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("[DB] Database 'handwriting_db' and table 'predictions' are ready.")

if __name__ == "__main__":
    setup()
