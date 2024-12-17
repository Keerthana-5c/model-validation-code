import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os

def execute_query(sql_file_path, output_csv):
    load_dotenv('pwd.env')
    try:
        with open(sql_file_path, 'r') as file:
            sql_query = file.read()
            print("SQL query read successfully.")

        connection = psycopg2.connect(
            host=os.getenv("AI_DB_HOST"),
            database=os.getenv("AI_DB_NAME"),
            user=os.getenv("AI_DB_USER"),
            password=os.getenv("AI_DB_PASSWORD"),
            port=os.getenv("AI_DB_PORT")
        )
        cursor = connection.cursor()
        
        cursor.execute(sql_query)
        result = cursor.fetchall()
        print("SQL query executed successfully.")
        
        colnames = [desc[0] for desc in cursor.description]
        
        df = pd.DataFrame(result, columns=colnames)
        df.to_csv(output_csv, index=False)
        print(df.head())
        print(f"Query result successfully written to {output_csv}")
    
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error: {error}")
    
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
