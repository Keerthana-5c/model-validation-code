from dotenv import load_dotenv
import psycopg2
import os
import datetime as dt
import json
from datetime import datetime, timedelta

dotenv_path = '.env'
load_dotenv(dotenv_path)

def initialize_postgres_connection(database_name):
    """
    Initializes a connection to a PostgreSQL database.

    Args:
        database_name (str): Name of the database.

    Returns:
        tuple: A tuple containing the connection and cursor objects.
    """
    connection = psycopg2.connect(
        host=os.getenv('PG_HOST'),
        port=os.getenv('PG_PORT'),
        user=os.getenv('PG_USER'),
        password=os.getenv('PG_PASSWORD'),
        database=database_name
    )
    cursor = connection.cursor()
    return connection, cursor

def insert_data_into_table(data, database_name, schema_name, table_name):
    """
    Inserts data into a PostgreSQL table.

    Args:
        data (dict): Data to insert.
        database_name (str): Name of the database.
        schema_name (str): Name of the schema.
        table_name (str): Name of the table.

    Returns:
        str: Success or error message.
    """
    try:
        connection, cursor = initialize_postgres_connection(database_name)
        schema_identifier = psycopg2.sql.Identifier(schema_name)
        table_identifier = psycopg2.sql.Identifier(table_name)

        if 'findings' in data:
            data['findings'] = json.dumps(data['findings'])

        # Define the SQL INSERT statement
        insert_statement = psycopg2.sql.SQL('INSERT INTO {0}.{1} ({2}) VALUES ({3})').format(
            schema_identifier,
            table_identifier,
            psycopg2.sql.SQL(', ').join(map(psycopg2.sql.Identifier, data.keys())),
            psycopg2.sql.SQL(', ').join(map(psycopg2.sql.Placeholder, data.keys()))
        )

        # Execute the SQL statement
        cursor.execute(insert_statement, data)

        # Commit the changes
        connection.commit()
        return "Successfully Inserted"
    
    except Exception as error:
        if "duplicate key value violates unique constraint" in str(error):
            return "Duplicate key error"
        else:
            return f"Insert failed: {error}"

def update_table_data(data, conditions, database_name, schema_name, table_name):
    """
    Updates data in a PostgreSQL table based on conditions.

    Args:
        data (dict): Data to update.
        conditions (dict): Conditions to match for the update.
        database_name (str): Name of the database.
        schema_name (str): Name of the schema.
        table_name (str): Name of the table.

    Returns:
        str: Success or error message.
    """
    try:
        connection, cursor = initialize_postgres_connection(database_name)
        schema_identifier = psycopg2.sql.Identifier(schema_name)
        table_identifier = psycopg2.sql.Identifier(table_name)

        update_clauses = []
        parameter_values = []

        if 'findings' in data:
            data['findings'] = json.dumps(data['findings'])

        for key, value in data.items():
            if value == "increment_by_1":
                update_clauses.append(psycopg2.sql.SQL("{} = {} + 1").format(psycopg2.sql.Identifier(key), psycopg2.sql.Identifier(key)))
                continue

            if value == "now":
                current_time = (datetime.utcnow() + timedelta(hours=5, minutes=30)).isoformat()
                parameter_values.append(current_time)
                update_clauses.append(psycopg2.sql.SQL("{} = %s").format(psycopg2.sql.Identifier(key)))
                continue

            try:
                datetime_value = datetime.fromisoformat(value)
                parameter_values.append(datetime_value.isoformat())
            except (ValueError, TypeError):
                if isinstance(value, dict):
                    parameter_values.append(json.dumps(value))
                    update_clauses.append(psycopg2.sql.SQL("{} = %s::jsonb").format(psycopg2.sql.Identifier(key)))
                else:
                    parameter_values.append(value)
                    update_clauses.append(psycopg2.sql.SQL("{} = %s").format(psycopg2.sql.Identifier(key)))

        for condition_key, condition_value in conditions.items():
            parameter_values.append(condition_value)

        update_statement = psycopg2.sql.SQL('UPDATE {0}.{1} SET {2} WHERE {3}').format(
            schema_identifier,
            table_identifier,
            psycopg2.sql.SQL(', ').join(update_clauses),
            psycopg2.sql.SQL(' AND ').join([psycopg2.sql.SQL('{} = %s').format(psycopg2.sql.Identifier(k)) for k in conditions.keys()])
        )

        cursor.execute(update_statement, parameter_values)
        connection.commit()

        return "Successfully Updated"
    
    except Exception as error:
        return f"Update failed: {error}"

def execute_sql_query(query, database_name):
    """
    Executes a SQL query on the specified PostgreSQL database.

    Args:
        query (str): SQL query to execute.
        database_name (str): Name of the database.

    Returns:
        list: Query results or error message.
    """
    try:
        connection, cursor = initialize_postgres_connection(database_name)
        cursor.execute(query)
        results = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]

        cursor.close()
        connection.close()

        query_results = []
        for row in results:
            row_dictionary = {}
            for column, value in zip(column_names, row):
                if isinstance(value, dt.date):
                    value = value.isoformat()
                row_dictionary[column] = value
            query_results.append(row_dictionary)

        if results:
            return query_results
        else:
            return "No results found"
    except Exception as error:
        print(f"{error}")
        return f"Query failed: {error}"

