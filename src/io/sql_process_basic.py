import os
import configparser

import pandas as pd
import psycopg2

from src.io.path_definition import get_file


def get_db_connection_psql(database):
    """
    Establishes a PostgreSQL database connection based on the provided database identifier.

    Args:
        database (str): The database identifier, which may include environment-specific information.

    Returns:
        psycopg2.extensions.connection: A connection to the PostgreSQL database.

    This function reads the database connection credentials from a configuration file or environment variables
    based on the provided 'database' identifier. It then establishes a connection to the PostgreSQL database and
    returns the connection object.

    Example:
        >>> db_connection = get_db_connection_psql(database='cosnova_dwh')
        # Returns a connection object to the 'cosnova_dwh_prod' PostgreSQL database.

    Note:
        Ensure that the 'get_file' function is correctly defined and accessible, and the necessary credentials are
        available either in a configuration file or as environment variables based on the provided 'database'
        identifier.
    """
    credential_file = get_file("config/credentials.ini")

    if os.path.exists(credential_file):
        config = configparser.ConfigParser()
        config.read(credential_file)
        connection = psycopg2.connect(**config[database])
    else:
        port = os.environ['PORT']
        if database == 'datascience':
            user = os.environ['BDX_USER']
            password = os.environ['BDX_PASSWORD']
            host = os.environ['BDX_DBSERVER']
        else:
            user = os.environ['USER']
            password = os.environ['PASSWORD']
            host = os.environ['DBSERVER']

        connection = psycopg2.connect(database=database,
                                      user=user,
                                      password=password,
                                      host=host,
                                      port=port)
    return connection


def load_sql_df(database, query) -> pd.DataFrame:
    """
    Loads data from a SQL database into a DataFrame based on the provided environment and query.

    Args:
        database (str): The database environment, either 'bdx' or a specific environment identifier.
        query (str): The SQL query to retrieve data from the database.

    Returns:
        pd.DataFrame: A DataFrame containing the data retrieved from the SQL database.

    This function establishes a connection to the appropriate SQL database based on the 'env' parameter and executes
    the provided SQL query to fetch data. The fetched data is then converted into a DataFrame, and the resulting
    DataFrame is returned.

    Example:
        >>> sql_query = "SELECT * FROM my_table WHERE condition=True;"
        >>> df_from_sql = load_sql_df(database=database, query=sql_query)
        # Returns a DataFrame containing data from the SQL query executed in the 'prod' environment.

    Note:
        Ensure that the 'get_db_connection_psql' function is correctly defined and accessible, and the 'env' parameter
        is either 'bdx' or a specific environment identifier.
    """
    con_datazeit = get_db_connection_psql(database)

    with con_datazeit.cursor() as db_cursor:
        db_cursor.execute(query)
        data = db_cursor.fetchall()
        columns = db_cursor.description
        columns = [c.name for c in columns]

        df = pd.DataFrame(data=data, columns=columns)

    con_datazeit.close()

    return df
