import logging
import platform

from src.io.sql_process_basic import load_sql_df

logger = logging.getLogger(__name__)
system = platform.system()


def load_sellout_monthly_table_():
    # SQL query to retrieve sellout data for a specific retailer and store identifier
    query = "<SQL QUERY>"

    df = load_sql_df(database="<DATABASE NAME>", query=query)

    return df
