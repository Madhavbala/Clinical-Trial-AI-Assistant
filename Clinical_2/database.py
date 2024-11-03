from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import streamlit as st

def configure_db(mysql_host, mysql_port, mysql_user, mysql_password, mysql_db):
    if not (mysql_host and mysql_user and mysql_password and mysql_db):
        st.error("Please provide all MySQL connection details.")
        st.stop()
    
    try:
        connection_string = f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_db}"
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            if result.fetchone() is None:
                raise Exception("Failed to execute test query.")
        
        return engine
    except SQLAlchemyError as e:
        st.error(f"SQLAlchemy Error: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.stop()
