import streamlit as st
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import urllib.parse
import pandas as pd
from database import configure_db
from prompts import (CLINICAL_TRIAL_PROMPTS, PROMPT_QUALITY_OF_LIFE, PROMPT_PLACEBO_EFFECT, PROMPT_ADVERSE_EVENT_DROPOUTS, PROMPT_DEMOGRAPHIC_IMPACT, PROMPT_DATA_INTEGRITY)

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Clinical Trial AI Assistant", page_icon="🧬")
st.title("🧬 Clinical Trial AI Assistant")

mysql_host = st.sidebar.text_input("MySQL Host", value="localhost")
mysql_port = st.sidebar.text_input("MySQL Port", value="3306")
mysql_user = st.sidebar.text_input("MySQL User")
mysql_password = st.sidebar.text_input("MySQL Password", type="password")
mysql_db = st.sidebar.text_input("MySQL Database")

mysql_password_encoded = urllib.parse.quote_plus(mysql_password)

try:
    engine = configure_db(mysql_host, mysql_port, mysql_user, mysql_password_encoded, mysql_db)
except SQLAlchemyError as e:
    st.error(f"Error connecting to the database: {e}")
    st.stop()

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", streaming=True)

db = SQLDatabase(engine)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

nav_option = st.sidebar.radio(
    "Navigation",
    ["Ask AI about Clinical Trials", "View Clinical Trial Data"]
)

if nav_option == "Ask AI about Clinical Trials":
    if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I assist you with your clinical trial queries?"}]
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_query = st.chat_input(placeholder="Ask anything about clinical trials")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            try:
                # Include all prompts in the response
                all_prompts = "\n".join(CLASSICAL_TRIAL_PROMPTS + [CLINICAL_TRIAL_PROMPTS,PROMPT_QUALITY_OF_LIFE, PROMPT_PLACEBO_EFFECT,
                                                                    PROMPT_ADVERSE_EVENT_DROPOUTS, 
                                                                    PROMPT_DEMOGRAPHIC_IMPACT, 
                                                                    PROMPT_DATA_INTEGRITY])
                full_query = f"{user_query} Here are some relevant prompts: \n{all_prompts}"
                response = agent.run(full_query)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

elif nav_option == "View Clinical Trial Data":
    st.subheader("View Clinical Trial Data")
    try:
        with engine.connect() as conn:
            tables = conn.execute(text("SHOW TABLES")).fetchall()
        
        table_names = [table[0] for table in tables]
        
        selected_table = st.selectbox("Select a clinical trial table to view", table_names)
        
        if selected_table:
            st.write(f"Showing data for table: {selected_table}")
            query = text(f"SELECT * FROM `{selected_table}`")
            with engine.connect() as conn:
                data = conn.execute(query).fetchall()
            
            columns = [col[0] for col in conn.execute(text(f"SHOW COLUMNS FROM `{selected_table}`")).fetchall()]
            df = pd.DataFrame(data, columns=columns)
            st.dataframe(df)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
