import re
import sqlite3
import time

import streamlit as st
from pandas import DataFrame

from agents import sqlQueryGeneratorAgent, sqlQueryAccuracyJudgeAgent
from fetch_examples import getTopSentenceMatches, fetchUsedTables, fetchSentencePairing, fetchTableSchemas, \
    loadSentences

# Page config
st.set_page_config(page_title="Pydantic AI Chatbot", layout="wide")
st.title("🤖 Pydantic AI Chatbot")
st.sidebar.markdown("### Chat History")
clearButton = st.sidebar.button("Clear Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "previous_queries" not in st.session_state:
    st.session_state.previous_queries = []
if "questions_by_db" not in st.session_state:
    st.session_state.questions_by_db = {}

if clearButton:
    st.session_state.messages = []
    st.session_state.previous_queries = []
    st.session_state.questions_by_db = {}

dbOptions: dict = {
    "Bakery Data": {
        "path": 'databases/bakery_1/bakery_1.sqlite',
        "description": "Database containing information about a bakery's operations, including products, sales, customers, and transactions.",
        "db_id": "bakery_1"
    },
    "Concert Singer": {
        "path": 'databases/concert_singer/concert_singer.sqlite',
        "description": "Database containing information on concerts, singers, songs, and performances.",
        "db_id": "concert_singer"
    },
    "Real Estate Properties": {
        "path": 'databases/real_estate_properties/real_estate_properties.sqlite',
        "description": "Database containing real estate listings, property details, owners, and agents.",
        "db_id": "real_estate_properties"
    },
    "Tennis tournaments": {
        "path": 'databases/wta_1/wta_1.sqlite',
        "description": "Database containing information on WTA tournaments, including players, rankings, and matches.",
        "db_id": "wta_1"
    }
}
dbDisplayOptions: list = [f"{name} - {info['description']}" for name, info in dbOptions.items()]
selectedDbDisplay = st.selectbox("Select Database", dbDisplayOptions)
selectedDb = next(key for key, value in dbOptions.items() if f"{key} - {value['description']}" == selectedDbDisplay)
selectedDbId = dbOptions[selectedDb]['db_id']

# Initialize questions for the selected database
if selectedDbId not in st.session_state.questions_by_db:
    st.session_state.questions_by_db[selectedDbId] = []

# Sidebar: Display questions for the selected database
st.sidebar.markdown("### Asked Questions")
for question in st.session_state.questions_by_db[selectedDbId]:
    if st.sidebar.button(question, key=question):
        st.session_state.selected_question = question  # Store the selected question in session state

# Check if a question is selected and process it
if "selected_question" in st.session_state and st.session_state.selected_question:
    userInput = st.session_state.selected_question  # Set the selected question as the current input
    st.session_state.selected_question = None  # Reset after processing

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
userInput = st.chat_input("Ask me anything...")

if userInput:
    # Save the current question to the session state
    if userInput not in st.session_state.questions_by_db[selectedDbId]:
        st.session_state.questions_by_db[selectedDbId].append(userInput)

    st.session_state.messages.append({"role": "user", "content": userInput})

    with st.chat_message("user"):
        st.markdown(userInput)

    # Fetch example queries and table schemas
    sentences = loadSentences(selectedDbId)
    sentenceMatches = getTopSentenceMatches(userInput, sentences)
    example_queries = fetchSentencePairing(sentenceMatches, selectedDbId)
    tables = fetchUsedTables(sentenceMatches, selectedDbId)
    table_schemas = fetchTableSchemas(tables, selectedDbId)

    with st.chat_message("assistant"):
        st.markdown(f"Example Queries: {example_queries}")
        st.markdown(f"Used tables: {', '.join(tables)}")

        # Generate and display 5 SQL queries
        for i in range(5):
            retries = 3
            while retries > 0:
                generated_query = sqlQueryGeneratorAgent.run_sync(userInput, deps={
                    "prompt_query_pairings": example_queries,
                    "table_schemas": table_schemas,
                    "previous_queries": st.session_state.previous_queries
                })
                cleaned_query = re.sub(r'^sql\s*|$', '', generated_query.data).strip()
                accuracy_response = sqlQueryAccuracyJudgeAgent.run_sync(f"Generated Query: {cleaned_query}",
                                                                        deps={"original_prompt": userInput})

                # Parse the response
                match_result = re.match(r'(\d\.\d+)\s*\|\s*\((.*)\)', accuracy_response.data)
                if match_result:
                    accuracy_score = float(match_result.group(1))
                    feedback = match_result.group(2)
                else:
                    accuracy_score = float(accuracy_response.data)
                    feedback = ""

                if accuracy_score >= 0.6:
                    try:
                        conn = sqlite3.connect(dbOptions[selectedDb]['path'])
                        cursor = conn.cursor()

                        # Measure execution time
                        start_time = time.time()
                        cursor.execute(cleaned_query)
                        results = cursor.fetchall()
                        execution_time = time.time() - start_time

                        conn.close()

                        # Convert results to DataFrame and display
                        df = DataFrame(results, columns=[desc[0] for desc in cursor.description])

                        st.markdown("### Query Results")
                        st.code(cleaned_query, language='sql')
                        st.dataframe(df.head())

                        st.markdown(f"Accuracy Score: {accuracy_score}")
                        st.markdown(f"Execution Time: {execution_time:.2f} seconds")
                        st.session_state.previous_queries.append(cleaned_query)
                        break
                    except Exception as e:
                        # Pass the error message back into the re-generating process
                        userInput = f"{userInput}\nThe query '{cleaned_query}' failed with error: {e}. Please fix the issues and regenerate."
                        retries -= 1
                else:
                    retries -= 1
                    userInput = f"{userInput}\nThe query '{cleaned_query}' is not accurate. Please fix the issues: {feedback} and regenerate."

            if retries == 0:
                st.markdown("Failed to generate a valid query after 3 attempts.")

        # Store response in session state
        st.session_state.messages.append({"role": "assistant", "content": "hello!"})