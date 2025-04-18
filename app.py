import sqlite3
from re import match as re_match, sub as re_sub
from time import time
from typing import Match

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

st.session_state.messages = []

if "messages" not in st.session_state or not st.session_state.get("messages"):
    st.session_state.messages = []
if "previous_queries" not in st.session_state:
    st.session_state.previous_queries = []

if clearButton:
    st.session_state.messages = []

# Database options
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

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
userInput = st.chat_input("Ask me anything...")

if userInput:
    st.session_state.messages.append({"role": "user", "content": userInput})
    previouslyGeneratedQueries: list = []

    with st.chat_message("user"):
        st.markdown(userInput)

    # Fetch example queries and table schemas
    sentences = loadSentences(selectedDbId)
    sentenceMatches = getTopSentenceMatches(userInput, sentences)
    exampleQueries = fetchSentencePairing(sentenceMatches, selectedDbId)
    tables = fetchUsedTables(sentenceMatches, selectedDbId)
    tableSchemas = fetchTableSchemas(tables, selectedDbId)

    error: None | str = None
    failedQuery: None | str = None

    with st.chat_message("assistant"):
        # Generate and display 5 SQL queries
        for _ in range(5):
            retries: int = 3
            while retries > 0:
                generatedQuery = sqlQueryGeneratorAgent.run_sync(userInput, deps={
                    "prompt_query_pairings": exampleQueries,
                    "table_schemas": tableSchemas,
                    "previous_queries": previouslyGeneratedQueries,
                    "error": error if error else None,
                    "failed_query": failedQuery if failedQuery else None
                })

                # Reset error
                error = None

                cleanedQuery: str = re_sub(r'^sql\s*|$', '', generatedQuery.data).strip()
                accuracyResponse = sqlQueryAccuracyJudgeAgent.run_sync(f"Generated Query: {cleanedQuery}",
                                                                       deps={"original_prompt": userInput})

                # Parse the response
                matchResult: Match[str] | None = re_match(r'(\d\.\d+)\s*\|\s*\((.*)\)', accuracyResponse.data)
                if matchResult:
                    accuracyScore: float = float(matchResult.group(1))
                    feedback: str = matchResult.group(2)
                else:
                    accuracyScore: float = float(accuracyResponse.data)
                    feedback: str = ""

                if accuracyScore >= 0.6:
                    try:
                        conn = sqlite3.connect(dbOptions[selectedDb]['path'])
                        cursor = conn.cursor()

                        # Measure execution time
                        startTime = time()
                        cursor.execute(cleanedQuery)
                        results = cursor.fetchall()
                        executionTime = time() - startTime

                        conn.close()

                        # Convert results to DataFrame and display
                        df = DataFrame(results, columns=[desc[0] for desc in cursor.description])

                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.markdown("### Query")
                            st.code(cleanedQuery, language='sql')
                        with col2:
                            st.markdown("### Query Results")
                            st.dataframe(df.head())

                        st.markdown(f"Accuracy Score: {accuracyScore} | Execution Time: {executionTime:.2f} seconds")
                        previouslyGeneratedQueries.append(cleanedQuery)
                        break

                    # Possible SQL error
                    except Exception as e:
                        retries -= 1
                        error = f"The following SQL error occurred when executing the query: {str(e)}"
                        failedQuery = cleanedQuery

                # Confidence score too low
                else:
                    retries -= 1
                    error = feedback
                    failedQuery = cleanedQuery

            if retries == 0:
                st.markdown("Failed to generate a valid query after 3 attempts.")
