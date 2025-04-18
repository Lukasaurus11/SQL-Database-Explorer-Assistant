from os import environ

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

load_dotenv()

model: OpenAIModel = OpenAIModel(
    "gpt-4-turbo",
    # "gpt-3.5-turbo",
    api_key=environ['OPENAIKEY']
)

sqlQueryGeneratorAgent: Agent = Agent(
    model,
    name="SQL Query Generator",
)

# Remove the previously generated queries section to prove that the same query would be generated 5 times
@sqlQueryGeneratorAgent.system_prompt
def system_prompt(ctx) -> str:
    error: None | str = ctx.deps.get('error')
    failedQuery: None | str = ctx.deps.get('failed_query')

    previousQueries: list = ctx.deps.get('previous_queries', [])
    previousQueriesStr: str = "\n".join(previousQueries)

    if error:
        return f"""
        You are an expert SQL assistant. Your job is to generate correct, syntactically valid, and semantically accurate SQL queries.
        Only return the SQL query, without any additional text or explanation. Avoid using ```sql ``` or any other code formatting.
    
        CONTEXT:
        - Prompt-query examples: {ctx.deps['prompt_query_pairings']}
        - Table schemas: {ctx.deps['table_schemas']}
        - The last query that failed: {failedQuery}
        - Error message: {error}
    
        GUIDELINES:
        1. Fix the issue from the last query based on the error.
        2. Do NOT generate any queries that are the same or semantically similar to any of the following:
        {previousQueriesStr}
        3. If needed, reformulate the query to avoid repeating structure or logic while still satisfying the prompt.
    
        Now regenerate a new SQL query that fulfills the original user request, taking the above into account.
        """
    else:
        return f"""
        You are an expert SQL assistant. Your job is to generate correct and semantically accurate SQL queries.
        Only return the SQL query, without any additional text or explanation. Avoid using ```sql ``` or any other code formatting.
    
        CONTEXT:
        - Prompt-query examples: {ctx.deps['prompt_query_pairings']}
        - Table schemas: {ctx.deps['table_schemas']}
    
        GUIDELINES:
        1. Generate a new SQL query that is NOT the same or semantically similar to any of the following:
        {previousQueriesStr}
        2. Reformulate the logic or structure if needed, while satisfying the prompt requirements.
    
        Now generate a new SQL query based on the latest prompt.
        """


sqlQueryAccuracyJudgeAgent: Agent = Agent(
    model,
    name="SQL Query Accuracy Judge",
)

@sqlQueryAccuracyJudgeAgent.system_prompt
def system_prompt(ctx) -> str:
    return f"""
    You are an SQL expert and you are in charge of judging the accuracy of the following SQL query.
    The original user-based question to tackle is: {ctx.deps['original_prompt']}
    YOU CAN ONLY return the accuracy score from 0 to 1, WITHOUT ANY EXTRA CHARACTERS. If the score is below 0.6, also 
    provide the components that are lacking in the format: score | (things to adjust).
    """
