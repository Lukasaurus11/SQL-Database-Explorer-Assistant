from os import environ

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

load_dotenv()

model = OpenAIModel(
    # "gpt-4-turbo",
    "gpt-3.5-turbo",
    api_key=environ['OPENAIKEY']
)

sqlQueryGeneratorAgent: Agent = Agent(
    model,
    name="SQL Query Generator",
    system_prompt="You are in charge of generating an SQL query and only an SQL query, where the results should all be "
                  "returned in a text format, avoiding completely using ```sql ``` in the output."
)


# Remove the previously generated queries section to prove that the same query would be generated 5 times
@sqlQueryGeneratorAgent.system_prompt
def system_prompt(ctx) -> str:
    error = ctx.deps.get('error')  # Safely get the 'error' key
    previous_queries = ctx.deps.get('previous_queries', [])
    previous_queries_str = "\n".join(previous_queries)

    if error:
        return f"""
        To generate an SQL query, this is the prompt-query pairings: {ctx.deps['prompt_query_pairings']}
        and the table schemas are: {ctx.deps['table_schemas']}.

        While executing the previously generated query: {ctx.deps['previous_query']}, an error occurred: {error}.
        Please fix the issues and regenerate the SQL query. The error message is: {error}.

        Please generate a new SQL query that is different from the previously generated queries. Ensure the new query uses
        different SQL constructs, joins, or conditions to achieve the same result.
        {previous_queries_str}
        """
    else:
        return f"""
        To generate an SQL query, this is the prompt-query pairings: {ctx.deps['prompt_query_pairings']}
        and the table schemas are: {ctx.deps['table_schemas']}.

        Please generate a new SQL query that is different from the previously generated queries. Ensure the new query uses
        different SQL constructs, joins, or conditions to achieve the same result.
        {previous_queries_str}
        """


sqlQueryAccuracyJudgeAgent: Agent = Agent(
    model,
    name="SQL Query Accuracy Judge",
    system_prompt="You are in charge of judging the accuracy of the following SQL query. Please return only an accuracy "
                  "score from 0 to 1, and skip all the reasoning and explanation"
)


@sqlQueryAccuracyJudgeAgent.system_prompt
def system_prompt(ctx) -> str:
    return f"""
    You are an SQL expert and you are in charge of judging the accuracy of the following SQL query.
    The original user-based question to tackle is: {ctx.deps['original_prompt']}
    YOU CAN ONLY return the accuracy score from 0 to 1, WITHOUT ANY EXTRA CHARACTERS. If the score is below 0.6, also 
    provide the components that are lacking in the format: score | (things to adjust).
    """
