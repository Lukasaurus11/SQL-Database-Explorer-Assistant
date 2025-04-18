import json
import re


def analyzeQueryComplexity(sql_query):
    """
    Analyzes the complexity of a SQL query and labels it as simple, complex, or multi-table.

    Parameters:
        sql_query (str): The SQL query to analyze.

    Returns:
        dict: A dictionary containing the analysis results:
            - num_tables (int): Number of tables in the query.
            - has_subqueries (bool): Whether the query contains subqueries.
            - has_aggregates (bool): Whether the query contains aggregate functions or GROUP BY.
            - complexity (str): The complexity label ("simple", "complex", "multi-table").
    """
    # Normalize the query (remove extra spaces and convert to lowercase)
    normalized_query = re.sub(r'\s+', ' ', sql_query.strip()).lower()

    # Count the number of tables (look for FROM and JOIN keywords)
    tables = re.findall(r'\bfrom\b\s+([a-zA-Z0-9_]+)|\bjoin\b\s+([a-zA-Z0-9_]+)', normalized_query)
    num_tables = len(tables)

    # Check for subqueries (look for SELECT within parentheses)
    has_subqueries = bool(re.search(r'\(\s*select', normalized_query))

    # Check for aggregate functions or GROUP BY
    has_aggregates = bool(re.search(r'\b(group by|sum\(|avg\(|count\(|min\(|max\()\b', normalized_query))

    # Determine complexity
    if num_tables == 1 and not has_subqueries and not has_aggregates:
        complexity = "simple"
    elif num_tables > 1 or has_subqueries:
        complexity = "multi-table"
    else:
        complexity = "complex"

    return {
        "num_tables": num_tables,
        "has_subqueries": has_subqueries,
        "has_aggregates": has_aggregates,
        "complexity": complexity
    }


if __name__ == "__main__":
    # sqlFiles = [
    #    "tests/regeneration/regen_1_gpt_3_5.json",
    #    "tests/regeneration/regen_5_gpt_3_5.json",
    #    "tests/regeneration/regen_10_gpt_3_5.json",
    #    "tests/regeneration/regen_19_gpt_3_5.json",
    #    "tests/regeneration/regen_42_gpt_3_5.json",
    # ]
    sqlFiles = [
        "tests/base/base_1_gpt3_5.json",
        "tests/base/base_5_gpt3_5.json",
        "tests/base/base_10_gpt3_5.json",
        "tests/base/base_19_gpt3_5.json",
        "tests/base/base_42_gpt3_5.json",
    ]

    results: dict = {
        "simple": {
            "failed": 0,
            "passed": 0,
        },
        "complex": {
            "failed": 0,
            "passed": 0,
        },
        "multi-table": {
            "failed": 0,
            "passed": 0,
        },
    }

    correctlyGeneratedQueries: dict = {
        "simple": {
            "failed": 0,
            "passed": 0,
        },
        "complex": {
            "failed": 0,
            "passed": 0,
        },
        "multi-table": {
            "failed": 0,
            "passed": 0,
        },
    }

    for sqlFile in sqlFiles:
        with open(sqlFile) as json_file:
            data = json.load(json_file)

            for test in data:
                testQuery = test["test"]
                generatedQuery = test["generated"]

                if not generatedQuery:
                    print(f"Failed to generate a query equivalent to the test case: {testQuery}")
                    continue

                result = test["match"]

                testComplexity = analyzeQueryComplexity(testQuery)
                generatedComplexity = analyzeQueryComplexity(generatedQuery)

                if testComplexity["complexity"] != generatedComplexity["complexity"]:
                    correctlyGeneratedQueries[testComplexity["complexity"]]["failed"] += 1

                else:
                    correctlyGeneratedQueries[testComplexity["complexity"]]["passed"] += 1

                results[generatedComplexity["complexity"]]["passed"] += 1 if result else 0
                results[generatedComplexity["complexity"]]["failed"] += 1 if not result else 0

    # Merging results and correctlyGeneratedQueries into a single dictionary
    mergedResults = {}

    for complexity in results.keys():
        mergedResults[complexity] = {
            "results": {
                "failed": results[complexity]["failed"],
                "passed": results[complexity]["passed"],
            },
            "correctlyGeneratedQueries": {
                "failed": correctlyGeneratedQueries[complexity]["failed"],
                "passed": correctlyGeneratedQueries[complexity]["passed"],
            },
        }

    # Example of saving the merged dictionary to a JSON file
    with open("tests/base/base_query_analysis_gpt3_5_results.json", "w") as outfile:
        json.dump(mergedResults, outfile, indent=4)
