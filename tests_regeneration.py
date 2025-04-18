import json
import random
import re
import sqlite3
import time

from agents import sqlQueryGeneratorAgent, sqlQueryAccuracyJudgeAgent
from fetch_examples import loadSentences, getTopSentenceMatches, fetchSentencePairing, fetchUsedTables, \
    fetchTableSchemas


def split_sentences_with_ratio_single_group(sentences, test_ratio=0.2, seed=None):
    if seed is not None:
        random.seed(seed)

    # Shuffle sentences deterministically
    shuffled = sentences[:]
    random.shuffle(shuffled)

    # Split into training and testing sets
    split_index = int(len(shuffled) * (1 - test_ratio))
    train = shuffled[:split_index]
    test = shuffled[split_index:]

    return train, test


def execute_query(db_path, query):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        return results, None
    except sqlite3.Error as e:
        return None, str(e)


def regenerate_query(user_query, example_queries, table_schemas, db_path, max_retries=3):
    retries = 0
    error_message = None
    previous_query = None
    generation_tokens = 0
    judging_tokens = 0

    while retries < max_retries:
        print(f"Regenerating query for: '{user_query}' (Attempt {retries + 1}/{max_retries})")

        if error_message:
            response = sqlQueryGeneratorAgent.run_sync(user_query, deps={
                "prompt_query_pairings": example_queries,
                "table_schemas": table_schemas,
                "error": error_message,
                "previous_queries": [],
                "previous_query": previous_query
            })
        else:
            response = sqlQueryGeneratorAgent.run_sync(user_query, deps={
                "prompt_query_pairings": example_queries,
                "table_schemas": table_schemas,
                "previous_queries": [],
                "previous_query": previous_query
            })

        generated_query = response.data
        generation_tokens = response.usage().total_tokens
        cleaned_query = re.sub(r'^```sql\s*|```$', '', generated_query).strip()

        # Check accuracy
        print(f"Generated query: {cleaned_query}")
        accuracy_response = sqlQueryAccuracyJudgeAgent.run_sync(f"Generated Query: {cleaned_query}",
                                                                deps={"original_prompt": user_query})
        judging_tokens = accuracy_response.usage().total_tokens
        accuracy_data = accuracy_response.data.strip()
        match_result = re.search(r'(\d+(?:\.\d+)?)(?:\s*\|\s*(.*))?', accuracy_data)

        if match_result:
            accuracy_score = float(match_result.group(1))
            feedback = match_result.group(2)
        else:
            accuracy_score = 0.0
            feedback = "Unable to parse accuracy score."

        if accuracy_score < 0.6:
            print(f"Accuracy below threshold (score: {accuracy_score}). Feedback: {feedback}")
            retries += 1
            continue

        # Execute query
        results, error_message = execute_query(db_path, cleaned_query)
        if error_message:
            print(f"SQL error: {error_message}")
            retries += 1
            continue

        previous_query = cleaned_query
        print(f"Successfully regenerated query: {cleaned_query}")
        return cleaned_query, results, True, generation_tokens, judging_tokens

    print(f"Failed to regenerate query for: '{user_query}' after {max_retries} attempts.")
    return None, None, False, generation_tokens, judging_tokens


def test_regeneration(db_id, db_path, user_query, test_query, sentences, max_retries=3):
    top_sentences = getTopSentenceMatches(user_query, sentences)
    example_queries = fetchSentencePairing(top_sentences, db_id)
    tables = fetchUsedTables(top_sentences, db_id)
    table_schemas = fetchTableSchemas(tables, db_id)

    # Initialize metrics
    confidence_score = 0
    execution_time = None
    generation_failed = False

    # Generate the query
    generated_query, generated_results, success, generation_tokens, judging_tokens = regenerate_query(
        user_query, example_queries, table_schemas, db_path, max_retries
    )

    if not success or not generated_results:
        generation_failed = True
        return {
            "question": user_query,
            "generated": None,
            "test": test_query,
            "generated_results": None,
            "test_results": None,
            "match": False,
            "confidence_score": confidence_score,
            "execution_time": execution_time,
            "generation_tokens": generation_tokens,
            "judging_tokens": judging_tokens,
            "generation_failed": generation_failed
        }

    # Judge the query
    try:
        accuracy_response = sqlQueryAccuracyJudgeAgent.run_sync(f"Generated Query: {generated_query}",
                                                                deps={"original_prompt": user_query})
        confidence_score = float(accuracy_response.data.strip())
    except Exception as e:
        print(f"Query judging failed: {e}")
        confidence_score = 0

    # Execute the ground-truth query and measure execution time
    try:
        start_execution = time.time()
        test_results, _ = execute_query(db_path, test_query)
        execution_time = time.time() - start_execution
    except Exception as e:
        print(f"Test query execution failed: {e}")
        test_results = None

    # Check for exact match
    match = generated_results == test_results

    return {
        "question": user_query,
        "generated": generated_query,
        "test": test_query,
        "generated_results": generated_results,
        "test_results": test_results,
        "match": match,
        "confidence_score": confidence_score,
        "execution_time": execution_time,
        "generation_tokens": generation_tokens,
        "judging_tokens": judging_tokens,
        "generation_failed": generation_failed
    }


if __name__ == "__main__":
    db_id = "bakery_1"
    db_path = 'databases/bakery_1/bakery_1.sqlite'
    sentences = loadSentences(db_id)

    testSeeds: list = [1, 5, 10, 19, 42]
    match_percentages: list = []
    total_failures = 0
    total_execution_time = 0
    total_tokens_used = 0
    total_confidence_score = 0
    total_tests = 0

    for testSeed in testSeeds:
        print(f"Testing with seed: {testSeed}")

        # Split sentences using the current seed
        training_sentences, testing_sentences = split_sentences_with_ratio_single_group(
            sentences, test_ratio=0.2, seed=testSeed
        )

        # Recalculate embeddings for the current training group
        print("Recalculating embeddings for training sentences...")
        training_embeddings = getTopSentenceMatches("", training_sentences)  # Force recalculation

        results = []

        # Test comparing results for each test sentence
        for test_sentence in testing_sentences:
            user_query = test_sentence

            # Fetch the correct query pairing for the test sentence
            test_query_pairings = fetchSentencePairing([(test_sentence, 0)], db_id)
            if not test_query_pairings:
                print(f"No query pairing found for test sentence: {test_sentence}")
                continue

            test_query = test_query_pairings[0][1]  # Extract the SQL query from the pairing
            # Compare results
            result = test_regeneration(db_id, db_path, user_query, test_query, training_sentences)
            results.append(result)

            # Update metrics
            if result["generation_failed"]:
                total_failures += 1
            else:
                total_execution_time += result["execution_time"] or 0
                total_tokens_used += result["generation_tokens"] + result["judging_tokens"]
                total_confidence_score += result["confidence_score"]

        # Write results to a JSON file for the current seed
        with open(f'tests/regeneration/regen_{testSeed}_gpt_3_5.json', 'w') as f:
            json.dump(results, f, indent=4)

        # Calculate and display the percentage of matches for the current seed
        total_tests += len(results)
        total_matches = sum(1 for result in results if result["match"])
        match_percentage = (total_matches / len(results)) * 100 if results else 0
        print(f"Seed {testSeed} - Match Percentage: {match_percentage:.2f}%")

        match_percentages.append(match_percentage)

    # Write global results to a JSON file in the requested format
    global_results_summary = {}

    for i, seed in enumerate(testSeeds):
        global_results_summary[f"accuracy_seed{seed}"] = f"{match_percentages[i]:.2f}%"

    # Add the overall average match percentage
    average_match_percentage = sum(match_percentages) / len(match_percentages) if match_percentages else 0
    global_results_summary["accuracy_test"] = f"{average_match_percentage:.2f}%"

    # Add additional metrics
    global_results_summary[
        "generation_failure_rate"] = f"{(total_failures / total_tests) * 100:.2f}%" if total_tests else "0.00%"
    global_results_summary[
        "average_execution_time"] = f"{(total_execution_time / (total_tests - total_failures)):.2f}s" if total_tests - total_failures > 0 else "N/A"
    global_results_summary[
        "average_tokens_used"] = f"{(total_tokens_used / total_tests):.2f}" if total_tests else "0.00"
    global_results_summary[
        "average_confidence_score"] = f"{(total_confidence_score / total_tests):.2f}" if total_tests else "0.00"

    with open('tests/regeneration/regen_results_gpt_3_5.json.json', 'w') as f:
        json.dump(global_results_summary, f, indent=4)

    # Display the average match percentage
    print(f"Average Match Percentage: {average_match_percentage:.2f}%")
    print(f"Generation Failure Rate: {global_results_summary['generation_failure_rate']}")
    print(f"Average Execution Time: {global_results_summary['average_execution_time']}")
    print(f"Average Tokens Used: {global_results_summary['average_tokens_used']}")
    print(f"Average Confidence Score: {global_results_summary['average_confidence_score']}")
