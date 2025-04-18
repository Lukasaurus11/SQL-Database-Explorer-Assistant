from json import load as json_load
from pickle import dump as pickle_dump, load as pickle_load

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
from sqlparse import parse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML
from torch import torch

_model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')
_cacheFile: str = "data/embeddings.pkl"


def loadSentences(db_id: str) -> list:
    """
    Function used to load the example sentences from the data.json file

    :param db_id: The database ID to filter the sentences
    :return: The list of example sentences
    """
    with open("data/data.json", "r") as file:
        data: dict = json_load(file)
        data = {key: value for key, value in data.items() if value["db_id"] == db_id}
        return [data[key]["question"] for key in data]


def saveEmbeddings(embeddings: list, sentences: list) -> None:
    """
    Function to save the embeddings into a file, to be used later (faster for retrival)

    :param embeddings: The embeddings to save
    :param sentences: The sentences that the embeddings correspond to
    :return: Nothing
    """
    with open(_cacheFile, "wb") as file:
        pickle_dump({"embeddings": embeddings, "sentences": sentences}, file)


def loadEmbeddings() -> dict | None:
    """
    Function to load the embeddings from the file

    :return: The embeddings and the sentences
    """
    try:
        with open(_cacheFile, "rb") as file:
            return pickle_load(file)

    except FileNotFoundError:
        return None


def getTopSentenceMatches(userQuery: str, sentences: list) -> list:
    """
    Function used to determine the top matching sentences from the provided sentences.
        - defaults:
            threshold: 0.6
            returned sentences: 5-10

    :param userQuery: The user query to compare against the example sentences
    :param sentences: The list of example sentences
    :return: The list of top matches
    """
    cachedEmbeddings: dict | None = loadEmbeddings()

    if cachedEmbeddings and cachedEmbeddings["sentences"] == sentences:
        sentenceEmbeddings: torch.Tensor = torch.tensor(cachedEmbeddings["embeddings"])
    else:
        print("Computing embeddings...")
        sentenceEmbeddings: torch.Tensor = _model.encode(sentences, convert_to_tensor=True)
        saveEmbeddings(sentenceEmbeddings.tolist(), sentences)

    encodedUserInput: torch.Tensor = _model.encode(userQuery, convert_to_tensor=True)
    similarities: torch.Tensor = pytorch_cos_sim(encodedUserInput, sentenceEmbeddings)[0]

    scoredSentences: list = [(sentences[i], i, similarities[i].item()) for i in range(len(sentences))]
    scoredSentences.sort(key=lambda x: x[2], reverse=True)

    threshold: float = 0.6
    minSentences: int = 5
    maxSentences: int = 10

    filteredSentences: list = [(sentence, index) for sentence, index, score in scoredSentences if score >= threshold]

    while len(filteredSentences) < minSentences and threshold > 0:
        threshold -= 0.02
        filteredSentences = [(sentence, index) for sentence, index, score in scoredSentences if score >= threshold]

    return filteredSentences[:maxSentences]


def fetchUsedTables(sqlQueries: list, db_id: str) -> set:
    """
    Function to grab the used tables from the sentences provided from the data.json file

    :param sqlQueries: The sentences to grab the tables from
    :param db_id: The database ID to filter the sentences
    :return: The set of used tables
    """
    questions: list = [sentence for sentence, index in sqlQueries]

    with open("data/data.json", "r") as file:
        data: dict = json_load(file)

    data = {value["question"]: value["query"] for i, value in enumerate(data.values()) if
            value["question"] in questions and value["db_id"] == db_id}
    sqlQueries = list(data.values())

    tables: set = set()
    for query in sqlQueries:
        parsed: list = parse(query)

        for statement in parsed:
            if statement.get_type() != "SELECT":
                continue

            seenFrom: bool = False
            for token in statement.tokens:
                if seenFrom:
                    if isinstance(token, Identifier):
                        tables.add(token.get_real_name())
                        seenFrom = False  # Reset after finding the table

                    elif isinstance(token, IdentifierList):
                        for identifier in token.get_identifiers():
                            tables.add(identifier.get_real_name())
                        seenFrom = False  # Reset after finding the tables

                if token.ttype is Keyword and token.value.upper() in ["FROM", "JOIN", "INNER JOIN", "LEFT JOIN",
                                                                      "RIGHT JOIN"]:
                    seenFrom = True

                elif token.ttype is DML:
                    seenFrom = False

    return tables


def fetchSentencePairing(sentences: list, db_id: str) -> list[tuple]:
    """
    Function to fetch the sentence-pairings from the data.json file

    :param sentences: List of tuples containing sentences and their indexes
    :param db_id: The database ID to filter the sentences
    :return: List of tuples containing sentences and their corresponding queries
    """
    with open("data/data.json", "r") as file:
        data: dict = json_load(file)

    question_to_query = {value["question"]: value["query"] for value in data.values() if value["db_id"] == db_id}
    sentence_pairings = [(sentence, question_to_query[sentence]) for sentence, _ in sentences if
                         sentence in question_to_query]

    return sentence_pairings


def fetchTableSchemas(tableList: list | set, db_id: str) -> list:
    """
    Function to fetch the table schemas from the table_information.json file given a list of tables

    :param tableList: The list of tables to fetch schemas for
    :param db_id: The database ID to filter the schemas
    :return: The list of table schemas
    """
    with open("data/table_information.json", "r") as file:
        data: dict = json_load(file)
    return [data[db_id][table] for table in tableList]
