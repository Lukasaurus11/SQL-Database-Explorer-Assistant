import json

"""
This file is in charge of creating the data will that will be used to test our application. The data is gathered from 
the spider_data dataset https://yale-lily.github.io/spider. We are taking the dev.json and test.json files, and shortening
them to to only include the fields that we are interested in (db_id, query, question)
"""


def generateData() -> None:
    """
    This function will be used to generate the data that will be used to test the application. Data is gathered from
    the dev.json and test.json files.

    :return: Nothing since the data will just be saved under the data/ folder
    """
    finalData: dict = {}
    with open("../raw/spider_data/spider_data/dev.json") as devFile:
        devData: dict = json.load(devFile)
        for i, values in enumerate(devData):
            finalData[i] = {
                "db_id": values["db_id"],
                "question": values["question"],
                "query": values["query"]
            }

    with open("../raw/spider_data/spider_data/test.json") as testFile:
        testData: dict = json.load(testFile)
        for i, values in enumerate(testData):
            finalData[i + len(devData)] = {
                "db_id": values["db_id"],
                "question": values["question"],
                "query": values["query"]
            }

    with open("../data/data.json", "w") as dataFile:
        json.dump(finalData, dataFile, indent=4)


def generateTableSchema(tableInformation: dict) -> dict:
    def mapTableIndices(tableNames: list) -> dict:
        """
        This function maps the table indices to their respective names

        example:
            - input:
                ['perpetrator', 'people']
            - output:
                {0: 'perpetrator', 1: 'people'}

        :param tableNames:
        :return:
        """
        return {index: name for index, name in enumerate(tableNames)}

    def organizeColumns(tableInfo: dict, tableIndexToName: dict) -> tuple:
        """


        :param tableInfo:
        :param tableIndexToName:
        :return:
        """
        typeMapping: dict = {"text": "TEXT", "number": "INT", "real": "REAL"}
        tables: dict = {name: [] for name in tableInfo['table_names']}
        columnMap: dict = {}
        for index, ((table_index, column_name), column_type) in enumerate(
                zip(tableInfo['column_names_original'], tableInfo['column_types'])):

            if table_index != -1:
                tableName: str = tableIndexToName[table_index]
                sqlType: str = typeMapping.get(column_type, "TEXT")
                tables[tableName].append((column_name, sqlType))
                columnMap[index] = (tableName, column_name)
        return tables, columnMap

    def identifyPrimaryKeys(tableInfo: dict, columnMap: dict) -> dict:
        primaryKeys = {}
        for primaryKeyIndex in tableInfo["primary_keys"]:
            tableName, columnName = columnMap[primaryKeyIndex]
            if tableName not in primaryKeys:
                primaryKeys[tableName] = []
            primaryKeys[tableName].append(columnName)
        return primaryKeys

    def identifyForeignKeys(tableInfo: dict, column_map: dict) -> dict:
        foreignKeys = {}
        for columnIndex, refColumnIndex in tableInfo["foreign_keys"]:
            columnTableName, columnName = column_map[columnIndex]
            referenceTableName, referenceColumnName = column_map[refColumnIndex]
            foreignKeys.setdefault(columnTableName, []).append((columnName, referenceTableName, referenceColumnName))
        return foreignKeys

    def generateCreateTableStatements(tables: dict, primaryKeys: dict, foreignKeys: dict) -> dict:
        sqlStatements = {}
        for tableName, columns in tables.items():
            columnDefinitions = [f'"{col}" {ctype}' for col, ctype in columns]
            if tableName in primaryKeys:
                primaryKeyColumns = primaryKeys[tableName]
                columnDefinitions.append(f'PRIMARY KEY ({", ".join(f"{pk}" for pk in primaryKeyColumns)})')
            if tableName in foreignKeys:
                for columnName, referenceTable, referenceColumn in foreignKeys[tableName]:
                    columnDefinitions.append(
                        f'FOREIGN KEY ("{columnName}") REFERENCES "{referenceTable}"("{referenceColumn}")')
            sqlStatements[tableName] = f'CREATE TABLE "{tableName}" (\n  ' + ",\n  ".join(columnDefinitions) + "\n);"
        return sqlStatements

    tableIndexToName = mapTableIndices(tableInformation['table_names'])
    tables, columnMappings = organizeColumns(tableInformation, tableIndexToName)
    primaryKeyMappings = identifyPrimaryKeys(tableInformation, columnMappings)
    foreignKeyMappings = identifyForeignKeys(tableInformation, columnMappings)
    return generateCreateTableStatements(tables, primaryKeyMappings, foreignKeyMappings)


def generateTableInformation() -> None:
    """
    This function generates the table information so that it will be easier to work with when accessing necessary fields

    :return: Nothing as the data will just be saved under the data/ folder
    """

    def processTableData(dataToProcess: dict) -> dict:
        tableInfo: dict = {}
        for table in dataToProcess:
            sqlPrompt: dict = generateTableSchema(table)

            for key, value in sqlPrompt.items():
                if table['db_id'] not in tableInfo:
                    tableInfo[table['db_id']] = {}

                tableInfo[table['db_id']][key] = value

        return tableInfo

    tableInformation: dict = {}
    with open("../raw/spider_data/spider_data/tables.json") as tableFile:
        tableData: dict = json.load(tableFile)
        tableInformation.update(processTableData(tableData))

    with open("../raw/spider_data/spider_data/test_tables.json") as testTableFile:
        testTableData: dict = json.load(testTableFile)
        tableInformation.update(processTableData(testTableData))

    with open("../data/table_information.json", "w") as tableInfoFile:
        json.dump(tableInformation, tableInfoFile, indent=4)


if __name__ == "__main__":
    generateTableInformation()
