"""
cosmos.py

This script provides functions for interacting with Azure Cosmos DB.

NOTE: Import libraries within functions for Modal's remote environment.
"""

from pathlib import Path
from typing import Dict, Any, List
import modal

from .config import DBConfig, Config, ProjectConfig

SHARED_ROOT = "/root/.cache"

modal_image = modal.Image.debian_slim().pip_install("azure-cosmos")
stub = modal.Stub(
    ProjectConfig._stub_db,
    image=modal_image,
    secrets=[modal.Secret.from_name("cosmos-secret")],
    mounts=[
        modal.Mount.from_local_dir(
            Path(__file__).parent.parent / "configs", remote_path="/root/configs"
        )
    ],
)

stub.cache = modal.Dict.new()

if stub.is_inside():
    import os
    import azure.cosmos.cosmos_client as cosmos_client
    import azure.cosmos.exceptions as exceptions
    from azure.cosmos.partition_key import PartitionKey


@stub.function(
    network_file_systems={
        SHARED_ROOT: modal.NetworkFileSystem.from_name(ProjectConfig._shared_vol)
    },
)
def get_container(db_config: DBConfig):
    """
    Get the container client for Azure Cosmos DB.

    Parameters:
        db_config (DBConfig): The configuration for the Azure Cosmos DB.

    Returns:
        azure.cosmos.cosmos_client.CosmosContainer: The container client.

    """
    container = create_db(db_config=db_config)
    return container


@stub.function(
    network_file_systems={
        SHARED_ROOT: modal.NetworkFileSystem.from_name(ProjectConfig._shared_vol)
    },
    timeout=3600,
)
def get_all_papers(
    db_config: DBConfig,
    max_item_count: int = None,
    force=False,
) -> List[Dict[str, Any]]:
    """
    Get all papers from the Azure Cosmos DB.

    Parameters:
        db_config (DBConfig): The configuration for the Azure Cosmos DB.
        max_item_count (int, optional): The maximum number of items to retrieve. Defaults to None.
        force (bool, optional): Force to retrieve the items from the database even if cached. Defaults to False.

    Returns:
        List[Dict[str, Any]]: The list of papers.

    """

    dict_str: str = (
        f"{db_config.uri}-{db_config.database_id}-{db_config.container_id}-all-papers"
    )
    if (
        force
        or stub.name != ProjectConfig._stub_db
        or not stub.is_inside()
        or not stub.app.cache.contains(dict_str)
    ):
        container = create_db(db_config)
        item_list = list(container.read_all_items())
        item_list = sorted(item_list, key=lambda x: int(x["id"]))
        if stub.is_inside():
            stub.app.cache[dict_str] = item_list
    else:
        item_list = stub.app.cache[dict_str]

    if max_item_count is not None and 0 < max_item_count:
        return item_list[:max_item_count]
    else:
        return item_list


@stub.function(
    network_file_systems={
        SHARED_ROOT: modal.NetworkFileSystem.from_name(ProjectConfig._shared_vol)
    },
)
def query_items(
    db_config: DBConfig,
    query: str = 'SELECT * FROM c WHERE c.id < "200"',
    force: bool = False,
) -> List[Dict[str, str]]:
    """
    Query items from the Azure Cosmos DB.

    Parameters:
        db_config (DBConfig): The configuration for the Azure Cosmos DB.
        container (azure.cosmos.cosmos_client.CosmosContainer): The container client.
        query (str, optional): The query string. Defaults to 'SELECT * FROM c WHERE c```python
        .id < "200"'.
        force (bool, optional): Force to retrieve the items from the database even if cached. Defaults to False.

    Returns:
        List[Dict[str, Any]]: The list of items returned from the query.

    """
    from modal import container_app

    dict_str: str = f"{db_config.uri}-{db_config.database_id}-{db_config.container_id}-query-{query}"
    items = []
    if (
        force
        or stub.name != ProjectConfig._stub_db
        or not stub.is_inside()
        or not stub.app.cache.contains(dict_str)
    ):
        container = create_db(db_config)
        items = list(
            container.query_items(query=query, enable_cross_partition_query=True)
        )
        if stub.is_inside():
            container_app.cache[dict_str] = items
    else:
        items = container_app.cache[dict_str]
    return items


@stub.function(
    network_file_systems={
        SHARED_ROOT: modal.NetworkFileSystem.from_name(ProjectConfig._shared_vol)
    },
)
def get_num_papers(db_config: DBConfig) -> int:
    """
    Get the number of papers in the Azure Cosmos DB.

    Parameters:
        db_config (DBConfig): The configuration for the Azure Cosmos DB.

    Returns:
        int: The number of papers in the database.

    """
    query = "SELECT VALUE COUNT(1) FROM c"
    container = create_db(db_config)
    result = list(container.query_items(query, enable_cross_partition_query=True))
    return result[0]


@stub.function(
    network_file_systems={
        SHARED_ROOT: modal.NetworkFileSystem.from_name(ProjectConfig._shared_vol)
    },
)
def register_from_json(db_config: DBConfig, json_path: str) -> None:
    """
    Register papers from a JSON file to the Azure Cosmos DB.

    Parameters:
        db_config (DBConfig): The configuration for the Azure Cosmos DB.
        json_path (str): The path to the JSON file.

    """
    import json

    container = create_db(db_config)
    json_path = Path(SHARED_ROOT) / json_path

    with open(str(json_path), "r", encoding="utf-8") as f:
        papers = json.load(f)

    for idx, paper in enumerate(papers):
        paper["id"] = str(idx)
        container.create_item(body=paper)


@stub.function(
    network_file_systems={
        SHARED_ROOT: modal.NetworkFileSystem.from_name(ProjectConfig._shared_vol)
    },
    timeout=720,
)
def upsert_item(db_config: DBConfig, item: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Upsert an item in the Azure Cosmos DB.

    Parameters:
        db_config (DBConfig): The configuration for the Azure Cosmos DB.
        item (Dict[Any, Any]): The item to upsert.

    Returns:
        Dict[str, Any]: The upserted item.

    """
    container = create_db(db_config)
    return container.upsert_item(body=item)


@stub.function(
    network_file_systems={
        SHARED_ROOT: modal.NetworkFileSystem.from_name(ProjectConfig._shared_vol)
    },
    secret=modal.Secret.from_name("cosmos-secret"),
)
def create_db(db_config: DBConfig):
    """
    Create the Azure Cosmos DB and the container.

    Parameters:
        db_config (DBConfig): The configuration for the Azure Cosmos DB.

    Returns:
        azure.cosmos.cosmos_client.CosmosContainer: The container client.

    """
    import os
    import azure.cosmos.cosmos_client as cosmos_client
    import azure.cosmos.exceptions as exceptions
    from azure.cosmos.partition_key import PartitionKey

    client = cosmos_client.CosmosClient(
        db_config.uri,
        {"masterKey": os.environ["PRIMARY_KEY"]},
        user_agent="CosmosDBPythonQuickstart",
        user_agent_overwrite=True,
    )

    try:
        db = client.create_database(id=db_config.database_id)
        # print("Database with id '{0}' created".format(db_config.database_id))

    except exceptions.CosmosResourceExistsError:
        db = client.get_database_client(db_config.database_id)
        # print("Database with id '{0}' was found".format(db_config.database_id))

    # setup container
    try:
        container = db.create_container(
            id=db_config.container_id,
            partition_key=PartitionKey(path="/partitionKey"),
            offer_throughput=400,
        )
        # print("Container with id '{0}' created".format(db_config.container_id))

    except:
        container = db.get_container_client(db_config.container_id)
        # print("Container with id '{0}' was found".format(db_config.container_id))
    return container


@stub.local_entrypoint()
def main(config_file: str = "configs/defaults.toml"):
    """
    The main entrypoint of the script.

    Parameters:
        config_file (str, optional): The path to the configuration file. Defaults to "configs/defaults.toml".

    """
    import dacite
    import toml

    config = dacite.from_dict(data_class=Config, data=toml.load(config_file))
    print(config)
    results = get_all_papers.call(config.db)
    print(len(results))
