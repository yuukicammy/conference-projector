"""  
summaries_generator.py

Generate summaries for papers by OpenAI Chat API with Function calling.
"""

from typing import List, Any, Dict

import modal

from .config import ProjectConfig, Config

stub = modal.Stub(ProjectConfig._stub_summary)
SHARED_ROOT = "/root/.cache"


@stub.function(
    image=modal.Image.debian_slim().pip_install("openai"),
    shared_volumes={
        SHARED_ROOT: modal.SharedVolume.from_name(ProjectConfig._shared_vol)
    },
    retries=0,
    secret=modal.Secret.from_name("my-openai-secret"),
)
def request_chat(
    prompt: str,
    paper: Dict[str, Any],
    function_schema: List[Dict[str, Any]],
    model: str,
    retry: int = 0,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Send a chat completion request to the OpenAI API to generate summaries for papers.
    Args:
        prompt (str): The prompt for the chat completion.
        paper (Dict[str, Any]): Paper information.
        function_schema (List[Dict[str, Any]]): Function schema for the chat completion.
        model (str): The GPT-3 model to use for chat completion.
        retry (int): Number of retries in case of failure. Default is 0.
        overwrite (bool): Whether to overwrite existing properties. Default is False.

    Returns:
        Dict[str, Any]: The generated summary as a dictionary if successful, None otherwise.
    """

    import json
    import time
    import openai

    if not overwrite and all(
        [
            key in paper.keys() and 0 < len(paper[key])
            for key in function_schema[0]["parameters"]["properties"].keys()
        ]
    ):
        # already has all properties
        return None

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    while 0 <= retry:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                functions=function_schema,
                function_call="auto",
                temperature=0,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            message = response["choices"][0]["message"]
            if message.get("function_call"):
                result = response["choices"][0]["message"]["function_call"]["arguments"]
                print(result)
                if response["choices"][0]["finish_reason"] == "length":
                    result = result + '"}'
                    json_result = json.loads(result)
                    for key in function_schema[0]["parameters"]["properties"].keys():
                        if json_result.get(key) is None:
                            json_result[key] = ""
                    print(f"converted resut: {json_result}")
                    return json_result
                else:
                    return json.loads(result)
            else:
                print("Faied to summarize a paper. retry.")
                raise Exception
        except Exception as e:
            print(e)
            print("Fail to get the function_call result from chat completion.")
            retry -= 1
            time.sleep(1)
    print(f"All retries failed. Return None. \n{prompt}")
    return None


def generate_summary(idx: int, config: Config) -> None:
    import time

    # Load the current information
    items = modal.Function.lookup(ProjectConfig._stub_db, "query_items").call(
        db_config=config.db,
        query=f'SELECT * FROM c WHERE c.id = "{str(idx)}"',
        force=True,
    )
    if items is None or len(items) == 0:
        print(f"Cannot load paper {idx} from db: {config.db}.")
        return
    paper = items[0]
    if all(
        [
            paper.get(key) and 0 < len(paper[key])
            for key in config.summary.function_schema["parameters"]["properties"].keys()
        ]
    ):
        print(f"The paper {idx} has all generated texts.")
        return
    retry = config.summary.retry
    while 0 <= retry:
        try:
            time_sta = time.perf_counter()
            result = request_chat(
                config.summary.prompt.format(paper["title"], paper["abstract"]),
                paper,
                [config.summary.function_schema],
                config.summary.model,
                0,
                False,
            )
            for key in config.summary.function_schema["parameters"][
                "properties"
            ].keys():
                if result.get(key) is None or len(result[key]) == 0:
                    raise Exception(
                        f"Fail to generate summary. {key} is not generated. Paper id {idx}."
                    )
                paper[key] = (
                    result[key]
                    .encode("utf-8")
                    .decode("utf-8")  # Convert Unicode strings to UTF-8
                )
            time_end = time.perf_counter()
            print(f"Request chat latency: {time_end - time_sta}")
            # Update the paper information
            modal.Function.lookup(config.project._stub_db, "upsert_item").call(
                config.db, paper
            )
            return
        except Exception as e:
            print(f"Error happen in generate_summary.", e)
            retry -= 1
    print(f"The upper limit of RETRY is now reached in generating summary {idx}.")


@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "openai"
    ),  # dockerfile_commands(["RUN apt-get update", "RUN apt-get install -y python3-pip"])
    shared_volumes={
        SHARED_ROOT: modal.SharedVolume.from_name(ProjectConfig._shared_vol)
    },
    secret=modal.Secret.from_name("my-openai-secret"),
    retries=0,
    cpu=8,
    timeout=36000,
)
def generate_summaries(config: Config) -> None:
    """
    Generate summaries for papers based on the provided configuration.

    Args:
        config (Config): The configuration object.

    Returns:
        None
    """
    import concurrent

    num_papers = modal.Function.lookup(ProjectConfig._stub_db, "get_num_papers").call(
        db_config=config.db
    )

    with concurrent.futures.ThreadPoolExecutor(config.project.num_workers) as executor:
        futures = [
            executor.submit(generate_summary, i, config)
            for i in range(min(num_papers, config.project.max_papers))
        ]
        concurrent.futures.wait(futures)


@stub.local_entrypoint()
def main(config_file: str = "configs/icml2023.toml"):
    import dacite
    import toml

    config = dacite.from_dict(data_class=Config, data=toml.load(config_file))
    print(config)

    generate_summaries.call(config)
