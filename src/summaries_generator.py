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
    max_tokens: int = 4000,
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
            key in paper.keys() and 0 < len(paper[key]) and "ã" not in paper[key]
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
                max_tokens=max_tokens,
            )
            message = response["choices"][0]["message"]
            if 0 < len(message["function_call"]):
                result = response["choices"][0]["message"]["function_call"]["arguments"]
                print(result)
                if "ã" not in result:
                    # TODO: Processing when finish_reason is length.
                    return json.loads(result)
                else:
                    print("Wrong text.", response)
                    raise Exception("Wrong text.", response)
            else:
                raise Exception("Failed to summarize a paper. retry.")
        except Exception as e:
            print("Fail to get the function_call result from chat completion. ", e)
            retry -= 1
            time.sleep(1)
    print(f"All retries failed. Return None. \n{prompt}")
    return None


def generate_summary(idx: int, config: Config) -> None:
    import time
    import re

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
            key in paper.keys() and 0 < len(paper[key]) and "ã" not in paper[key]
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
                prompt=config.summary.prompt.format(
                    re.sub(r"\\.", "", paper["title"])
                    .replace("\\", "\\\\")
                    .replace('"', "")
                    .replace("'", "")
                    .replace("$", ""),
                    re.sub(
                        r"\\.",
                        "",
                        paper["abstract"]
                        .replace("\n", " ")
                        .replace("{", "(")
                        .replace("}", ")")
                        .replace("\\", "\\\\")
                        .replace('"', "")
                        .replace("'", "")
                        .replace("$", ""),
                    ),
                ),
                paper=paper,
                function_schema=[config.summary.function_schema],
                model=config.summary.model,
                retry=0,
                overwrite=False,
                max_tokens=config.summary.max_tokens,
            )
            for key in config.summary.function_schema["parameters"][
                "properties"
            ].keys():
                if key not in result.keys() or len(result[key]) == 0:
                    key_ = key.replace("_ja", "_en")
                    if retry == 0 and key_ in result.keys() and 0 < len(result[key_]):
                        result[key] = result[key_]
                    else:
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
            time.sleep(config.summary.sleep)
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
    print("Num papers: ", num_papers, ", DB :", config.db)

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
