from pathlib import Path
import subprocess
import json

import dacite
import toml
import modal

from .config import Config, ProjectConfig, DBConfig
from .webapp import CONFIG_FILE

stub = modal.Stub(ProjectConfig._stub_pipeline)


def run_pipeline(config: Config):
    try:
        subprocess.run(["modal", "nfs", "create", f"{config.project._shared_vol}"])
        if config.pipeline.initialize_volume:
            print("Initializing the shared volume on Modal. Delete all existing files.")
            subprocess.run(
                ["modal", "nfs", "rm", "-r", f"{config.project._shared_vol}", "/"]
            )

        # Deploy all modal stubs
        if config.pipeline.deploy_stubs:
            print("Deploying modal stubs...")
            for name in config.project.stub_files:
                if name == Path(__file__).name:
                    continue  # skip this stub
                subprocess.run(["modal", "deploy", f"{name}"])

        # Parsing a website with a list of papers and creating a metadata file of papers.
        if config.pipeline.run_html_parse:
            print("Parsing a website with a list of papers...")
            modal.Function.lookup(config.project._stub_html_parser, "parse_html").call(
                config=config
            )

        # Generate summaries
        if config.pipeline.run_summarize:
            print("Generating paper summaries...")
            modal.Function.lookup(
                config.project._stub_summary, "generate_summaries"
            ).call(config=config)

        # Generate and save embeddings of title and abstract
        if config.pipeline.run_embed:
            print("Generating embeddings...")
            modal.Function.lookup(
                config.project._stub_embedding, "save_embeddings"
            ).call(config=config)

        # Extract representative images from PDFs of papers.
        if config.pipeline.run_paper_image:
            print("Extracting images that are representative of papers...")
            modal.Function.lookup(
                config.project._stub_paper_image, "extract_representative_images"
            ).call(
                config=config,
            )

        # Extract high-impact papers
        if config.pipeline.run_highlight:
            print("Extracting award information...")
            modal.Function.lookup(config.project._stub_highlights, "scrape_award").call(
                config=config,
            )

    except Exception as e:
        print(e)
    finally:
        # Stop all apps launched.
        if config.pipeline.stop_stubs:
            print("Stopping launched apps...")
            apps = json.loads(
                subprocess.run(
                    ["modal", "app", "list", "--json"], capture_output=True, text=True
                ).stdout
            )
            for app in apps:
                if (
                    app["Name"] in config.project.stub_names
                    and app["State"] != "stopped"
                ):
                    cmd = ["modal", "app", "stop", app["App ID"]]
                    print(f'{app["Name"]}: {cmd}')
                    subprocess.run(cmd)

        if config.pipeline.download_data_locally:
            print("Downloading data to the local disk.")
            Path(config.files.local_output_dir).mkdir(exist_ok=True, parents=True)
            subprocess.run(
                [
                    "modal",
                    "volume",
                    "get",
                    "--force",
                    f"{config.project._shared_vol}",
                    f"/**",
                    f"{config.files.local_output_dir}",
                ]
            )


@stub.function()
def clone_db(from_db_config: DBConfig, to_db_config: DBConfig) -> None:
    papers = modal.Function.lookup(ProjectConfig._stub_db, "get_all_papers").call(
        from_db_config
    )
    for p in papers:
        modal.Function.lookup(ProjectConfig._stub_db, "upsert_item").call(
            to_db_config, p
        )


@stub.local_entrypoint()
def main(config_file: str = "configs/debug.toml"):
    config = dacite.from_dict(data_class=Config, data=toml.load(config_file))
    print(config)

    # from_db_config = DBConfig(
    #     database_id="cvpr2023", container_id="Container-01", uri=config.db.uri
    # )
    # to_db_config = config.db
    # clone_db.call(from_db_config=from_db_config, to_db_config=to_db_config)

    run_pipeline(config)


if __name__ == "__main__":
    main()
