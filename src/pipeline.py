from pathlib import Path
import subprocess
import json
from typing import List

import modal

from .config import Config, ProjectConfig, DBConfig

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
            modal.Function.lookup(
                config.project._stub_highlights, "award_from_config"
            ).call(
                config=config,
            )

        if 0 < len(config.pipeline.deploy_webapp):
            print("Deploying a webapp...")
            subprocess.run(
                [
                    "modal",
                    "deploy",
                    config.pipeline.deploy_webapp,
                ]
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
                    and app["Name"] != ProjectConfig._stub_pipeline
                    and app["Name"] != ProjectConfig._stub_db
                    and app["Name"] != ProjectConfig._stub_webapp
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


@stub.function(timeout=3600)
def clone_db(from_db_config: DBConfig, to_db_config: DBConfig) -> None:
    db_size = modal.Function.lookup(ProjectConfig._stub_db, "get_num_papers").call(
        db_config=from_db_config
    )
    start_id = modal.Function.lookup(ProjectConfig._stub_db, "get_num_papers").call(
        db_config=to_db_config
    )
    print(start_id)
    for i in range(start_id, db_size):
        query = f'SELECT * FROM c WHERE c.id= "{str(i)}"'
        paper = modal.Function.lookup(ProjectConfig._stub_db, "query_items").call(
            db_config=from_db_config, query=query, force=True
        )[0]
        modal.Function.lookup(ProjectConfig._stub_db, "upsert_item").call(
            db_config=to_db_config, item=paper
        )


@stub.function(
    network_file_systems={
        "/root/source": modal.NetworkFileSystem.from_name("paper-viz-vol"),
        "/root/target": modal.NetworkFileSystem.from_name("cp-debug-vol"),
    },
)
def clone_nfs(from_dir: str | None = "cvpr2023", to_dir: str | None = None):
    import shutil

    from_root_dir = Path("./source")
    if from_dir is not None and 0 < len(from_dir):
        from_root_dir = from_root_dir / from_dir
    to_root_dir = Path("./target")
    if to_dir is not None and 0 < len(to_dir):
        to_root_dir = to_root_dir / to_dir

    for source_file_path in from_root_dir.rglob("*.png"):
        if source_file_path.is_file():
            relative_path = source_file_path.relative_to(from_root_dir)
            destination_file_path = to_root_dir / relative_path
            destination_file_path.parent.mkdir(parents=True, exist_ok=True)
            # destination_file_path = destination_file_path.with_name(
            #     destination_file_path.name.replace("cvpr2023", "debug")
            # )
            print(f"from: {source_file_path}, to: {destination_file_path}")
            shutil.copy2(source_file_path, destination_file_path)


@stub.function(timeout=3600)
def add_columns(config: Config, keys: List[str]):
    num_papers = modal.Function.lookup(ProjectConfig._stub_db, "get_num_papers").call(
        db_config=config.db
    )
    for i in range(min(num_papers, config.project.max_papers)):
        paper = modal.Function.lookup(ProjectConfig._stub_db, "query_items").call(
            db_config=config.db,
            query=f'SELECT * FROM c WHERE c.id = "{str(i)}"',
            force=True,
        )[0]
        do_upsert = False
        for key in keys:
            if key not in paper.keys():
                do_upsert = True
                paper[key] = ""

        if do_upsert:
            modal.Function.lookup(ProjectConfig._stub_db, "upsert_item").call(
                db_config=config.db, item=paper
            )


@stub.local_entrypoint()
def main(config_file: str = "configs/debug.toml"):
    import dacite
    import toml

    config = dacite.from_dict(data_class=Config, data=toml.load(config_file))
    config.project.config_file = config_file
    print(config)

    # clone_nfs.call(from_dir="cvpr2023", to_dir="cvpr2023")

    # from_db_config = DBConfig(
    #     database_id="cvpr2023", container_id="Container-01", uri=config.db.uri
    # )
    # to_db_config = config.db
    # clone_db.call(from_db_config=from_db_config, to_db_config=to_db_config)
    # run_pipeline(config)


if __name__ == "__main__":
    main()
