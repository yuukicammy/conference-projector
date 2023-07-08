from pathlib import Path
import subprocess

import dacite
import toml
import modal

from .config import Config, ProjectConfig
from .webapp import CONFIG_FILE

stub = modal.Stub(ProjectConfig._stab_pipeline)


def run_pipeline(config: Config):
    try:
        subprocess.run(["modal", "volume", "create", f"{config.project._shared_vol}"])
        if config.pipeline.initialize_volume:
            print("Initializing the shared volume on Modal. Delete all existing files.")
            subprocess.run(
                ["modal", "volume", "rm", "-r", f"{config.project._shared_vol}", "/"]
            )

        # Deploy all modal stabs
        if config.pipeline.deplpoy_stubs:
            print("Deploying modal stubs.")
            for name in config.project.stab_files:
                if name == Path(__file__).name:
                    continue  # skip this stub
                subprocess.run(["modal", "deploy", f"{name}"])

        # Parsing a website with a list of papers and creating a metadata file of papers.
        if config.pipeline.run_html_parse:
            print("Parsing a website with a list of papers.")
            modal.Function.lookup(config.project._stab_html_parser, "parse_html").call(
                html_config=config.html_parser,
                file_config=config.files,
                project_config=config.project,
            )

        # Extract representative images from PDFs of papers.
        if config.pipeline.run_paper_image:
            print("Extracting images that are representative of papers.")
            modal.Function.lookup(
                config.project._stab_paper_image, "extract_representative_images"
            ).call(
                file_config=config.files,
                project_config=config.project,
            )

        # Generate summaries
        if config.pipeline.run_summarize:
            print("Generating paper summaries.")
            for n in range(100, 2600, 100):
                config.project.max_papers = n
                print(f"{str(n-100)} to {str(n)}")
                modal.Function.lookup(
                    config.project._stab_summary, "generate_summaries"
                ).call(config.summary, config.files, config.project)

        # Generate and save embeddings of title and abstract
        if config.pipeline.run_embed:
            print("Generating embeddings.")
            modal.Function.lookup(
                config.project._stab_embedding, "save_embeddings"
            ).call(config.embedding, config.files, config.project)

    except Exception as e:
        print(e)
    finally:
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


@stub.local_entrypoint()
def main(config_file: str = "configs/defaults.toml"):
    config = dacite.from_dict(data_class=Config, data=toml.load(config_file))
    print(config)

    # Set web sebserver config
    CONFIG_FILE = config_file

    run_pipeline(config)


if __name__ == "__main__":
    main()
