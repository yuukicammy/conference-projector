from pathlib import Path
from typing import Dict, Any

import modal
from PIL import Image

from .config import ProjectConfig, Config, DBConfig

stub = modal.Stub(ProjectConfig._stab_paper_image)
SHARED_ROOT = "/root/.cache"


@stub.function()
def resize_image(img: Image, max_size: int | None) -> Image:
    """
    Resize the image to the specified maximum size.

    Args:
        img (Image): Image object.
        max_size (int | None): Maximum size of the image. None means no resizing.

    Returns:
        Image: Resized image.
    """
    if max_size is not None and (max_size < img.size[0] or max_size < img.size[1]):
        if img.size[0] < img.size[1]:
            img = img.resize(
                (int((float(max_size) / img.size[1]) * img.size[0]), max_size)
            )
        else:
            img = img.resize(
                (max_size, int((float(max_size) / img.size[0]) * img.size[1]))
            )
    return img


@stub.function(timeout=36000)
def save_arxiv_image(
    arxiv_id: str, save_path: str, image_size_limit: int = None
) -> bool:
    """
    Save the image associated with the given arXiv ID.

    Args:
        arxiv_id (str): arXiv ID of the paper.
        save_path (str): Path to save the image.
        image_size_limit (int): Maximum size limit for the image. Default is None.

    Returns:
        bool: True if the image is successfully saved, False otherwise.
    """
    import os
    from pathlib import Path
    import requests
    import tarfile
    import zipfile
    from PIL import Image
    from pdf2image import convert_from_path

    if len(arxiv_id) == 0:
        return False

    try:
        archive_url = f"https://arxiv.org/e-print/{arxiv_id}"
        response = requests.get(archive_url)

        try:
            tmpname = Path(arxiv_id + ".tar.gz")

            with open(tmpname, "wb") as f:
                f.write(response.content)

            with tarfile.open(tmpname, "r") as tar:
                tar.extractall(arxiv_id)
        except Exception as e:
            tmpname = Path(arxiv_id + ".zip")

            with open(tmpname, "wb") as f:
                f.write(response.content)

            with zipfile.ZipFile(tmpname, "r") as zip:
                zip.extractall(arxiv_id)
        finally:
            os.remove(tmpname)

        image_extensions = [".jpg", ".jpeg", ".png", ".pdf"]

        max_size = 0
        max_size_image_path = None

        for path in Path(arxiv_id).glob("**/*"):
            if path.is_file() and path.suffix.lower() in image_extensions:
                file_size = path.stat().st_size
                if max_size < file_size and file_size < 512000:
                    max_size = file_size
                    max_size_image_path = path

        if max_size_image_path.suffix.lower() == ".pdf":
            images = convert_from_path(
                Path("/root") / max_size_image_path, first_page=1, last_page=1
            )
            images[0] = resize_image(images[0], image_size_limit)
            images[0].save(Path(SHARED_ROOT) / save_path, "PNG")
            return True
        elif max_size_image_path is not None:
            img = Image.open(max_size_image_path)
            img = resize_image(img, image_size_limit)
            img.save(Path(SHARED_ROOT) / save_path, "PNG")
            return True
    except Exception as e:
        print(f"Failed in save_arxiv_image {e}, arxiv id: {arxiv_id}")
        return False


@stub.function(timeout=36000)
def search_arxiv(title: str) -> str:
    """
    Search for an arXiv ID based on the given title.

    Args:
        title (str): Title of the paper.

    Returns:
        str: arXiv ID if found, an empty string otherwise.
    """
    import arxiv

    try:
        for res in arxiv.Search(query=f'ti:"{title}"').results():
            if res.title == title:
                arxiv_id = res.entry_id.split("/")[-1]
                return arxiv_id
        return ""
    except Exception as e:
        print(e)
        return ""


@stub.function(
    timeout=36000,
    shared_volumes={
        SHARED_ROOT: modal.SharedVolume.from_name(ProjectConfig._shared_vol)
    },
)
def extract_image(
    paper: Dict[str, Any], save_path: str, idx: int, config: Config
) -> None:
    """
    Extract and save the representative image for a paper.

    Args:
        paper (Dict[str, Any]): Paper information.
        save_path (str): Path to save the image.
        idx (int): Index of the paper.
        config (Config): Configuration object.
    """
    from pathlib import Path

    if len(paper["arxiv_id"]) == 0:
        paper["arxiv_id"] = search_arxiv(title=paper["title"])
    if not save_arxiv_image(
        paper["arxiv_id"], save_path, image_size_limit=config.files.image_max_size
    ):
        save_pdf_image(
            url=paper["pdf_url"],
            save_path=save_path,
            image_size_limit=config.files.image_max_size,
        )
    if not (Path(SHARED_ROOT) / paper["image_path"]).is_file():
        paper["image_path"] = ""
    paper["id"] = str(idx)
    modal.Function.lookup(config.project._stab_db, "upsert_item").call(config.db, paper)


@stub.function(
    image=modal.Image.debian_slim().pip_install("pymupdf", "Pillow", "requests"),
    shared_volumes={
        SHARED_ROOT: modal.SharedVolume.from_name(ProjectConfig._shared_vol)
    },
    retries=0,
    cpu=1,
    timeout=36000,
)
def save_pdf_image(
    url: str = None,
    file_path: str = None,
    save_path: str = None,
    method: str = "max",
    min_width: int = 100,
    min_height: int = 100,
    page_limit: int = -1,
    image_size_limit: int = None,
) -> bool:
    """
    Extract and save the image from a PDF.

    Args:
        url (str): URL of the PDF file.
        file_path (str): Path to the PDF file.
        save_path (str): Path to save the image.
        method (str): Method for selecting the image. Options are "max" and "thresh". Default is "max".
        min_width (int): Minimum width of the image. Default is 100.
        min_height (int): Minimum height of the image. Default is 100.
        page_limit (int): Maximum number of pages to process. Default is -1 (process all pages).
        image_size_limit (int): Maximum size of the image. Default is None.

    Returns:
        bool: True if the image is saved successfully, False otherwise.
    """
    from pathlib import Path
    import io
    import requests

    from PIL import Image
    import fitz

    if url:
        response = requests.get(url)
        pdf_data = response.content

        doc = fitz.open(stream=pdf_data, filetype="pdf")
    elif file_path:
        doc = fitz.open(file_path, filetype="pdf")
    if page_limit < 0:
        page_limit = doc.page_count

    best_img = None
    max_area = 0
    for page in doc:
        if page_limit <= 0:
            break
        page_limit -= 1
        for image, info in zip(page.get_images(), page.get_image_info()):
            xref = image[0]
            x1, y1, x2, y2 = info["bbox"]
            width = x2 - x1
            height = y2 - y1
            area = width * height
            assert 0 <= area
            if (method == "max" and max_area < area) or (
                method == "thresh" and (min_width < width and min_height < height)
            ):
                base_image = doc.extract_image(xref)
                best_img = Image.open(io.BytesIO(base_image["image"]))
                max_area = area
                if method == "thresh":
                    break

    if best_img:
        best_img = resize_image(best_img, image_size_limit)
        best_img.convert("RGB").save(
            Path(SHARED_ROOT) / save_path, "PNG", optimize=True
        )
        return True
    else:
        return False


@stub.function(
    image=modal.Image.debian_slim()
    .apt_install("poppler-utils")
    .pip_install("pymupdf", "Pillow", "requests", "pdf2image", "arxiv"),
    shared_volumes={
        SHARED_ROOT: modal.SharedVolume.from_name(ProjectConfig._shared_vol)
    },
    retries=0,
    cpu=12,
    timeout=36000,
)
def extract_representative_images(config: Config) -> None:
    """
    Extract and save representative images for the given papers.

    Args:
        config (Config): Configuration object.
    """
    import json
    import concurrent

    papers = modal.Function.lookup(config.project._stab_db, "get_all_papers").call(
        config.db, config.project.max_papers
    )

    dir = Path(config.project.dataname) / "top_images"
    Path(SHARED_ROOT / dir).mkdir(parents=True, exist_ok=True)  # recursive

    tasks = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config.project.num_workers
    ) as executor:
        for i, paper in enumerate(papers):
            im_path = dir / (str(i).zfill(config.files.image_name_width) + ".png")
            paper["image_path"] = str(im_path)
            if (
                config.files.force_extract_image
                or not (Path(SHARED_ROOT) / paper["image_path"]).is_file()
            ):
                tasks.append(
                    executor.submit(extract_image, paper, str(im_path), i, config)
                )
            else:
                paper["id"] = str(i)
                tasks.append(
                    executor.submit(
                        modal.Function.lookup(
                            config.project._stab_db, "upsert_item"
                        ).call,
                        config.db,
                        paper,
                    ),
                )

    concurrent.futures.wait(tasks)  # wait until all tasks are completed.

    if config.files.save_json:
        for paper in papers:
            if not (Path(SHARED_ROOT) / paper["image_path"]).is_file():
                # cannote extract good images from PDF
                paper["image_path"] = ""
        with open(
            str(Path(SHARED_ROOT) / config.files.json_file), "w", encoding="utf-8"
        ) as fw:
            json.dump(papers, fw, indent=config.files.json_indent, ensure_ascii=False)


@stub.local_entrypoint()
def main(config_file: str = "configs/defaults.toml"):
    """
    Main function for extracting representative images.

    Args:
        config_file (str): Path to the configuration file. Default is "configs/defaults.toml".
    """
    import dacite
    import toml

    config = dacite.from_dict(data_class=Config, data=toml.load(config_file))
    print(config)
    extract_representative_images.call(config)


if __name__ == "__main__":
    main()
