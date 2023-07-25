from typing import List
from pathlib import Path


def insert_line_breaks(
    text: str,
    max_chars_per_line: int,
    prefix: str = "",
    suffix: str = "",
    newline="\n",
    is_japanese=False,
) -> str:
    """
    Insert line breaks into the text to ensure that each line has a maximum number of characters.

    Args:
        text (str): The input text.
        max_chars_per_line (int): The maximum number of characters allowed per line.
        prefix (str, optional): The prefix to be added at the beginning of each line. Defaults to "".
        suffix (str, optional): The suffix to be added at the end of each line. Defaults to "".
        newline (str, optional): The newline character to be used. Defaults to "\n".

    Returns:
        str: The text with line breaks inserted.
    """
    if is_japanese:
        words = text
        max_chars_per_line = max_chars_per_line // 2
    else:
        words = text.split()

    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) <= max_chars_per_line:
            current_line += word
            if not is_japanese:
                current_line += " "
        else:
            lines.append(current_line.strip())
            current_line = word
            if not is_japanese:
                current_line += " "
    lines.append(current_line.strip())
    return prefix + newline.join(lines) + suffix


def overwrite_image(
    img_url: str | None, image_path: str, img_ignore_paths: List[str]
) -> str | None:
    import requests

    if img_url is None:
        return None

    if img_url in img_ignore_paths:
        return image_path

    response = requests.get(img_url)
    if response.status_code == 200:
        Path(image_path).parent.mkdir(parents=True, exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(response.content)
    return image_path
