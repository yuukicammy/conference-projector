def insert_line_breaks(
    text: str,
    max_chars_per_line: int,
    prefix: str = "",
    suffix: str = "",
    newline="\n",
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
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) <= max_chars_per_line:
            current_line += word + " "
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    lines.append(current_line.strip())
    return prefix + newline.join(lines) + suffix
