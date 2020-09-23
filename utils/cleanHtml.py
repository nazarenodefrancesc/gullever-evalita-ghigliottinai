import os
import html2text


def clean_html_in_path(path):
    h2t = html2text.HTML2Text()
    h2t.ignore_links = True

    cleaned_htmls = ""
    for filename in os.listdir(path):
        if filename.endswith(".html") or filename.endswith(".htm"):
            raw_html = open(os.path.join(path, filename), "r", encoding="utf8").read()
            cleaned_html = h2t.handle(raw_html)
            cleaned_htmls += cleaned_html
    return cleaned_htmls


if __name__ == "__main__":
    path = (
        "/home/nazareno/CELI/repositories/python_projects/texmega_py/resources/raw/TODO"
    )
    cleaned_htmls = clean_html_in_path(path)
    open(os.path.join(path, "cleaned_htmls.txt"), "w", encoding="utf8").write(
        cleaned_htmls
    )
