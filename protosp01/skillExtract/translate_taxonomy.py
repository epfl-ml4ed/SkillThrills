# %%
import pandas as pd
import argparse

from utils import *

parser = argparse.ArgumentParser()
# fmt: off
parser.add_argument("--taxonomy", type=str, help="Path to taxonomy file in csv format", default = "../data/taxonomy/taxonomy_V4.csv")

args = parser.parse_args([])

# fmt: on

taxonomy, _, _ = load_taxonomy(args)


def translate_text(text, src_lang="de", dest_lang="en"):
    """
    Translates text from one language to another.
    text (str): Text to be translated.
    src_lang (str): Source language.
    dest_lang (str): Target language.
    """
    translator = Translator()
    try:
        translation = translator.translate(text, src=src_lang, dest=dest_lang)
    except:
        print("Time out error. Waiting for 10 seconds...")
        time.sleep(10)
        translator = Translator()
        translation = translator.translate(text, src=src_lang, dest=dest_lang)
    return translation.text


# %%
cols_to_translate = [
    "Dimension",
    "Type Level 1",
    "Type Level 2",
    "Type Level 3",
    "Type Level 4",
    "Example",
    "Definition",
]
# translate taxonomy into English
print("Translating taxonomy into English...")
taxonomy_en = taxonomy.copy()
for col in cols_to_translate:
    taxonomy_en[col] = taxonomy_en[col].apply(translate_text)

# %%
