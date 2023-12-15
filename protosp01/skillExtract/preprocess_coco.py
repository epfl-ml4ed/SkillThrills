# %%
import pandas as pd
import argparse
import os
import tiktoken
import codecs
import json
import re
from utils import *

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, help="Language to keep", default="all")
parser.add_argument("--input_path", type=str, help="Input directory", default="../data/raw/coco_courses/courses_latest.csv")
parser.add_argument("--output_dir", type=str, help="Output directory", default="../data/processed")
parser.add_argument("--model", type=str, help="Model to use for generation", default="gpt-3.5-turbo")
parser.add_argument("--sample_size", type=int, help="Number of samples to generate", default=0)

args = parser.parse_args()


# fmt: on
def longest_sentence(sentences):
    return max(sentences, key=len)


def num_tokens_from_string(stringl):
    encoding = tiktoken.encoding_for_model(args.model)
    return len(encoding.encode(stringl))


def decode_unicode(text):
    return codecs.decode(text, "unicode_escape")


def clean_non_ascii(text):
    pattern = re.compile("[^\x00-\x7F]+")
    return pattern.sub("", text)


def convert_to_list(value):
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def join_strings(x):
    if isinstance(x, list):
        return " ".join(x)
    elif isinstance(x, str):
        return x
    else:
        return ""


def subsample(df, n):
    # get unique course ids
    course_ids = df["id"].unique()
    # sample n course ids
    np.random.seed(42)
    sampled_course_ids = np.random.choice(course_ids, size=n, replace=False)
    # filter df by sampled course ids
    return df[df["id"].isin(sampled_course_ids)]


def num_words_from_string(stringl):
    return len(stringl.split())


def main():
    # %%

    data = pd.read_csv(
        "../data/raw/coco_courses/course_latest.csv", index_col=0, encoding="utf-8"
    )
    print("num rows before filtering:", len(data))

    # rename course_id to id
    data.rename(columns={"course_id": "id"}, inplace=True)

    language_dict = {
        "english": "en",
        "portuguese": "pt",
        "spanish": "es",
        "turkish": "tr",
        "russian": "ru",
        "german": "de",
        "arabic": "ar",
        "japanese": "ja",
        "bulgarian": "bg",
        "hindi": "hi",
        "chinese": "zh",
        "urdu": "ur",
        "hungarian": "hu",
        "croatian": "hr",
        "french": "fr",
        "afrikaans": "af",
        "thai": "th",
        "uzbek": "uz",
        "korean": "ko",
        "italian": "it",
        "finnish": "fi",
        "polish": "pl",
        "hebrew": "he",
        "rumanian": "ro",
        "dutch": "nl",
        "slovak": "sk",
        "telugu": "te",
        "indonesian": "id",
        "vietnamese": "vi",
        "swedish": "sv",
        "filipino": "tl",
        "latvian": "lv",
        "ukranian": "uk",
        "persian": "fa",
        "norwegian": "no",
        "greek": "el",
        "estonian": "et",
        "marathi": "mr",
        "serbian": "sr",
        "czech": "cs",
        "malay": "ms",
        "catalan": "ca",
        "albanian": "sq",
        "danish": "da",
        "bengalese": "bn",
        "azeri": "az",
    }

    # replace the language column with the language code
    data["language"] = data["language"].apply(lambda x: language_dict[x])

    print("language:", args.language)
    print("sample size:", args.sample_size)
    if args.language != "all":
        data = data[data["language"] == args.language]
        print(f"num rows after filtering by language ({args.language}):", len(data))

    # %%
    # drop non-IT/mgmt courses
    first_level_certain = ["development", "it-and-software", "office-productivity"]
    second_level_to_add = [
        "project-management",
        "data-and-analytics",
        "web-design",
        "user-experience",
        "search-engine-optimization",
        "analytics-and-automation",
    ]

    df_1 = data[data["first_level_category"].isin(first_level_certain)]
    df_2 = data[data["second_level_category"].isin(second_level_to_add)]

    data = pd.concat([df_1, df_2], ignore_index=True)

    print("num rows after filtering by category:", len(data))
    # %%
    # Apply the conversion function to multiple columns in the DataFrame
    text_cols = [
        "short_description",
        # "long_description",
        "objectives",
        "requirements",
        # "target_audience",
    ]

    for col in text_cols:
        data[col] = data[col].apply(convert_to_list)

    for col in text_cols:
        data[col] = data[col].apply(join_strings)

    # %%
    acq_data = data.copy()
    acq_data["fulltext"] = (
        acq_data["short_description"].fillna("")
        # + acq_data["long_description"].fillna("")
        + acq_data["objectives"].fillna("")
    )
    acq_data["skill_type"] = "to_acquire"

    req_data = data.copy()
    req_data["fulltext"] = req_data["requirements"].fillna("")
    req_data["skill_type"] = "required"

    # create data with both required and to acquire skills sorted by course_id
    data = pd.concat([acq_data, req_data], ignore_index=True)
    data.sort_values(by=["id", "skill_type"], inplace=True)

    print("number of rows after combining required and to acquire skills:", len(data))
    # %%
    # heuristics to clean fulltext
    data["fulltext"] = data["fulltext"].apply(replace_html_tags)
    data["fulltext"] = data["fulltext"].apply(clean_non_ascii)
    data["fulltext"] = data["fulltext"].apply(decode_unicode)

    # %%
    language = "en" if args.language == "all" else args.language

    # filter out ids with too short fulltext
    data["num_words"] = data["fulltext"].apply(num_words_from_string)
    print("--avg num words:", data["num_words"].mean())

    short_text_ids = data[data["num_words"] < 20]["id"].unique()
    data = data[~data["id"].isin(short_text_ids)]
    print("number of rows after dropping short texts:", len(data))

    # filter out ids with too many words in a sentence
    data["sentences"] = data["fulltext"].apply(split_sentences, language=language)
    data["longest_sentence"] = data["sentences"].apply(longest_sentence)
    print("--avg sentence length:", data["longest_sentence"].apply(len).mean())
    data["num_tokens"] = data["longest_sentence"].apply(num_tokens_from_string)
    print("--avg num tokens:", data["num_tokens"].mean())

    long_sentence_ids = data[data["num_tokens"] > 1500]["id"].unique()
    data = data[~data["id"].isin(long_sentence_ids)]
    print("number of rows after dropping long sentences:", len(data))

    drop_new_cols = ["sentences", "longest_sentence", "num_tokens"]
    data.drop(columns=drop_new_cols, inplace=True)

    _sample = ""
    # %%

    if args.sample_size > 0:
        data = subsample(data, args.sample_size)
        _sample = f"_sample{args.sample_size}"
        print(f"num rows after sampling ({args.sample_size}):", len(data))

    # save data to disk
    output_path = os.path.join(
        args.output_dir, f"course_coco_{args.language}{_sample}.csv"
    )
    data.to_csv(output_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
