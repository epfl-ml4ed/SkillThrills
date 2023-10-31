# %%
import pandas as pd
import re
import spacy
from spacy.language import Language
from spacy_language_detection import LanguageDetector


def get_lang_detector(nlp, name):
    return LanguageDetector(seed=42)  # We use the seed 42


nlp_model = spacy.load("en_core_web_sm")
Language.factory("language_detector", func=get_lang_detector)
nlp_model.add_pipe("language_detector", last=True)


def detect_language(text):
    maximum = max(len(text), 150)
    doc = nlp_model(text[:maximum])
    detect_language = doc._.language
    return detect_language["language"]


def replace_html_tags(text):
    def replace_tags(_):
        nonlocal tag_count
        tag_count += 1
        if tag_count % 10 == 0:
            return ". "
        else:
            return " "

    tag_count = 0
    pattern = r"<.*?>"
    result = re.sub(pattern, replace_tags, text)
    return result


def load_data(type, lang="de", sample_size=100):
    if type == "job":
        df = pd.read_json("../raw/vacancies.json")
        df["fulltext"] = df["name"] + "\n" + df["description"]

    elif type == "course":
        df = pd.read_json("../raw/learning_opportunities.json")
        df = df[df["active"] == True]
        keep_ids = {1, 5, 9}
        df = df[df["study_ids"].apply(lambda x: bool(set(x) & keep_ids))]
        df["fulltext"] = (
            "TO_ACQUIRE: "
            + df["name"]
            + df["intro"].fillna("")
            + df["key_benefits"].fillna("")
            + df["learning_targets_description"].fillna("")
            + "\n REQUIRED: "
            + df["admission_criteria_description"].fillna("")
            + df["target_group_description"].fillna("")
        )
        # replace every 10 tags with a period to avoid too long sentences

    df["fulltext"] = df["fulltext"].apply(replace_html_tags)

    df["language"] = df["fulltext"].apply(detect_language)
    print(df["language"].value_counts())
    df = df[df["language"] == lang]

    df = df[["id", "fulltext"]].sample(sample_size, random_state=42)

    return df


def main():
    job_df = load_data("job", sample_size=100)
    course_df = load_data("course", sample_size=100)

    # save data
    job_df.to_json(
        "../processed/job_sample_100.json",
        orient="records",
        force_ascii=False,
    )
    course_df.to_json(
        "../processed/course_sample_100.json",
        orient="records",
        force_ascii=False,
    )


if __name__ == "__main__":
    main()
