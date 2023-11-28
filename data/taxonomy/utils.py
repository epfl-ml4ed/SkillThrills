COLAB = True

import jupyterannotate
from ipywidgets import Dropdown, widgets, interact, Layout, HBox, VBox
from IPython.display import display
from IPython.display import HTML
import json
import pandas as pd


def load_data(data_type):
    if data_type != "job" and data_type != "course":
        raise ValueError("data_type must be 'job' or 'course'")
    if COLAB:
        df = pd.read_json(f"{data_type}_sample_100.json")
    else:
        df = pd.read_json(f"../processed/{data_type}_sample_100.json")
    return df


def get_lowest_level(row):
    """
    Returns the lowest level of the taxonomy that is not NaN in each
    """
    for level in ["Type Level 4", "Type Level 3", "Type Level 2", "Type Level 1"]:
        value = row[level]
        if not pd.isna(value):
            return value
            # appending level also just in case different levels have the same name


def load_taxonomy(level="lowest"):
    if COLAB:
        df = pd.read_csv("taxonomy_V4.csv", sep=",")
    else:
        df = pd.read_csv("../taxonomy/taxonomy_V4.csv", sep=",")
    df = df.dropna(subset=["Definition", "Type Level 2"])
    keep_cols = [
        "Type Level 1",
        "Type Level 2",
        "Type Level 3",
        "Type Level 4",
    ]
    df = df[keep_cols]
    if level == "lowest":
        list_levels = list(set(df.apply(get_lowest_level, axis=1)))
    if level == "level2":
        list_levels = list(set(df["Type Level 2"]))
    list_levels = list_levels + ["NONE", "ADD_NEW", "KNOWLEDGE"]

    return list_levels


def doc_widget(jc):
    text_input = widgets.BoundedIntText(
        value=1,  # Initial value
        min=1,  # Minimum value
        max=100,  # Maximum value
        description=f"{jc} #:",
        layout=widgets.Layout(width="70%"),
    )
    text_input.layout.width = "30%"
    style = widgets.HTML(
        "<style>.widget-text .widget-label, .widget-text input {font-size: 20px; font-weight: bold;}</style>"
    )

    return style, text_input


def job_course_widget():
    dropdown = widgets.Dropdown(
        options=["Job", "Course"], description="Select Job or Course:", disabled=False
    )
    dropdown.layout.width = "30%"
    dropdown.style.description_width = "initial"

    display(dropdown)
    display(HTML("<style>.widget-label { font-size: 20px; font-weight: bold;}</style>"))

    return dropdown


def get_skills_per_doc(span):
    list_of_skills = []
    for skill in span:
        text = skill["text"]
        text = text.strip()
        list_of_skills.append(text)
    skills = [{"example": skill} for skill in list_of_skills]
    df = pd.DataFrame(skills)
    return df


# df = load_data("job")  # TODO! update to variable

LEVEL_LABELS = ["Unknown", "Beginner", "Intermediate", "Expert"]
TAX_ELEMENTS = load_taxonomy("lowest")


def extraction_step(text_input, jc, doc_idx, DOCUMENTS):
    text_widget = widgets.HTML(
        value=f"<h2 style='font-size: 20px; font-weight: bold;'>Annotating document {text_input.value} of 100 "
        f"( {jc} ID: {DOCUMENTS[doc_idx]['id']})</h2>",
    )

    display(text_widget)

    extraction_widget = jupyterannotate.AnnotateWidget(
        docs=DOCUMENTS[doc_idx]["fulltext"],
        labels=LEVEL_LABELS,
        # change size of text
    )

    return extraction_widget


def save_extractions(doc_idx, extraction_widget, jc, DOCUMENTS):
    extractions_list = []
    try:
        extractions_list.append(
            {
                "sample_num": doc_idx,  # zero-index 0-99
                "doc_id": DOCUMENTS[doc_idx]["id"],
                "doc_type": jc,
                "extraction": extraction_widget.spans[0],
            }
        )
    except:
        print("No terms highlighted from extraction step")

    try:
        with open("anno_extractions.json", "r", encoding="utf-8") as f:
            extractions = json.load(f)
    except FileNotFoundError:
        extractions = []

    updated = False
    for idx, extraction in enumerate(extractions):
        if extractions_list[0]["doc_id"] == extraction["doc_id"]:
            extractions[idx] = extractions_list[0]
            updated = True
            break

    if not updated:
        extractions.append(extractions_list[0])

    # Writing the updated/modified data back to the file
    with open("anno_extractions.json", "w", encoding="utf-8") as f:
        json.dump(extractions, f, ensure_ascii=False)


def matching_step(extraction_widget, doc_idx, text_input, jc, DOCUMENTS):
    user_inputs = {}
    widget_sets = []

    text_widget = widgets.HTML(
        value=f"<h2 style='font-size: 20px; font-weight: bold;'>Annotating document {text_input.value} of 100 "
        f"({jc} ID: {DOCUMENTS[doc_idx]['id']})</h2>",
    )

    display(text_widget)
    try:
        for index, item in enumerate(extraction_widget.spans[0]):
            # create a text widget
            text_widget = widgets.HTML(
                value=f"<h2 style='font-size: 18px; font-weight: bold;'>{item['text']}</h2>",
            )

            req_v_optional = widgets.RadioButtons(
                options=["SELECT_BELOW", "Unknown", "Required", "Optional"],
                description="Required or Optional Skill:",
                disabled=False,
            )

            # add checkbox default not selected but can check called implicit
            implicit_skill = widgets.Checkbox(
                value=False,
                description="Implicit Skill",
                disabled=False,
                indent=False,
            )

            unsure_skill = widgets.Checkbox(
                value=False,
                description="Unsure about Skill",
                disabled=False,
                indent=False,
            )

            implicit_level = widgets.Checkbox(
                value=False,
                description="Implicit Level",
                disabled=False,
                indent=False,
            )

            unsure_level = widgets.Checkbox(
                value=False,
                description="Unsure about Level",
                disabled=False,
                indent=False,
            )

            match_1 = widgets.Combobox(
                options=TAX_ELEMENTS,
                placeholder="Select or type to add",
                ensure_option=True,
                description="Label 1:",
            )

            match_1_ = widgets.Combobox(
                options=TAX_ELEMENTS,
                placeholder="NONE",
                ensure_option=True,
                description="Label 1:",
            )

            match_2 = widgets.Combobox(
                options=TAX_ELEMENTS,
                placeholder="NONE",
                ensure_option=True,
                description="Label 2:",
            )

            match_2_ = widgets.Combobox(
                options=TAX_ELEMENTS,
                placeholder="NONE",
                ensure_option=True,
                description="Label 2:",
            )

            match_3 = widgets.Combobox(
                options=TAX_ELEMENTS,
                placeholder="NONE",
                ensure_option=True,
                description="Label 3:",
            )

            match_3_ = widgets.Combobox(
                options=TAX_ELEMENTS,
                placeholder="NONE",
                ensure_option=True,
                description="Label 3:",
            )

            match_1.layout.width = "80%"
            match_2.layout.width = "80%"
            match_3.layout.width = "80%"
            match_1_.layout.width = "80%"
            match_2_.layout.width = "80%"
            match_3_.layout.width = "80%"

            checkbox_skill_group = VBox([implicit_skill, unsure_skill])
            checkbox_level_group = VBox([implicit_level, unsure_level])
            options_group = HBox(
                [
                    req_v_optional,
                    checkbox_skill_group,
                    checkbox_level_group,
                ]
            )
            widget_group = VBox(
                [
                    text_widget,
                    options_group,
                    match_1,
                    match_1_,
                    match_2,
                    match_2_,
                    match_3,
                    match_3_,
                ]
            )
            display(widget_group)

            # Save each widget set with the associated text item
            widget_sets.append(
                {
                    "text": item["text"],
                    "widgets": (
                        req_v_optional,
                        implicit_skill,
                        unsure_skill,
                        implicit_level,
                        unsure_level,
                        match_1,
                        match_1_,
                        match_2,
                        match_2_,
                        match_3,
                        match_3_,
                    ),
                }
            )
    except:
        print("No skills were selected in the previous step")

    label_keys = [
        "req_status",
        "implicit_skill",
        "unsure_skill",
        "implicit_level",
        "unsure_level",
        "match_1",
        "match_1s",
        "match_2",
        "match_2s",
        "match_3",
        "match_3s",
    ]

    def on_value_change(change):
        for widget_set in widget_sets:
            labels = {}
            for idx, widget in enumerate(widget_set["widgets"]):
                labels[label_keys[idx]] = widget.value
            user_inputs[widget_set["text"]] = labels

    # Capture and save user inputs when the widget values change
    for widget_set in widget_sets:
        for widget in widget_set["widgets"]:
            widget.observe(on_value_change, names="value")

    def submit_button_matching(_):
        nonlocal user_inputs  # Access the outer variable

        # Load anno_extractions.json
        try:
            with open("anno_extractions.json", "r", encoding="utf-8") as f:
                extractions = json.load(f)
        except FileNotFoundError:
            print("Error: Cannot find anno_extractions.json in current directory")

        # Load anno_matching.json
        try:
            with open("anno_matching.json", "r", encoding="utf-8") as f:
                matching = json.load(f)
        except FileNotFoundError:
            matching = []  # Set to empty list if file doesn't exist

        new_extractions = []
        for doc in extractions:
            for extraction in doc["extraction"]:
                text = extraction["text"]
                if text in user_inputs:
                    extraction.update(user_inputs[text])
                    new_extractions.append(doc)
        # Update matching based on extractions
        for extraction in new_extractions:
            for entry in matching:
                if entry["doc_id"] == extraction["doc_id"]:
                    entry["extraction"] = extraction["extraction"]
                    break
            else:
                matching.append(extraction)

        # Write the updated matching to anno_matching.json
        with open("anno_matching.json", "w", encoding="utf-8") as f:
            json.dump(matching, f, ensure_ascii=False)

    # Create a submit button
    submit_button = widgets.Button(description="Submit", button_style="success")

    # Assign the submit_button_matching function to the button's on_click event
    submit_button.on_click(submit_button_matching)

    # Display the button widget
    display(VBox([submit_button]))
