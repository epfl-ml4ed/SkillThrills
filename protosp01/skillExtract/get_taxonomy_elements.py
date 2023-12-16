# %%
import os
import pandas as pd

# %%
from utils import *

# %%
def get_taxonomy():
    # Navigating to the folder where the data is stored
    os.chdir("../data/taxonomy/")
    assert os.getcwd().split("/")[-1] == "taxonomy", "check path"

    # Reading in the data
    taxonomy = pd.read_excel("KompetenzmodellKodierbuch.xlsx", sheet_name="V4")
    print("num taxonomy rows:", len(taxonomy))
    taxonomy = taxonomy.dropna(subset=["Type Level 1", "Type Level 2", "Definition"])
    print("num taxonomy rows after dropping empty Level 1 and 2:", len(taxonomy))
    taxonomy = taxonomy.dropna(axis=1, how="all")
   

    tech = pd.read_excel("KompetenzmodellKodierbuch.xlsx", sheet_name="Technologies")
    print("num tech rows:", len(tech))
    certif = pd.read_excel(
        "KompetenzmodellKodierbuch.xlsx", sheet_name="Certifications"
    )
    print("num certification rows:", len(certif))
    lang = pd.read_excel("KompetenzmodellKodierbuch.xlsx", sheet_name="Languages")
    print("num language rows:", len(lang))
    mastery = pd.read_excel(  # keep only 2nd column onwards
        "KompetenzmodellKodierbuch.xlsx", sheet_name="Expert Levels", usecols="B:C"
    )

    # clean mastery df
    mastery = mastery.dropna(how="all")
    # rename column "Unnamed: 1" to "Level 2"
    mastery = mastery.rename(columns={"Unnamed: 1": "Level 2"})
    mastery["Level 2"] = mastery["Level 2"].fillna(method="ffill")
    mastery["Level 3"] = mastery["Level 3"].str.strip()
    mastery["Level 3"] = mastery["Level 3"].str.replace("Kennntnisse", "Kenntnisse")
    mastery["Level 2"] = mastery["Level 2"].str.replace("Experte", "Expert")
    mastery = mastery.dropna()  # drop again to remove empty rows
    mastery = pd.concat(
        [mastery, mastery[mastery["Level 3"].str.contains(r"\(|/")]]
    ).sort_values(by=["Level 2", "Level 3"])
    mastery["Level 3"] = mastery["Level 3"].str.lower()

    # this joins three sheets vertically into one df
    tech_certif_lang = pd.concat([tech, certif, lang], ignore_index=True)
    tech_certif_lang.columns = tech_certif_lang.columns.str.strip()
    # for any row where the first column is empty, fill it with the value from the previous row
    tech_certif_lang["Level 1"] = tech_certif_lang["Level 1"].fillna(method="ffill")

    print("num tech_certif_lang rows:", len(tech_certif_lang))

    # Assigning unique ids
    ## Creating unique ids to each taxonomy element based on the most granular level available
    taxonomy.insert(
        0,
        "unique_id",
        taxonomy.apply(get_lowest_level, axis=1).astype("category").cat.codes,
    )

    ## Setting n=highest unique id in taxonomy to make ids increment from unique ids of taxonomy
    n = max(taxonomy["unique_id"]) + 1
    tech_certif_lang.insert(
        0, "unique_id", list([i + n for i in range(len(tech_certif_lang))])
    )

    print(
        "\nRange of unique ids in taxonomy:",
        min(taxonomy["unique_id"]),
        max(taxonomy["unique_id"]),
    )
    print(
        "Range of unique ids in tech_certif_lang:",
        min(tech_certif_lang["unique_id"]),
        max(tech_certif_lang["unique_id"]),
    )

    # pre-procress taxonomy
    taxonomy = taxonomy.dropna(subset=["Definition", "Type Level 2"])
    # taxonomy["name"] use lowest level name (function works by row)
    taxonomy["name"] = taxonomy.apply(get_lowest_level, axis=1)
    taxonomy["name+definition"] = taxonomy.apply(concatenate_cols_skillname, axis=1)

    keep_cols = [
        "unique_id",
        # "ElementID",
        # "Dimension",
        # "Type Level 1",
        # "Type Level 2",
        # "Type Level 3",
        # "Type Level 4",
        # "Example",
        "name",
        # "Definition",
        "name+definition",
    ]
    taxonomy = taxonomy[keep_cols]

    # Exporting processed data
    taxonomy.to_csv("taxonomy_V4.csv", index=False)
    tech_certif_lang.to_csv("tech_certif_lang.csv", index=False)
    mastery.to_csv("mastery_to_edit.csv", index=False)


if __name__ == "__main__":
    get_taxonomy()
# %%
