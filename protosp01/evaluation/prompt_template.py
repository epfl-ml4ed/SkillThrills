PROMPT_TEMPLATES = {
    "gnehm": {
    "system": "You are an expert human resource from Germany. You need to analyse skills required in German job offers.",
    "instruction": {
        "ner": "You are given a sentence from a job description in German. Highlight all the skills and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'.\n",
        "extract":"You are given a sentence from a job description in German. Extract all the skills and competencies that are required from the candidate as list.\n",
        }
    },
    "skillspan": {
    "system": "You are an expert human resource. You need to analyse skills required in job offers.",
    "instruction": {
        "ner": "You are given a sentence from a job description. Highlight all the skills and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'.\n",
        "extract":"You are given a sentence from a job description. Extract all the skills and competencies that are required from the candidate as list.\n",
        }
    },

}