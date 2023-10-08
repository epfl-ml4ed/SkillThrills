PROMPT_TEMPLATES = {
    "gnehm": {
    "system": "You are an expert human resource from Germany. You need to analyse skills required in German job offers.",
    "instruction": {
        "ner": "You are given a sentence from a job description in German. Highlight all the skills and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'.\n",
        "extract":"You are given a sentence from a job description in German. Extract all the skills and competencies that are required from the candidate, printing one per line. If the sentence doesn't contain any skill, output \"None\".\n",
        }
    },
    "all": {
    "system": "You are an expert human resource. You need to analyse skills required in job offers.",
    "instruction": {
        "ner": "You are given a sentence from a job description. Highlight all the skills and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'.\n",
        "extract":"You are given a sentence from a job description. Extract all the skills and competencies that are required from the candidate, printing one per line. Make sure to keep the exact same words as found in the sentence. If the sentence doesn't contain any skill, output \"None\".\n",
        }
    },

}