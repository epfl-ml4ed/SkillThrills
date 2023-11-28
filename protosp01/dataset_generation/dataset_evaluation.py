import sys
sys.path.append("../skillExtract/")
import pandas as pd
from transformers import (AutoModel, AutoTokenizer)
from utils import load_taxonomy
from api_key import API_KEY
import pickle
from utils import embed_taxonomy
from tqdm.notebook import tqdm
tqdm.pandas()
from utils import (OPENAI,
                   Splitter,
                   select_candidates_from_taxonomy)
from api_key import API_KEY
import evaluate
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (precision_score, 
                             recall_score)
import numpy as np


ESCO_DIR = "../../esco/"

class Args():
    def __init__(self):
        self.taxonomy = ESCO_DIR + "tech_managment_taxonomy.csv" 

        ## RELATED TO PROMPT CREATION
        self.datapath = "remote-annotated-en"
        # self.candidates_method = "embeddings"  ## putting "rules" doesn't give the embeddings
        self.candidates_method = "mixed"
        self.shots = 6
        self.prompt_type = "skills"
        self.data_type = "en_job"
        
        ## RELATED TO CHAT GPT GENERATION
        self.max_tokens = 200      ## ?? default value but JobBERT suposedly takes 512
        self.api_key = API_KEY
        self.model = "gpt-3.5-turbo"
        self.temperature = 0       ## default val
        self.top_p = 1             ## default val
        self.frequency_penalty = 0 ## default val
        self.presence_penalty = 0  ## default val

args = Args()

## Loading the embedded taxonomy
emb_sh = "_jbEn"
with open(f"../../data/taxonomy/taxonomy_embeddings{emb_sh}.pkl", "rb") as f:
    emb_tax = pickle.load(f)
    emb_tax["name"] = emb_tax["name+definition"].apply(
        lambda x : x.split(" : ")[0]
    )

## ESCO technical skills
ESCO_DIR = "../../../esco/"
tech_skills = pd.read_csv(ESCO_DIR + "tech_managment_taxonomy_narrow.csv")
tech_skills_names = list(tech_skills.name.unique())
tech_skills["name+definition"] = tech_skills["name+defintion"]
tech_skills["Example"] = tech_skills["altLabels"]

with open(ESCO_DIR + "embedded_tech_management_tax.pkl", "rb") as emb:
    tech_emb_tax = pickle.load(emb)

word_emb = "jjzha/jobbert-base-cased"
word_emb_model = AutoModel.from_pretrained(word_emb)
word_emb_tokenizer = AutoTokenizer.from_pretrained(word_emb)




def fidelity(dataset, model_id='jjzha/jobbert-base-cased'):
    perplexity = evaluate.load("perplexity", module_type="metric")
    results = perplexity.compute(model_id=model_id,
                                add_start_token=False,
                                predictions=dataset)
    return results





def sentence_level_quality(dataset, label_key="label"):
    """
        dataset is given as a dataframe
    """
    def compute_cos_sim(entry):
        sentence_embedding = entry["embeddings"].detach().numpy()
        label = entry[label_key]
        if(label in known_label_set):
            label_embedding = emb_tax[emb_tax["name"] == label]["embeddings"].values[0].detach().numpy()
            return cosine_similarity(sentence_embedding, label_embedding)[0][0]

    known_label_set = set(emb_tax["name"].values)    
    ## compute the embeddings of the sentences
    dataset["embeddings"] = dataset["sentence"]\
                    .progress_apply(lambda st : \
                    word_emb_model(**word_emb_tokenizer(st, return_tensors="pt", max_length=512, padding=True, truncation=True))\
                    .last_hidden_state[:, 0, :]\
                    .detach()
                    )
    dataset["sim"] = dataset[["embeddings", label_key]].progress_apply(compute_cos_sim, axis=1)

    return dataset


def entity_level_quality_1_to_1(predictions, label_key):
    preds_gt = []
    tbp = []
    for tpred_item in tqdm(predictions):
        pred_item = tpred_item[0]
        tbp.append([pred_item[label_key], list(pred_item["matched_skills"].keys())])
        labels = eval(pred_item[label_key]) if type(pred_item[label_key]) == "str" else pred_item[label_key]
        for label in labels:
            if(label not in ["LABEL NOT PRESENT", "UNDERSPECIFIED"]):
                
                # ONE ENTRY PER LABEL
                predicted_labels = [
                    x["name+definition"].split(" : ")[0] for x in pred_item["matched_skills"].values()
                ]
                if(label in predicted_labels):
                    # print("-"*45)
                    # print("we have a match for :", label)
                    # print("in : ", predicted_labels)

                    preds_gt.append([label, label])
                else :
                    if(len(predicted_labels) > 0):
                        preds_gt.append([label, predicted_labels[0]])
                    else :
                        preds_gt.append([label, "LABEL NOT PRESENT"])
                
                # THIS VERSION HAS ONE PREDICTION ENTRY PER 
                # (LABEL, PREDICTION) AND THUS HAS WAY TOO 
                # MUCH ENTRY TO PREVIDE MEANINGFUL RESULTS
                # for matched_pred in predicted_labels:
                #     preds_gt.append([label, matched_pred])
    gts, preds = zip(*preds_gt)
    gts = np.array(gts)
    preds = np.array(preds)

    acc = np.sum((gts == preds).astype(int)) / gts.shape[0]

    print(tbp)

    precision_micro = precision_score(gts, preds, average="micro")
    precision_macro = precision_score(gts, preds, average="macro")
    recall_micro = recall_score(gts, preds, average='micro')
    recall_macro = recall_score(gts, preds, average="macro")
    return {
        "accuracy": acc,
        "precision_micro": precision_micro,
        "precision_macro": precision_macro,
        "recall_micro": recall_micro,
        "recall_macro": recall_macro,
        "F1_micro": (2*precision_micro*recall_micro / (precision_micro + recall_micro)),
        "F1_macro": (2*precision_macro*recall_macro / (precision_macro + recall_macro))
    }

def entity_level_quality_many_to_many(predictions, label_key):
    ## simply an accuracy based on Jaccard Coefficient
    jaccards = []
    for pred_item in predictions:
        labels = set(pred_item[0][label_key])
        preds = set(pred_item[0]["matched_skills"])
        jaccards.append(len(labels.intersection(preds)) / len(labels.union(preds)))
    
    return {
        'jaccard_accuracy':sum(jaccards) / len(jaccards)
    }

def entity_level_quality(predictions, label_key="label"):
    print("new")
    return {
        '1_to_1': entity_level_quality_1_to_1(predictions, label_key), 
        'many_to_many': entity_level_quality_many_to_many(predictions, label_key)
    }



def pipeline_prediction(dataset):
    """
        takes dataset as record list
    """
    args = Args()

    ## loading embeddings taxonomy less
    
    extraction_cost = 0
    matching_cost = 0
    ress = []
    for i, annotated_record in tqdm(enumerate(dataset)):

        ## EXTRACTION
        api = OPENAI(args, [annotated_record])
        sentences_res_list, cost = api.do_prediction("extraction")
        extraction_cost += cost
        

        ## CANDIDATE SELECTION
        if "extracted_skills" in sentences_res_list[0]:
            splitter = Splitter()
            max_candidates = 10
            for idxx, sample in enumerate(sentences_res_list):
                sample = select_candidates_from_taxonomy(
                    sample,
                    tech_skills,
                    splitter,
                    word_emb_model,
                    word_emb_tokenizer,
                    max_candidates,
                    method=args.candidates_method,
                    emb_tax=None if args.candidates_method == "rules" else tech_emb_tax,
                )
                sentences_res_list[idxx] = sample


        if("skill_candidates" in sentences_res_list[0]):
            api = OPENAI(args, sentences_res_list)
            sentences_res_list, cost = api.do_prediction("matching")
            matching_cost += cost



        ress.append(sentences_res_list)
    return ress

