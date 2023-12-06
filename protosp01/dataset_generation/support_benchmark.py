from dataset_evaluation import Predictor
import pandas as pd
import pickle
import os




def main():
    ESCO_DIR = "../../../esco/"
    with open(ESCO_DIR + "dev.json") as f:
        devsp = eval(",".join(f.read().split("\n")))
    devsp = pd.DataFrame(devsp).drop("idx", axis=1)
    devsp.columns = ["sentence", "skills"]
    devsp = devsp[["skills", "sentence"]]

    devsp = devsp.to_dict(orient='records')

    print("Number of dev sp samples : ", len(devsp))
    predictor = Predictor(
            test_domain="SkillSpan-dev", ## domain for the test samples 
            train_domain="SkillSpan", ## domain of the demonstration retrieval
            candidates_method="mixed"
        )
    
    all_res = []

    log = ""

    all_kwargs = [
        { ## BASELINE
            'dataset': devsp,
            'support_type': None
        },
        { ## Matching 1-10
            'dataset': devsp,
            "support_type": "kNN",
            "support_size_match": 1,
            "nb_candidates": 10
        },
        { ## Extr 5
            'dataset': devsp,
            "support_size_extr": 5,
        },
        { ## Extr 7
            'dataset': devsp,
            "support_size_extr": 7,
        },
        { ## Extr 10
            'dataset': devsp,
            "support_size_extr": 10,
        },
        { ## MATCH 1-10, Extr 7
            'dataset': devsp,
            "support_type": "kNN",
            "support_size_extr": 7,
            "support_size_match": 1,
            "nb_candidates": 10
        },
        { ## MATCH 3-10, Extr 7
            'dataset': devsp,
            "support_type": "kNN",
            "support_size_extr": 7,
            "support_size_match": 3,
            "nb_candidates": 10
        },
    ]
    
    for kwargs in all_kwargs:
        try :
            all_res.append(bootstrap(predictor, kwargs))
        except Exception as e:
            log += "Failed for : " +  str(kwargs) + "\n"
            log += str(e) + "\n"
            continue

    print(log)
            

    
    
    with open("BENCHMARK_BOOTSTRAP_ADDI.pkl", "wb") as f:
        pickle.dump(all_res, f)



def bootstrap(predictor, pred_kwargs):
    spec_res = []
    for _ in range(10):
        res1 = predictor.pipeline_prediction(**pred_kwargs) ## BASELINE
        spec_res.append(compute_metrics(res1))
    fname = prepare_kwargs(pred_kwargs, content="res")
    with open(fname, "wb") as f:
        pickle.dump(spec_res, f)
    fname = prepare_kwargs(pred_kwargs, content="preds")
    with open(fname, "wb") as f:
        pickle.dump(res1, f)
    return spec_res

    
def prepare_kwargs(pred_kwargs, content):
    fname = content
    for k, v in list(pred_kwargs.items())[1:]:
        fname += f"_{k}_{v}"
    return fname + ".pkl"

def compute_metrics(preds):
    TP, FN, FP = computed_confusion_matrix(preds)
    R = TP / (TP + FN) if(TP + FN != 0) else 0
    P = TP / (TP + FP) if(TP + FP != 0) else 0
    
    # print("Precision : ", P)
    # print("Recall : ", R)
    # print("F1 : ", 2*P*R/(P + R))
    return R, P, (2*P*R/(P + R) if ((P + R) != 0) else 0)



def computed_confusion_matrix(preds):
    TP = 0
    TPs = []
    FN = 0
    FNs = []
    FP = 0
    FPs = []
    for item in preds:
        titem = item[0]
        
        label_skills = titem["skills"]
        matched_sk = []
        for mskill in titem["matched_skills"]:

            skill_item = titem['matched_skills'][mskill]
            skill_name = skill_item["name+definition"].split(" : ")[0]
            if(skill_name in titem["skills"]):
                TP += 1 ## we predicted a label and indeed in the target ==> TP
                TPs.append([skill_name, titem["skills"]])
            else :
                FP += 1 ## we predicted a lebl as positive but it's false
                FPs.append([skill_name, titem["skills"]])
            matched_sk.append(skill_item)
        for skill in titem["skills"]:
            if(skill != "UNK"):
                if(skill not in matched_sk):
                    FN += 1 ## we predicted a label as not in the skills but it it ==> false negative
                    FNs.append([skill, matched_sk])

    return TP, FN, FP


if __name__ == "__main__":
    main()
            