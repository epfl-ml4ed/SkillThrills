import pandas as pd
import ipdb
import evaluate
seqeval = evaluate.load("seqeval")

def calc_avg_score(dataset, filtered = False):    
    #seqeval_values = dataset['seqeval_score'].apply(lambda x: list(x.values())).apply(pd.Series)
    skillscore_values = dataset['skill_level_score'].apply(lambda x: list(x.values())).apply(pd.Series)

    #seqeval_values.columns = ['precision', 'recall', 'f1']
    skillscore_values.columns = ['skillevel_precision', 'skillevel_recall', 'skillevel_f1']
    if filtered:
        skillscore_values.columns = ['filtered_skillevel_precision', 'filtered_skillevel_recall', 'filtered_skillevel_f1']

    # Compute the mean of each element across all rows
    #mean_seqeval_values = seqeval_values.mean().apply(lambda x: round(x, 3)).to_dict()
    mean_skillscore_values = skillscore_values.mean().apply(lambda x: round(x, 3)).to_dict()
    return mean_skillscore_values

def skill_level_metrics(row):
    # for each span of row['skill_spans'], check if at least one word was retrieved.

    TP = 0
    FP = 0
    FN = 0

    nb_skills = len(row['skill_spans'])
    nb_predicted_skills = row['list_of_selection'].count('B')
    if nb_skills != 0:
        predicted_skills_indices = [index for index, pred in enumerate(row['list_of_selection']) if pred!='O']
        for span in row['skill_spans']:
            span = range(span[1][0], span[1][1]+1)
            # test if element of predicted_skills_indices is in span
            if any(found_item in span for found_item in predicted_skills_indices):
                TP += 1
            else:
                FN += 1
        FP = nb_predicted_skills - TP 

    # Calculate Precision and Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {'precision': precision, 'recall': recall, 'f1': f1_score}


def seq_eval(row):
    score = seqeval.compute(predictions=[row['list_of_selection']], references=[row['tags_skill_clean']])
    if 'precision' not in score:
        score = {k.replace('overall_', ''): v for k, v in score.items()}
    score = {k:v for k, v in score.items() if k in ['precision', 'recall', 'f1']}
    return score

def eval(save_path):    
    df = pd.read_json(save_path)
    all_metrics = {}
    print("Removing examples where the number of spans and the number of predictions don't match:", len(df[df['list_of_selection'].apply(len) != df['tags_skill_clean'].apply(len)]))
    df = df[df['list_of_selection'].apply(len) == df['tags_skill_clean'].apply(len)]
    
    # TODO raise error

    seqeval_score = seqeval.compute(predictions=df['list_of_selection'].values.tolist(), references=df['tags_skill_clean'].values.tolist())
    
    seqeval_score = {k.replace('overall_', 'seqeval_'):v for k, v in seqeval_score.items() if k in ['overall_precision', 'overall_recall', 'overall_f1']}
    all_metrics.update(seqeval_score)

    df['skill_level_score'] = df.apply(lambda row: skill_level_metrics(row), axis=1)
    df['has_item'] = df.apply(lambda row: len(row['skill_spans'])>0, axis=1)
    dataset_filtered = df[df['has_item'] == True]

    print('non-filtered avg:')
    res_dict = calc_avg_score(df, filtered = False)
    all_metrics.update(res_dict)
    print('seqeval_score:', seqeval_score)
    print('skill_level_scores:', res_dict)

    print('filtered avg:')
    res_dict_filtered = calc_avg_score(dataset_filtered, filtered = True)
    all_metrics.update(res_dict_filtered)
    seqeval_score_filtered = seqeval.compute(predictions=dataset_filtered['list_of_selection'].values.tolist(), references=dataset_filtered['tags_skill_clean'].values.tolist())
    seqeval_score_filtered = {k.replace('overall_', 'filtered_seqeval_'):v for k, v in seqeval_score_filtered.items() if k in ['overall_precision', 'overall_recall', 'overall_f1']}
    all_metrics.update(seqeval_score_filtered)

    print('seqeval_score_filtered:', seqeval_score_filtered)
    print('skill_level_scores_filtered:', res_dict_filtered)    
    return all_metrics

    # df.to_json(path, orient='records', indent=4)
    # TODO save number of elements as well




    
    