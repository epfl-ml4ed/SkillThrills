import pandas as pd
import ipdb
import evaluate
seqeval = evaluate.load("seqeval")

def calc_avg_score(dataset, filtered = False):
    precision = 0
    recall = 0
    skill_level_accuracy = 0
    f1 = 0

    for i in range(len(dataset)):
        row = dataset.iloc[i]
        try:
            score = row['seqeval_score']
            precision += score['precision']
            recall += score['recall']
            skill_level_accuracy += row['skill_level_accuracy']
            f1 += score['f1']
        except Exception as e:
            print(e, row['seqeval_score'])


    precision /= len(dataset)
    recall /=len(dataset)
    if filtered:
        skill_level_accuracy /= len(dataset)
        skill_level_accuracy = round(skill_level_accuracy, 3)
    else:
        skill_level_accuracy = None
    f1 /= len(dataset)

    res = {'precision': round(precision, 3), 'recall': round(recall, 3), 'f1': round(f1, 3), 'skill_level_accuracy': skill_level_accuracy}
    print(res)
    return res

def skill_level_accuracy(row):
    # for each span of row['skill_spans'], check if at least one word was retrieved.
    # if yes, add 1 to the count of correct spans
    # TODO compute this as precision and recall
    found_spans = 0
    nb_skills = len(row['skill_spans'])
    if nb_skills == 0:
        return None
    predicted_skills_indices = [index for index, pred in enumerate(row['list_of_selection']) if pred!='O']
    for span in row['skill_spans']:
        span = range(span[1][0], span[1][1]+1)
        # test if element of predicted_skills_indices is in span
        if any(found_item in span for found_item in predicted_skills_indices):
            found_spans += 1
    return found_spans/nb_skills


def seq_eval(row):
        score = seqeval.compute(predictions=[row['list_of_selection']], references=[row['tags_skill_clean']])
        try:
            score = score['ICT']
        except:
            score_renamed = {}
            for key, value in score.items():
                score_renamed[key.replace('overall_', '')] = value
            score = score_renamed
        return score

def eval(save_path):    
    df = pd.read_json(save_path)
    df['seqeval_score'] = df.apply(lambda row: seq_eval(row), axis=1)
    df['skill_level_accuracy'] = df.apply(lambda row: skill_level_accuracy(row), axis=1)
    df['has_item'] = df.apply(lambda row: len(row['skill_spans'])>0, axis=1)
    dataset_filtered = df[df['has_item'] == True]

    print('non-filtered avg:')
    res_dict = calc_avg_score(df, filtered = False)

    print('filtered avg:')
    res_dict_filtered = calc_avg_score(dataset_filtered, filtered = True)

    try:
        df.drop(columns='score', inplace=True)
    except:
        pass

    return res_dict, res_dict_filtered

    # df.to_json(path, orient='records', indent=4)




    
    