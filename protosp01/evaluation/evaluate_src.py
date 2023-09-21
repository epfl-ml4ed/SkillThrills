from datasets import load_metric
import pandas as pd


def calc_avg_score(dataset):
    precision = 0
    recall = 0
    # accuracy = 0
    f1 = 0

    for i in range(len(dataset)):
        row = dataset.iloc[i]
        try:
            score = row['seqeval_score']
            precision += score['precision']
            recall += score['recall']
            # accuracy += score['accuracy']
            f1 += score['f1']
        except:
            print(row['seqeval_score'])
            input('press enter to continue')

    precision /= len(dataset)
    recall /=len(dataset)
    # accuracy /= len(dataset)
    f1 /= len(dataset)

    print(f'precision: {round(precision, 3)}')
    print(f'recall: {round(recall, 3)}')
    print(f'f1: {round(f1, 3)}')


def eval(args):
    metric = load_metric("seqeval")
    df = pd.read_json(args.save_path)

    def seq_eval(row):
        score = metric.compute(predictions=[row['list_of_selection']], references=[row['tags_skill']])
        try:
            score = score['ICT']
        except:
            score_renamed = {}
            for key, value in score.items():
                score_renamed[key.replace('overall_', '')] = value
            score = score_renamed
        return score
                               
    df['seqeval_score'] = df.apply(lambda row: seq_eval(row), axis=1)
    df['has_item'] = df.apply(lambda row: any(row['tags_skill_boolean']), axis=1)
    dataset_filtered = df[df['has_item'] == True]

    print('non-filtered avg:')
    calc_avg_score(df)

    print('filtered avg:')
    calc_avg_score(dataset_filtered)

    try:
        df.drop(columns='score', inplace=True)
    except:
        pass

    # df.to_json(path, orient='records', indent=4)




    
    