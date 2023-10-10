#!/bin/bash
for prompt_type in ner; #extract
do
    for data in  skillspan sayfullina gnehm kompetencer fijo; #green
    do
        echo "Processing $data"
        python main.py  --dataset_name $data --process --knn --run --sample 100 --prompt_type $prompt_type --shots 4
    done
done