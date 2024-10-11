#!/bin/bash
set +e

declare -A DATASET_NAMES
DATASET_NAMES['TruthfulQA']="multi_choice_qa"
DATASET_NAMES['GSM8K']="open_number_qa"
DATASET_NAMES['CommonsenseQA']="multi_choice_qa"
DATASET_NAMES['SciQ']="multi_choice_qa"
DATASET_NAMES['professional']="multi_choice_qa"
DATASET_NAMES['BigBench']="open_number_qa"

MODEL_NAMES=(
    "allenai/tulu-2-dpo-7b" 
    "allenai/tulu-2-7b" 
)

PROMPT_TYPE="vanilla2"
USE_COT="false" # use cot or not

for DATASET_NAME in "${!DATASET_NAMES[@]}"
do
    TASK_TYPE=${DATASET_NAMES[$DATASET_NAME]}

    for MODEL_NAME in "${MODEL_NAMES[@]}"
    do
        POST_SLASH="${MODEL_NAME##*/}"
        CLEAN_MODEL_NAME="${POST_SLASH//\//-}"

        USE_COT_FLAG=""
        if [ "$USE_COT" = "true" ] ; then
            USE_COT_FLAG="--use_cot"
            PROMPT_TYPE="cot"
        fi

        OUTPUT="verbalized_output/$PROMPT_TYPE/$CLEAN_MODEL_NAME/$DATASET_NAME"
        if [ -d "${OUTPUT}" ]; then
            echo "Process ${OUTPUT}"
            RESULTS_FILE="$OUTPUT/results.json"

            VISUAL_FOLDER="$OUTPUT/visuals"
            mkdir -p $VISUAL_FOLDER

            python parse_eval.py \
                --input_file $RESULTS_FILE \
                $USE_COT_FLAG \
                --dataset $DATASET_NAME \
                --metric_folder $OUTPUT \
                --task_type $TASK_TYPE \
                --model_name $CLEAN_MODEL_NAME \
                --visual_folder $VISUAL_FOLDER 2> >(tee $VISUAL_FOLDER/err.log >&2) | tee $VISUAL_FOLDER/logging.log 
        fi
    done
done
