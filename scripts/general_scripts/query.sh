#!/bin/bash
set +e

# whether to use chat_template or not
declare -A MODELS
MODELS["OpenLLMAI/Llama-3-8b-sft-mixture"]="true"   
MODELS["OpenLLMAI/Llama-3-8b-rlhf-100k"]="true"            


# SFT Model

declare -A DATASET_NAME
DATASET_NAME['TruthfulQA']="multi_choice_qa"
DATASET_NAME['GSM8K']="open_number_qa"
DATASET_NAME['CommonsenseQA']="multi_choice_qa"
DATASET_NAME['SciQ']="multi_choice_qa"
DATASET_NAME['BigBench']="open_number_qa"
DATASET_NAME['professional']="multi_choice_qa"

declare -A DATASET_PATH
DATASET_PATH['TruthfulQA']="dataset/truthfulqa"
DATASET_PATH['GSM8K']="dataset/grade_school_math/data/test.jsonl"
DATASET_PATH['CommonsenseQA']="dataset/"
DATASET_PATH['SciQ']="dataset/"
DATASET_PATH['BigBench']="dataset/BBH/object_counting/task.json"
DATASET_PATH['professional']="dataset/MMLU/professional"

# in case the tokenizer does not have chat template, which template to use (Default, Ziya, Alapaca)
# Default will use tokenizer chat template
declare -A CONV_TYPE
CONV_TYPE["OpenLLMAI/Llama-3-8b-rlhf-100k"]="Default"            
CONV_TYPE["OpenLLMAI/Llama-3-8b-sft-mixture"]="Default"  


USE_COT=true # use cot or not
USE_TOPK=false # use topk or not
PROMPT_TYPE="vanilla"
MAX_NEW_TOKENS=16

for DATASET_NAME in "${!DATASET_NAME[@]}"
do
    TASK_TYPE=${DATASET_NAME[$DATASET_NAME]}
    DATASET_PATH=${DATASET_PATH[$DATASET_NAME]}
    
    # Try only use batch size = 1 to avoid floating point instability
    for MODEL_NAME in "${!MODELS[@]}"
    do
        USE_CHAT_TEMPLATE_FLAG=""
        if [ "${MODELS[$MODEL_NAME]}" = "true" ]; then
            USE_CHAT_TEMPLATE_FLAG="--use_chat_template"
        fi

        if [ "$USE_COT" = true ] ; then
            USE_COT_FLAG="--use_cot"
            PROMPT_TYPE="cot"
            MAX_NEW_TOKENS=256
        fi

        if [ "$USE_TOPK" = true ] ; then
            USE_TOPK_FLAG="--use_top_k"
            PROMPT_TYPE="topk"
            MAX_NEW_TOKENS=256
        fi

        # remove the prefix before the first slash
        POST_SLASH="${MODEL_NAME##*/}"
        CLEAN_MODEL_NAME="${POST_SLASH//\//-}"
        
        OUTPUT="verbalized_output/$PROMPT_TYPE/$CLEAN_MODEL_NAME/$DATASET_NAME"
        mkdir -p $OUTPUT

        # check if results.json exists in the output directory, so we can avoid rerunning
        if [ -f "$OUTPUT/results.json" ]; then
            echo "Results already exist for $OUTPUT, skipping..."
            continue  # Skip the rest of the loop and move to the next iteration
        fi


        master_port=$((RANDOM % 5000 + 20000))
        CMD="accelerate launch --main_process_port $master_port query/query.py \
        --dataset $DATASET_NAME \
        --data_path $DATASET_PATH \
        --model_name_or_path $MODEL_NAME \
        --task_type $TASK_TYPE \
        --seed 1234 \
        --dtype auto \
        --per_device_eval_batch_size 1 \
        --max_new_tokens $MAX_NEW_TOKENS \
        $USE_COT_FLAG \
        $USE_TOPK_FLAG \
        $USE_CHAT_TEMPLATE_FLAG \
        --conv ${CONV_TYPE[$MODEL_NAME]} \
        --temperature 1.0 \
        --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log"

        echo "Running: $CMD"
        eval $CMD || echo "An error occurred with $CMD but continuing with other tasks..."
    done
done
