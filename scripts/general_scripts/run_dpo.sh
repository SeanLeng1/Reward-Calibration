#!/bin/bash
set +e

# reward model list and whether to use chat template
declare -A MODELS                     
MODELS["stabilityai/stablelm-zephyr-3b"]="stabilityai/stablelm-3b-4e1t"                                       
MODELS["stabilityai/stablelm-2-zephyr-1_6b"]="stabilityai/stablelm-2-1_6b"  
MODELS["allenai/tulu-2-dpo-7b"]="allenai/tulu-2-7b"

# try all modes and each with and without confidence score
# chosen or rejected without probability is already covered by normal without probability
MODES=("chosen" "rejected") #"normal" "both_high" "both_low")
FORMAT_PROMPTS=("false" "true")
PROBABILITIES_SIDES=('right')

for PROBABILITIES_SIDE in "${PROBABILITIES_SIDES[@]}"
do
    PROB_SIDE_FLAG=""
    if [ "$PROBABILITIES_SIDE" = "left" ]; then
        PROB_SIDE_FLAG='_left'
    fi
    for FORMAT_PROMPT in "${FORMAT_PROMPTS[@]}"
    do
        FORMAT_PROMPT_FLAG=""
        SAVE_FOLDER="no_prompt"
        if [ "$FORMAT_PROMPT" = "true" ]; then
            FORMAT_PROMPT_FLAG="--format_prompt"
            SAVE_FOLDER="prompt"
        fi
        for MODEL_NAME in "${!MODELS[@]}"
        do

            # remove the prefix before the first slash
            POST_SLASH="${MODEL_NAME##*/}"
            CLEAN_MODEL_NAME="${POST_SLASH//\//-}"

            for MODE in "${MODES[@]}"
            do
                # Adjust ADD_PROBABILITY based on the mode
                if [ "$MODE" = "normal" ]; then
                    ADD_PROBABILITY=("false" "true")
                else
                    ADD_PROBABILITY=("true")
                fi

                for USE_PROBABILITY in "${ADD_PROBABILITY[@]}"
                do
                    ADD_PROBABILITY_FLAG=""
                    SCORE_NAME=""
                    NAME=""
                    if [ "$USE_PROBABILITY" = "true" ]; then
                        ADD_PROBABILITY_FLAG="--add_probabilities"
                        NAME="_probability"
                        SCORE_NAME="_with_probability"
                    fi

                    OUTPUT="reward_results/${SAVE_FOLDER}${PROB_SIDE_FLAG}/${CLEAN_MODEL_NAME}/"
                    mkdir -p $OUTPUT
                    
                    # check if jsons exist in the output directory, so we can avoid rerunning
                    SCORE_FILE="${OUTPUT}/scores${SCORE_NAME}_${MODE}.json"
                    # check if jsons exist in the output directory, so we can avoid rerunning
                    if [ -f "${SCORE_FILE}" ]; then
                        echo "Required file exists for ${OUTPUT}, mode ${MODE}, skipping..."
                        continue  # Skip to the next iteration of the loop
                    fi

                    CMD="python query/run_dpo.py \
                        --model $MODEL_NAME \
                        --seed 1234 \
                        --mode $MODE \
                        $ADD_PROBABILITY_FLAG \
                        --ref_model ${MODELS[$MODEL_NAME]} \
                        --batch_size 8 \
                        $FORMAT_PROMPT_FLAG \
                        --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training_${MODE}${NAME}.log"

                    echo "Running: $CMD"
                    eval $CMD || echo "An error occurred with $CMD but continuing with other tasks..."
                done
            done
        done
    done
done
