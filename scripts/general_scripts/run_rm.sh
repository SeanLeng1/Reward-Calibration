#!/bin/bash
set +e

# reward model list and whether to trust remote code
declare -A MODELS
MODELS["openbmb/Eurus-RM-7b"]="true"            
MODELS["OpenLLMAI/Llama-3-8b-rm-mixture"]="true"



# in case tokenizer does not have a chat template
# which fastchat conversation template to use
declare -A CHAT_TEMPLATES
CHAT_TEMPLATES["openbmb/Eurus-RM-7b"]="mistral"
CHAT_TEMPLATES["OpenLLMAI/Llama-3-8b-rm-mixture"]="llama-3"


# whether or not loading as OpenRLHF reward models
# be careful with this one, reward model trained with OpenRLHF should assign True here
declare -A LOADINGS
LOADINGS["openbmb/Eurus-RM-7b"]="false"
LOADINGS["OpenLLMAI/Llama-3-8b-rm-mixture"]="true"


# try all modes and each with and without confidence score
MODES=("chosen" "rejected" "normal" "both_high" "both_low")
FORMAT_PROMPTS=("true" "false")
PROBABILITIES_SIDES=('right')

for PROBABILITIES_SIDE in "${PROBABILITIES_SIDES[@]}"
do
    PROB_SIDE_FLAG=""
    if [ "$PROBABILITIES_SIDE" = "left" ]; then
        PROB_SIDE_FLAG="_left"
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
            TRUST_REMOTE_FLAG=""
            if [ "${MODELS[$MODEL_NAME]}" = "true" ]; then
                TRUST_REMOTE_FLAG="--trust_remote_code"
            fi

            # corner case for starling
            TOKENIZER_PATH=$MODEL_NAME
            if [ "$MODEL_NAME" = "berkeley-nest/Starling-RM-7B-alpha" ]; then
                TOKENIZER_PATH="meta-llama/Llama-2-7b-chat-hf"
            fi

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
                    NAME=""
                    SCORE_NAME=""
                    if [ "$USE_PROBABILITY" = "true" ]; then
                        ADD_PROBABILITY_FLAG="--add_probabilities"
                        NAME="_probability"
                        SCORE_NAME="_with_probability"
                    fi
                    CUSTOMIZE_LOADING=""
                    if [ "${LOADINGS[$MODEL_NAME]}" = "true" ]; then
                        CUSTOMIZE_LOADING="--customize_loading"
                    fi

                    OUTPUT="reward_results/${SAVE_FOLDER}${PROB_SIDE_FLAG}/${CLEAN_MODEL_NAME}/"
                    mkdir -p $OUTPUT
                    SCORE_FILE="${OUTPUT}/scores${SCORE_NAME}_${MODE}.json"
                    # check if jsons exist in the output directory, so we can avoid rerunning
                    if [ -f "${SCORE_FILE}" ]; then
                        echo "Required file exists for ${OUTPUT}, mode ${MODE}, skipping..."
                        continue  # Skip to the next iteration of the loop
                    fi

                    CMD="python query/run_rm.py \
                        --model $MODEL_NAME \
                        --seed 42 \
                        --mode $MODE \
                        $ADD_PROBABILITY_FLAG \
                        $TRUST_REMOTE_FLAG \
                        --chat_template ${CHAT_TEMPLATES[$MODEL_NAME]} \
                        --batch_size 8 \
                        --tokenizer $TOKENIZER_PATH \
                        $FORMAT_PROMPT_FLAG \
                        $CUSTOMIZE_LOADING \
                        --probabilities_side $PROBABILITIES_SIDE \
                        --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training_${MODE}${NAME}.log"

                    echo "Running: $CMD"
                    eval $CMD || echo "An error occurred with $CMD but continuing with other tasks..."
                done
            done
        done
    done
done
