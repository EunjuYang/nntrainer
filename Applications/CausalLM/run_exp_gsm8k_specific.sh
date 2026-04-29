#!/system/bin/sh  
# ------------------------------------------------------------  
# 1️⃣ 인자 정의  
# ------------------------------------------------------------  
MODEL_PATH=$1           # 모델 경로  
PROMPT_ID=$2            # 찾고 싶은 프롬프트 ID (예: 43)  
CSV_FILE="gsm8k_sample_input.csv"  # 따옴표 없이 만든 CSV (같은 디렉터리)  

# ------------------------------------------------------------  
# 2️⃣ 실행 설정  
# ------------------------------------------------------------  
NUM_RUNS=5  
WAIT_TIME=2  

# ------------------------------------------------------------  
# 3️⃣ 프롬프트 추출 함수  
#    - 헤더를 무시하고 ID가 정확히 일치하면 첫 번째 콤마 뒤 전체를 반환  
# ------------------------------------------------------------  
get_prompt() {  
    local id=$1  
    local line cur_id prompt  

    while IFS= read -r line || [ -n "$line" ]; do  
        # Windows 개행(\r) 제거  
        line=${line%$'\r'}  

        # 숫자로 시작하지 않으면 헤더·빈줄이므로 건너뛰기  
        case "$line" in  
            [0-9]*)  
                # 첫 콤마 앞까지가 ID  
                cur_id=${line%%,*}  
                if [ "$cur_id" = "$id" ]; then  
                    # 첫 콤마 뒤 전체가 프롬프트 (콤마 포함)  
                    prompt=${line#*,}  
                    printf '%s' "$prompt"  
                    return 0  
                fi  
                ;;  
        esac  
    done < "$CSV_FILE"  

    return 1   # 못 찾음  
}  

# ------------------------------------------------------------  
# 4️⃣ 메인 루프  
# ------------------------------------------------------------  
i=1  
while [ $i -le $NUM_RUNS ]; do  
    echo "Running test $i of $NUM_RUNS..."  

    PROMPT=$(get_prompt "$PROMPT_ID")  
    if [ $? -ne 0 ] || [ -z "$PROMPT" ]; then  
        echo "⚠️  ID [$PROMPT_ID] 에 해당하는 프롬프트를 찾을 수 없습니다."  
        exit 1  
    fi  

    # run_causallm.sh 호출 (프롬프트 전체를 하나의 인수로 전달)  
    ./run_causallm.sh "$MODEL_PATH" "$PROMPT"  
    wait  

    echo "Test $i completed."  
    [ $i -lt $NUM_RUNS ] && sleep $WAIT_TIME  
    i=$((i + 1))  
done  

echo "All tests completed successfully."  