#!/system/bin/sh  
# ============================================================  
# 1️⃣ 인자 정의  
# ============================================================  
MODEL_PATH=$1                     # 1번째 인자 : 모델 파일 경로  
CSV_FILE="gsm8k_sample_input.csv" # 같은 디렉터리 혹은 절대 경로  

# ============================================================  
# 2️⃣ 실행 옵션 (필요 시 환경 변수로 오버라이드)  
# ============================================================  
WAIT_TIME=2               # 프롬프트·프롬프트 사이 대기 시간(초)  
MAX_RUNS=0                # 0 = CSV 전체, 양수 = 앞 N개만 처리  
RUN_PER_PROMPT=3          # 같은 프롬프트를 몇 번 반복 실행할지  
WAIT_BETWEEN_REPEATS=2    # 같은 프롬프트 내부 대기(0 = 바로 다음)  

# ============================================================  
# 3️⃣ 메인 : CSV 를 라인‑단위로 읽고, 각 Prompt 를 여러 번 실행  
# ============================================================  
if [ ! -f "$CSV_FILE" ]; then  
    echo "❌ CSV 파일을 찾을 수 없습니다: $CSV_FILE"  
    exit 1  
fi  

total_processed=0  

# IFS=',' 로 첫 번째 콤마만 구분자 → 두 변수에만 할당  
while IFS=',' read -r id prompt || [ -n "$id" ]; do  
    # ---------- ① Windows 개행(\r) 제거 ----------  
    id=${id%$'\r'}  
    prompt=${prompt%$'\r'}  

    # ---------- ② 헤더·빈줄 스킵 ----------  
    case "$id" in  
        ""|[!0-9]*)  
            # 빈줄 혹은 첫 글자가 숫자가 아니면(헤더) 건너뛰기  
            continue  
            ;;  
    esac  

    # ---------- ③ 양쪽 따옴표가 있으면 제거 ----------  
    prompt=${prompt#\"}  
    prompt=${prompt%\"}  

    echo "--------------------------------------------------"  
    echo "Processing Prompt ID = $id"  

    # ---------- ④ 같은 Prompt 를 RUN_PER_PROMPT 번 반복 ----------  
    repeat_idx=1  
    while [ $repeat_idx -le $RUN_PER_PROMPT ]; do  
        echo "[${repeat_idx}/${RUN_PER_PROMPT}] run_causallm.sh …"  
        ./run_causallm.sh "$MODEL_PATH" "$prompt"  
        wait   # 백그라운드 프로세스가 있으면 여기서 대기  

        # 같은 Prompt 내부 대기 (필요 시)  
        if [ $WAIT_BETWEEN_REPEATS -gt 0 ] && [ $repeat_idx -lt $RUN_PER_PROMPT ]; then  
            echo "   (waiting $WAIT_BETWEEN_REPEATS s before next repeat…)"  
            sleep $WAIT_BETWEEN_REPEATS  
        fi  
        repeat_idx=$((repeat_idx + 1))  
    done  

    echo "Prompt ID $id processed $RUN_PER_PROMPT times."  
    total_processed=$((total_processed + 1))  

    # ---------- ⑤ Prompt 사이 대기 ----------  
    if [ $WAIT_TIME -gt 0 ]; then  
        sleep $WAIT_TIME  
    fi  

    # ---------- ⑥ 전체 실행 제한 (MAX_RUNS) ----------  
    if [ $MAX_RUNS -gt 0 ] && [ $total_processed -ge $MAX_RUNS ]; then  
        echo "Maximum processed prompts ($MAX_RUNS) reached – stop."  
        break  
    fi  
done < "$CSV_FILE"  

echo "=== DONE ==="  
echo "Total distinct prompts processed : $total_processed"  
echo "Each prompt was run                : $RUN_PER_PROMPT times"  