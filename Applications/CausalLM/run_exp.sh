#!/system/bin/sh  

# 실행할 모델 경로  
MODEL_PATH=$1 

# 반복 실행 횟수 (원하는 횟수로 변경)  
NUM_RUNS=5  

# 각 실행 사이의 대기 시간 (초)  
WAIT_TIME=2  

i=1  
while [ $i -le $NUM_RUNS ]  
do  
    echo "Running test $i of $NUM_RUNS..."  
      
    # 스크립트 실행  
    ./run_causallm.sh "$MODEL_PATH"  
      
    # 프로세스가 완료될 때까지 대기  
    wait  
      
    echo "Test $i completed."  
      
    # 마지막 실행이 아니면 대기  
    if [ $i -lt $NUM_RUNS ]; then  
        echo "Waiting for $WAIT_TIME seconds before next run..."  
        sleep $WAIT_TIME  
    fi  
      
    i=$((i + 1))  
done  

echo "All tests completed successfully."  
