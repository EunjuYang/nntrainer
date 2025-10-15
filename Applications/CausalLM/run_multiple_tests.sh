#!/system/bin/sh  

# path to test model (this should be updated to be taken by an argument)
MODEL_PATH="models/qwen3-30b-a3b-slim-cached/"  

# number of trials (this should be updated to be taken by an argument)
NUM_RUNS=5  

# intermediate pause time (sec)
WAIT_TIME=2  

i=1  
while [ $i -le $NUM_RUNS ]  
do  
    echo "Running test $i of $NUM_RUNS..."  
      
    # run the script
    ./run_causallm.sh "$MODEL_PATH"  
      
    # wait until the process finishes
    wait  
      
    echo "Test $i completed."  
      
    # if it is not the final ends, pause
    if [ $i -lt $NUM_RUNS ]; then  
        echo "Waiting for $WAIT_TIME seconds before next run..."  
        sleep $WAIT_TIME  
    fi  
      
    i=$((i + 1))  
done  

echo "All tests completed successfully."  