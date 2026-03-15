INPUT_PATH=/home/xxf/NewVLLM/vllm/test/bench_results
NAME1=fifo
NAME2=fifo2

python /home/xxf/NewVLLM/vllm/experiment/plot_dispatch_metrics.py ${INPUT_PATH}/log-${NAME1} -o ${INPUT_PATH}/${NAME1}_plot

python /home/xxf/NewVLLM/vllm/experiment/plot_dispatch_metrics.py ${INPUT_PATH}/log-${NAME2} -o ${INPUT_PATH}/${NAME2}_plot

python /home/xxf/NewVLLM/vllm/experiment/plot_bench_results.py ${INPUT_PATH}/random2_${NAME1} ${NAME1} ${INPUT_PATH}/random2_${NAME2} ${NAME2} -o ${INPUT_PATH}/plot_compare