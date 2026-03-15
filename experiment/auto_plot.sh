INPUT_PATH=/home/xxf/NewVLLM/vllm/test/bench_results

python /home/xxf/NewVLLM/vllm/experiment/plot_dispatch_metrics.py ${INPUT_PATH}/log-fifo -o ${INPUT_PATH}/fifo_plot

python /home/xxf/NewVLLM/vllm/experiment/plot_dispatch_metrics.py ${INPUT_PATH}/log-fifo2 -o ${INPUT_PATH}/fifo2_plot

python /home/xxf/NewVLLM/vllm/experiment/plot_bench_results.py ${INPUT_PATH}/random2_fifo fifo ${INPUT_PATH}/random2_fifo2 fifo2 -o ${INPUT_PATH}/plot_compare