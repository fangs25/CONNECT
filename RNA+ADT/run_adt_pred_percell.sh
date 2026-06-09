scvipath='/home/zhangjingxiao/.conda/envs/scvi-env2/bin/python'


# Datasets=('10xNeuron' '10xPBMC_raw' 'SHAREseq' 'SNAREseq_human' 'SNAREseq_marmoset' 
#           'SNAREseq_mouse' 'Retina' 'parallelseq' 'issaacseq')
Datasets=('SHAREseq' 'SNAREseq_human' 'Retina' )
Datasets=('RPE008' 'RPE010' 'RPE012' 'RPE014' 'RPE015' 'RPE016')
# Datasets=('RPE016')

# 输出路径
LOG_DIR="./saved_logs"
mkdir -p $LOG_DIR

# seeds=$(seq 0 9)
for dataset in ${Datasets[@]};
do

    echo "Running: $dataset"

    # nohup ${scvipath} atac_mapping.py --dataset ${dataset} > "${LOG_DIR}/${dataset}_mapping.txt"  2>&1
    # nohup ${scvipath} rna_prediction_f_distance.py --dataset ${dataset} > "${LOG_DIR}/${dataset}_f.txt"  2>&1
    # nohup ${scvipath} rna_prediction_per_cell.py --dataset ${dataset} > "${LOG_DIR}/${dataset}_per_cell.txt"  2>&1
    nohup ${scvipath} adt_prediction_per_cell.py --dataset ${dataset} > "${LOG_DIR}/${dataset}_adt_per_cell.txt"  2>&1
    # nohup ${scvipath} adt_prediction_per_cell.py --dataset ${dataset} > "${LOG_DIR}/${dataset}_adt_per_cell.txt"  2>&1

    # nohup ${scvipath} adt_prediction_cell_pro.py --dataset ${dataset} > "${LOG_DIR}/${dataset}_cell_pro.txt"  2>&1
    # nohup ${scvipath} rna_prediction_cell_gene.py --dataset ${dataset} > "${LOG_DIR}/${dataset}_cell_gene.txt"  2>&1

    # script_pid=$!
    # echo "Started rna_prediction.py with PID $script_pid"
    
    # # 运行第二个Python程序，并将第一个程序的PID作为参数
    # if [[ -n "$script_pid" && "$script_pid" =~ ^[0-9]+$ ]]; then
    #     # python cuda_memory.py --pid "$script_pid" &
    #     ${scvipath} monitor_memory.py --pid "$script_pid" &

    #     monitor_pid=$!
    #     echo "Started monitor_memory.py with PID $monitor_pid and parameter $script_pid"

    #     # 等待进程结束
    #     wait "$script_pid"
    # else
    #     echo "Error: script_pid is invalid!"
    # fi

    echo $dataset Done
done
