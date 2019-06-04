cd scripts/tf_cnn_benchmarks/
OMP_NUM_THREADS=28 KMP_AFFINITY='explicit,proclist=[0-27]' python tf_cnn_benchmarks.py \
        --model=densenet40_k12 --num_inter_threads=1 --batch_size=1 --train_dir=densenet_train \
        --data_format NHWC --num_batches 1000 --data_name=cifar10