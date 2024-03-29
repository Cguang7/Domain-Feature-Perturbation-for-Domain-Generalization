dataset=PACS
command=$1
data_dir=$2
gpu_id=$3

CUDA_VISIBLE_DEVICES=${gpu_id} \
python3 -m domainbed.scripts.sweep ${command}\
       --datasets ${dataset}\
       --algorithms DFP\
       --data_dir ${data_dir}\
       --command_launcher local\
       --fixed_test_envs 0\
       --steps 7001 \
       --holdout_fraction 0.1\
       --n_hparams 8\
       --n_trials 3\
       --skip_confirmation\
       --hparams "$(<sweep/${dataset}/hparams.json)"\
       --output_dir "/output/${dataset}/outputs1"

#--single_test_envs\