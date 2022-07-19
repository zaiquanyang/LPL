# M=40
# for tmp in 1.0  0.1  0.2 0.5   5.0  10.0  #  (temp = 10, 5, 2, 1, 0.2, 0.1)
# do
#   for walk_num in `expr $M / 20` `expr $M / 10`  `expr $M / 5`  `expr $M / 3` `expr $M / 2`  `expr $M / 1`
#   do
#   python /data/code/ZSL/C2F_ZSL_CVPR/trainers/train_cub.py --opts temp $tmp \
#                                                           gama 1.0 \
#                                                           walk_num $walk_num \
#                                                           sub_ways $M \
#                                                           sub_shots 4 \
#                                                           key_word "Ablation" \
#                                                           train_stage 'Fine'
#   done
# done
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" python /data/code/LPL/trainers/train_cub.py --opts temp 1.0 gama 1.0 walk_num 6 sub_ways 20 sub_shots 4 transductive True key_word "Transductive_Mix_Warm5" train_stage 'Fine' warm_epoch 5