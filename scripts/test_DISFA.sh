exp_name="DISFA_stacking_head_results_fold"$1
resume_path="data/weights/stacking_head/DISFA/stacking_head_DISFA_mdl_params_fold"$1".pt"

python test.py --dataset DISFA \
                 --arc swin_transformer_base \
                 --exp-name $exp_name \
                 --resume $resume_path \
                 --fold $1 \
                 -b 32 