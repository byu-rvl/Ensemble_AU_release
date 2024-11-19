exp_name="BP4D_stacking_head_results_fold"$1
resume_path="data/weights/stacking_head/BP4D/stacking_head_BP4D_mdl_params_fold"$1".pt"

python test.py --dataset BP4D \
                 --arc swin_transformer_base \
                 --exp-name $exp_name \
                 --resume $resume_path \
                 --fold $1 \
                 -b 32 