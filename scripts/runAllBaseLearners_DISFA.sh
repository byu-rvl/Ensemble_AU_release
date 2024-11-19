#!/bin/bash

selected_model_number=$1

if [ "$selected_model_number" == 0 ]; then 
    echo "DISFA_lr0.0001_weight10_AU1_SWIN_TRANSFORMER_3Layers_allLam01_makeHuman_fold1_16-09-2024_05_20-15_PM__epoch2.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight10_AU1_SWIN_TRANSFORMER_3Layers_allLam01_makeHuman_fold1_16-09-2024_05_20-15_PM__epoch2.pth"
elif [ "$selected_model_number" == 1 ]; then 
    echo "DISFA_lr0.0001_weight10_AU2_SWIN_TRANSFORMER_3Layers_pretrained2_fold1_09-11-2024_11_47-51_PM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight10_AU2_SWIN_TRANSFORMER_3Layers_pretrained2_fold1_09-11-2024_11_47-51_PM__epoch1.pth"
elif [ "$selected_model_number" == 2 ]; then 
    echo "DISFA_lr0.0001_weight10_AU2_SWIN_TRANSFORMER_3Layers_pretrained2_fold3_10-11-2024_05_45-44_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight10_AU2_SWIN_TRANSFORMER_3Layers_pretrained2_fold3_10-11-2024_05_45-44_AM__epoch1.pth"
elif [ "$selected_model_number" == 3 ]; then 
    echo "DISFA_lr0.0001_weight10_AU4_SWIN_TRANSFORMER_3Layers_allLam01_pretrained2_makeHuman_fold1_11-11-2024_06_09-09_AM__epoch5.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight10_AU4_SWIN_TRANSFORMER_3Layers_allLam01_pretrained2_makeHuman_fold1_11-11-2024_06_09-09_AM__epoch5.pth"
elif [ "$selected_model_number" == 4 ]; then 
    echo "DISFA_lr0.0001_weight10_AU6_Resnet_3Layers_fold1_14-09-2024_06_20-13_AM__epoch3.pth"
    ./scripts/runBaseLearner.sh resnet50 3 DISFA "DISFA_lr0.0001_weight10_AU6_Resnet_3Layers_fold1_14-09-2024_06_20-13_AM__epoch3.pth"
elif [ "$selected_model_number" == 5 ]; then 
    echo "DISFA_lr0.0001_weight11_AU1_SWIN_TRANSFORMER_3Layers_allLam01_makeHuman_fold3_17-09-2024_05_00-48_AM__epoch2.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight11_AU1_SWIN_TRANSFORMER_3Layers_allLam01_makeHuman_fold3_17-09-2024_05_00-48_AM__epoch2.pth"
elif [ "$selected_model_number" == 6 ]; then 
    echo "DISFA_lr0.0001_weight11_AU5_Resnet_1Layers_fold3_15-09-2024_06_08-45_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh resnet50 1 DISFA "DISFA_lr0.0001_weight11_AU5_Resnet_1Layers_fold3_15-09-2024_06_08-45_AM__epoch1.pth"
elif [ "$selected_model_number" == 7 ]; then 
    echo "DISFA_lr0.0001_weight11_AU6_Resnet_3Layers_fold1_14-09-2024_06_19-58_AM__epoch4.pth"
    ./scripts/runBaseLearner.sh resnet50 3 DISFA "DISFA_lr0.0001_weight11_AU6_Resnet_3Layers_fold1_14-09-2024_06_19-58_AM__epoch4.pth"
elif [ "$selected_model_number" == 8 ]; then 
    echo "DISFA_lr0.0001_weight11_AU6_Resnet_3Layers_fold3_14-09-2024_11_06-37_AM__epoch5.pth"
    ./scripts/runBaseLearner.sh resnet50 3 DISFA "DISFA_lr0.0001_weight11_AU6_Resnet_3Layers_fold3_14-09-2024_11_06-37_AM__epoch5.pth"
elif [ "$selected_model_number" == 9 ]; then 
    echo "DISFA_lr0.0001_weight11_AU7_SWIN_TRANSFORMER_3Layers_allLam01_pretrained2_fold2_10-11-2024_08_34-35_PM__epoch3.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight11_AU7_SWIN_TRANSFORMER_3Layers_allLam01_pretrained2_fold2_10-11-2024_08_34-35_PM__epoch3.pth"
elif [ "$selected_model_number" == 10 ]; then 
    echo "DISFA_lr0.0001_weight11_AU7_SWIN_TRANSFORMER_3Layers_allLam01_pretrained2_fold3_10-11-2024_11_34-48_PM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight11_AU7_SWIN_TRANSFORMER_3Layers_allLam01_pretrained2_fold3_10-11-2024_11_34-48_PM__epoch1.pth"
elif [ "$selected_model_number" == 11 ]; then 
    echo "DISFA_lr0.0001_weight12_AU5_Resnet_1Layers_fold2_15-09-2024_03_54-54_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh resnet50 1 DISFA "DISFA_lr0.0001_weight12_AU5_Resnet_1Layers_fold2_15-09-2024_03_54-54_AM__epoch1.pth"
elif [ "$selected_model_number" == 12 ]; then 
    echo "DISFA_lr0.0001_weight12_AU6_Resnet_3Layers_fold3_14-09-2024_11_07-22_AM__epoch3.pth"
    ./scripts/runBaseLearner.sh resnet50 3 DISFA "DISFA_lr0.0001_weight12_AU6_Resnet_3Layers_fold3_14-09-2024_11_07-22_AM__epoch3.pth"
elif [ "$selected_model_number" == 13 ]; then 
    echo "DISFA_lr0.0001_weight12_AU7_SWIN_TRANSFORMER_3Layers_allLam01_pretrained2_fold1_10-11-2024_05_34-02_PM__epoch3.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight12_AU7_SWIN_TRANSFORMER_3Layers_allLam01_pretrained2_fold1_10-11-2024_05_34-02_PM__epoch3.pth"
elif [ "$selected_model_number" == 14 ]; then 
    echo "DISFA_lr0.0001_weight12_AU7_SWIN_TRANSFORMER_3Layers_allLam01_pretrained2_fold2_10-11-2024_08_35-32_PM__epoch4.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight12_AU7_SWIN_TRANSFORMER_3Layers_allLam01_pretrained2_fold2_10-11-2024_08_35-32_PM__epoch4.pth"
elif [ "$selected_model_number" == 15 ]; then 
    echo "DISFA_lr0.0001_weight12_AU7_SWIN_TRANSFORMER_3Layers_allLam01_pretrained2_fold3_10-11-2024_11_36-47_PM__epoch3.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight12_AU7_SWIN_TRANSFORMER_3Layers_allLam01_pretrained2_fold3_10-11-2024_11_36-47_PM__epoch3.pth"
elif [ "$selected_model_number" == 16 ]; then 
    echo "DISFA_lr0.0001_weight2_AU1_SWIN_TRANSFORMER_3Layers_allLam01_makeHuman_fold1_16-09-2024_05_20-15_PM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight2_AU1_SWIN_TRANSFORMER_3Layers_allLam01_makeHuman_fold1_16-09-2024_05_20-15_PM__epoch1.pth"
elif [ "$selected_model_number" == 17 ]; then 
    echo "DISFA_lr0.0001_weight2_AU1_SWIN_TRANSFORMER_3Layers_allLam01_makeHuman_fold2_16-09-2024_11_16-56_PM__epoch5.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight2_AU1_SWIN_TRANSFORMER_3Layers_allLam01_makeHuman_fold2_16-09-2024_11_16-56_PM__epoch5.pth"
elif [ "$selected_model_number" == 18 ]; then 
    echo "DISFA_lr0.0001_weight2_AU3_Resnet_3Layers_allLam01_pretrained2_fold2_10-11-2024_06_53-36_PM__epoch5.pth"
    ./scripts/runBaseLearner.sh resnet50 3 DISFA "DISFA_lr0.0001_weight2_AU3_Resnet_3Layers_allLam01_pretrained2_fold2_10-11-2024_06_53-36_PM__epoch5.pth"
elif [ "$selected_model_number" == 19 ]; then 
    echo "DISFA_lr0.0001_weight3_AU0_SWIN_TRANSFORMER_3Layers_allLam01_fold1_11-09-2024_09_42-44_PM__epoch2.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight3_AU0_SWIN_TRANSFORMER_3Layers_allLam01_fold1_11-09-2024_09_42-44_PM__epoch2.pth"
elif [ "$selected_model_number" == 20 ]; then 
    echo "DISFA_lr0.0001_weight3_AU0_SWIN_TRANSFORMER_3Layers_allLam01_fold3_12-09-2024_03_41-08_AM__epoch2.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight3_AU0_SWIN_TRANSFORMER_3Layers_allLam01_fold3_12-09-2024_03_41-08_AM__epoch2.pth"
elif [ "$selected_model_number" == 21 ]; then 
    echo "DISFA_lr0.0001_weight3_AU1_SWIN_TRANSFORMER_3Layers_allLam01_makeHuman_fold2_16-09-2024_11_16-56_PM__epoch4.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight3_AU1_SWIN_TRANSFORMER_3Layers_allLam01_makeHuman_fold2_16-09-2024_11_16-56_PM__epoch4.pth"
elif [ "$selected_model_number" == 22 ]; then 
    echo "DISFA_lr0.0001_weight3_AU2_SWIN_TRANSFORMER_3Layers_pretrained2_fold2_10-11-2024_03_01-28_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight3_AU2_SWIN_TRANSFORMER_3Layers_pretrained2_fold2_10-11-2024_03_01-28_AM__epoch1.pth"
elif [ "$selected_model_number" == 23 ]; then 
    echo "DISFA_lr0.0001_weight3_AU3_Resnet_3Layers_allLam01_pretrained2_fold1_10-11-2024_02_05-02_PM__epoch2.pth"
    ./scripts/runBaseLearner.sh resnet50 3 DISFA "DISFA_lr0.0001_weight3_AU3_Resnet_3Layers_allLam01_pretrained2_fold1_10-11-2024_02_05-02_PM__epoch2.pth"
elif [ "$selected_model_number" == 24 ]; then 
    echo "DISFA_lr0.0001_weight3_AU3_Resnet_3Layers_allLam01_pretrained2_fold1_10-11-2024_02_05-02_PM__epoch3.pth"
    ./scripts/runBaseLearner.sh resnet50 3 DISFA "DISFA_lr0.0001_weight3_AU3_Resnet_3Layers_allLam01_pretrained2_fold1_10-11-2024_02_05-02_PM__epoch3.pth"
elif [ "$selected_model_number" == 25 ]; then 
    echo "DISFA_lr0.0001_weight3_AU3_Resnet_3Layers_allLam01_pretrained2_fold2_10-11-2024_06_52-18_PM__epoch5.pth"
    ./scripts/runBaseLearner.sh resnet50 3 DISFA "DISFA_lr0.0001_weight3_AU3_Resnet_3Layers_allLam01_pretrained2_fold2_10-11-2024_06_52-18_PM__epoch5.pth"
elif [ "$selected_model_number" == 26 ]; then 
    echo "DISFA_lr0.0001_weight3_AU3_Resnet_3Layers_allLam01_pretrained2_fold3_10-11-2024_10_26-44_PM__epoch1.pth"
    ./scripts/runBaseLearner.sh resnet50 3 DISFA "DISFA_lr0.0001_weight3_AU3_Resnet_3Layers_allLam01_pretrained2_fold3_10-11-2024_10_26-44_PM__epoch1.pth"
elif [ "$selected_model_number" == 27 ]; then 
    echo "DISFA_lr0.0001_weight4_AU1_SWIN_TRANSFORMER_3Layers_allLam01_makeHuman_fold3_17-09-2024_05_17-43_AM__epoch3.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight4_AU1_SWIN_TRANSFORMER_3Layers_allLam01_makeHuman_fold3_17-09-2024_05_17-43_AM__epoch3.pth"
elif [ "$selected_model_number" == 28 ]; then 
    echo "DISFA_lr0.0001_weight4_AU3_Resnet_3Layers_allLam01_pretrained2_fold3_10-11-2024_10_26-44_PM__epoch5.pth"
    ./scripts/runBaseLearner.sh resnet50 3 DISFA "DISFA_lr0.0001_weight4_AU3_Resnet_3Layers_allLam01_pretrained2_fold3_10-11-2024_10_26-44_PM__epoch5.pth"
elif [ "$selected_model_number" == 29 ]; then 
    echo "DISFA_lr0.0001_weight5_AU0_SWIN_TRANSFORMER_3Layers_allLam01_fold2_12-09-2024_12_37-29_AM__epoch2.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight5_AU0_SWIN_TRANSFORMER_3Layers_allLam01_fold2_12-09-2024_12_37-29_AM__epoch2.pth"
elif [ "$selected_model_number" == 30 ]; then 
    echo "DISFA_lr0.0001_weight5_AU2_SWIN_TRANSFORMER_3Layers_pretrained2_fold2_10-11-2024_02_56-14_AM__epoch5.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight5_AU2_SWIN_TRANSFORMER_3Layers_pretrained2_fold2_10-11-2024_02_56-14_AM__epoch5.pth"
elif [ "$selected_model_number" == 31 ]; then 
    echo "DISFA_lr0.0001_weight5_AU4_SWIN_TRANSFORMER_3Layers_allLam01_pretrained2_makeHuman_fold3_11-11-2024_11_53-11_AM__epoch3.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight5_AU4_SWIN_TRANSFORMER_3Layers_allLam01_pretrained2_makeHuman_fold3_11-11-2024_11_53-11_AM__epoch3.pth"
elif [ "$selected_model_number" == 32 ]; then 
    echo "DISFA_lr0.0001_weight5_AU5_Resnet_1Layers_fold1_15-09-2024_01_54-57_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh resnet50 1 DISFA "DISFA_lr0.0001_weight5_AU5_Resnet_1Layers_fold1_15-09-2024_01_54-57_AM__epoch1.pth"
elif [ "$selected_model_number" == 33 ]; then 
    echo "DISFA_lr0.0001_weight5_AU5_Resnet_1Layers_fold3_15-09-2024_06_35-54_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh resnet50 1 DISFA "DISFA_lr0.0001_weight5_AU5_Resnet_1Layers_fold3_15-09-2024_06_35-54_AM__epoch1.pth"
elif [ "$selected_model_number" == 34 ]; then 
    echo "DISFA_lr0.0001_weight5_AU6_Resnet_3Layers_fold2_14-09-2024_09_43-07_AM__epoch3.pth"
    ./scripts/runBaseLearner.sh resnet50 3 DISFA "DISFA_lr0.0001_weight5_AU6_Resnet_3Layers_fold2_14-09-2024_09_43-07_AM__epoch3.pth"
elif [ "$selected_model_number" == 35 ]; then 
    echo "DISFA_lr0.0001_weight5_AU6_Resnet_3Layers_fold2_14-09-2024_09_43-07_AM__epoch5.pth"
    ./scripts/runBaseLearner.sh resnet50 3 DISFA "DISFA_lr0.0001_weight5_AU6_Resnet_3Layers_fold2_14-09-2024_09_43-07_AM__epoch5.pth"
elif [ "$selected_model_number" == 36 ]; then 
    echo "DISFA_lr0.0001_weight6_AU0_SWIN_TRANSFORMER_3Layers_allLam01_fold3_12-09-2024_03_42-08_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight6_AU0_SWIN_TRANSFORMER_3Layers_allLam01_fold3_12-09-2024_03_42-08_AM__epoch1.pth"
elif [ "$selected_model_number" == 37 ]; then 
    echo "DISFA_lr0.0001_weight7_AU4_SWIN_TRANSFORMER_3Layers_allLam01_pretrained2_makeHuman_fold3_13-11-2024_05_33-42_AM__epoch3.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight7_AU4_SWIN_TRANSFORMER_3Layers_allLam01_pretrained2_makeHuman_fold3_13-11-2024_05_33-42_AM__epoch3.pth"
elif [ "$selected_model_number" == 38 ]; then 
    echo "DISFA_lr0.0001_weight8_AU0_SWIN_TRANSFORMER_3Layers_allLam01_fold2_12-09-2024_12_37-22_AM__epoch2.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight8_AU0_SWIN_TRANSFORMER_3Layers_allLam01_fold2_12-09-2024_12_37-22_AM__epoch2.pth"
elif [ "$selected_model_number" == 39 ]; then 
    echo "DISFA_lr0.0001_weight8_AU2_SWIN_TRANSFORMER_3Layers_pretrained2_fold3_10-11-2024_05_54-13_AM__epoch5.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight8_AU2_SWIN_TRANSFORMER_3Layers_pretrained2_fold3_10-11-2024_05_54-13_AM__epoch5.pth"
elif [ "$selected_model_number" == 40 ]; then 
    echo "DISFA_lr0.0001_weight8_AU4_SWIN_TRANSFORMER_3Layers_allLam01_pretrained2_makeHuman_fold1_11-11-2024_06_23-59_AM__epoch4.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight8_AU4_SWIN_TRANSFORMER_3Layers_allLam01_pretrained2_makeHuman_fold1_11-11-2024_06_23-59_AM__epoch4.pth"
elif [ "$selected_model_number" == 41 ]; then 
    echo "DISFA_lr0.0001_weight8_AU4_SWIN_TRANSFORMER_3Layers_allLam01_pretrained2_makeHuman_fold2_13-11-2024_05_30-14_AM__epoch3.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight8_AU4_SWIN_TRANSFORMER_3Layers_allLam01_pretrained2_makeHuman_fold2_13-11-2024_05_30-14_AM__epoch3.pth"
elif [ "$selected_model_number" == 42 ]; then 
    echo "DISFA_lr0.0001_weight8_AU5_Resnet_1Layers_fold1_15-09-2024_01_44-54_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh resnet50 1 DISFA "DISFA_lr0.0001_weight8_AU5_Resnet_1Layers_fold1_15-09-2024_01_44-54_AM__epoch1.pth"
elif [ "$selected_model_number" == 43 ]; then 
    echo "DISFA_lr0.0001_weight8_AU5_Resnet_1Layers_fold2_15-09-2024_04_12-18_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh resnet50 1 DISFA "DISFA_lr0.0001_weight8_AU5_Resnet_1Layers_fold2_15-09-2024_04_12-18_AM__epoch1.pth"
elif [ "$selected_model_number" == 44 ]; then 
    echo "DISFA_lr0.0001_weight9_AU0_SWIN_TRANSFORMER_3Layers_allLam01_fold1_11-09-2024_09_37-21_PM__epoch2.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight9_AU0_SWIN_TRANSFORMER_3Layers_allLam01_fold1_11-09-2024_09_37-21_PM__epoch2.pth"
elif [ "$selected_model_number" == 45 ]; then 
    echo "DISFA_lr0.0001_weight9_AU2_SWIN_TRANSFORMER_3Layers_pretrained2_fold1_09-11-2024_11_51-38_PM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight9_AU2_SWIN_TRANSFORMER_3Layers_pretrained2_fold1_09-11-2024_11_51-38_PM__epoch1.pth"
elif [ "$selected_model_number" == 46 ]; then 
    echo "DISFA_lr0.0001_weight9_AU4_SWIN_TRANSFORMER_3Layers_allLam01_pretrained2_makeHuman_fold2_11-11-2024_06_26-05_AM__epoch5.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 DISFA "DISFA_lr0.0001_weight9_AU4_SWIN_TRANSFORMER_3Layers_allLam01_pretrained2_makeHuman_fold2_11-11-2024_06_26-05_AM__epoch5.pth"
    # end of the list of models
else
    echo "Invalid model number"
fi