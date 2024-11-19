#!/bin/bash

selected_model_number=$1

if [ "$selected_model_number" == 0 ]; then 
    echo "BP4D_lr0.0001_weight10_AU0_RESNET50_fold3_15-05-2024_09_41-16_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh resnet50 1 BP4D "BP4D_lr0.0001_weight10_AU0_RESNET50_fold3_15-05-2024_09_41-16_AM__epoch1.pth"
elif [ "$selected_model_number" == 1 ]; then 
    echo "BP4D_lr0.0001_weight10_AU10_RESNET50_fold2_15-05-2024_08_33-11_PM__epoch3.pth"
    ./scripts/runBaseLearner.sh resnet50 1 BP4D "BP4D_lr0.0001_weight10_AU10_RESNET50_fold2_15-05-2024_08_33-11_PM__epoch3.pth"
elif [ "$selected_model_number" == 2 ]; then 
    echo "BP4D_lr0.0001_weight10_AU1_SWIN_TRANSFORMER_3Layers_pretrained_fold3_09-06-2024_12_16-11_PM__epoch2.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight10_AU1_SWIN_TRANSFORMER_3Layers_pretrained_fold3_09-06-2024_12_16-11_PM__epoch2.pth"
elif [ "$selected_model_number" == 3 ]; then 
    echo "BP4D_lr0.0001_weight10_AU2_RESNET50_3Layers_fold1_30-05-2024_03_10-39_PM__epoch3.pth"
    ./scripts/runBaseLearner.sh resnet50 3 BP4D "BP4D_lr0.0001_weight10_AU2_RESNET50_3Layers_fold1_30-05-2024_03_10-39_PM__epoch3.pth"
elif [ "$selected_model_number" == 4 ]; then 
    echo "BP4D_lr0.0001_weight10_AU5_SWIN_TRANSFORMER_3Layers_fold2_20-05-2024_09_27-17_PM__epoch2.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight10_AU5_SWIN_TRANSFORMER_3Layers_fold2_20-05-2024_09_27-17_PM__epoch2.pth"
elif [ "$selected_model_number" == 5 ]; then 
    echo "BP4D_lr0.0001_weight10_AU7_SWIN_TRANSFORMER_3Layers_pretrained_fold2_09-06-2024_06_57-17_PM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight10_AU7_SWIN_TRANSFORMER_3Layers_pretrained_fold2_09-06-2024_06_57-17_PM__epoch1.pth"
elif [ "$selected_model_number" == 6 ]; then 
    echo "BP4D_lr0.0001_weight10_AU7_SWIN_TRANSFORMER_3Layers_pretrained_fold3_11-06-2024_08_58-35_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight10_AU7_SWIN_TRANSFORMER_3Layers_pretrained_fold3_11-06-2024_08_58-35_AM__epoch1.pth"
elif [ "$selected_model_number" == 7 ]; then 
    echo "BP4D_lr0.0001_weight10_AU8_SWIN_TRANSFORMER_3Layers_pretrained_fold2_09-06-2024_07_34-16_PM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight10_AU8_SWIN_TRANSFORMER_3Layers_pretrained_fold2_09-06-2024_07_34-16_PM__epoch1.pth"
elif [ "$selected_model_number" == 8 ]; then 
    echo "BP4D_lr0.0001_weight10_AU8_SWIN_TRANSFORMER_3Layers_pretrained_fold2_09-06-2024_07_34-16_PM__epoch2.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight10_AU8_SWIN_TRANSFORMER_3Layers_pretrained_fold2_09-06-2024_07_34-16_PM__epoch2.pth"
elif [ "$selected_model_number" == 9 ]; then 
    echo "BP4D_lr0.0001_weight10_AU8_SWIN_TRANSFORMER_3Layers_pretrained_fold3_11-06-2024_09_08-46_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight10_AU8_SWIN_TRANSFORMER_3Layers_pretrained_fold3_11-06-2024_09_08-46_AM__epoch1.pth"
elif [ "$selected_model_number" == 10 ]; then 
    echo "BP4D_lr0.0001_weight11_AU10_RESNET50_fold1_15-05-2024_08_28-15_PM__epoch2.pth"
    ./scripts/runBaseLearner.sh resnet50 1 BP4D "BP4D_lr0.0001_weight11_AU10_RESNET50_fold1_15-05-2024_08_28-15_PM__epoch2.pth"
elif [ "$selected_model_number" == 11 ]; then 
    echo "BP4D_lr0.0001_weight11_AU10_RESNET50_fold2_15-05-2024_08_29-33_PM__epoch2.pth"
    ./scripts/runBaseLearner.sh resnet50 1 BP4D "BP4D_lr0.0001_weight11_AU10_RESNET50_fold2_15-05-2024_08_29-33_PM__epoch2.pth"
elif [ "$selected_model_number" == 12 ]; then 
    echo "BP4D_lr0.0001_weight11_AU10_RESNET50_fold3_15-05-2024_08_30-56_PM__epoch1.pth"
    ./scripts/runBaseLearner.sh resnet50 1 BP4D "BP4D_lr0.0001_weight11_AU10_RESNET50_fold3_15-05-2024_08_30-56_PM__epoch1.pth"
elif [ "$selected_model_number" == 13 ]; then 
    echo "BP4D_lr0.0001_weight11_AU10_RESNET50_fold3_15-05-2024_08_30-56_PM__epoch2.pth"
    ./scripts/runBaseLearner.sh resnet50 1 BP4D "BP4D_lr0.0001_weight11_AU10_RESNET50_fold3_15-05-2024_08_30-56_PM__epoch2.pth"
elif [ "$selected_model_number" == 14 ]; then 
    echo "BP4D_lr0.0001_weight11_AU1_SWIN_TRANSFORMER_3Layers_pretrained_fold2_09-06-2024_08_29-43_AM__epoch3.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight11_AU1_SWIN_TRANSFORMER_3Layers_pretrained_fold2_09-06-2024_08_29-43_AM__epoch3.pth"
elif [ "$selected_model_number" == 15 ]; then 
    echo "BP4D_lr0.0001_weight11_AU1_SWIN_TRANSFORMER_3Layers_pretrained_fold3_09-06-2024_12_08-22_PM__epoch2.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight11_AU1_SWIN_TRANSFORMER_3Layers_pretrained_fold3_09-06-2024_12_08-22_PM__epoch2.pth"
elif [ "$selected_model_number" == 16 ]; then 
    echo "BP4D_lr0.0001_weight11_AU2_RESNET50_3Layers_fold2_02-06-2024_09_06-11_AM__epoch2.pth"
    ./scripts/runBaseLearner.sh resnet50 3 BP4D "BP4D_lr0.0001_weight11_AU2_RESNET50_3Layers_fold2_02-06-2024_09_06-11_AM__epoch2.pth"
elif [ "$selected_model_number" == 17 ]; then 
    echo "BP4D_lr0.0001_weight11_AU3_SWIN_TRANSFORMER_3Layers_withMakeHuman_fold2_06-06-2024_09_50-37_PM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight11_AU3_SWIN_TRANSFORMER_3Layers_withMakeHuman_fold2_06-06-2024_09_50-37_PM__epoch1.pth"
elif [ "$selected_model_number" == 18 ]; then 
    echo "BP4D_lr0.0001_weight11_AU4_SWIN_TRANSFORMER_1Layers_pretrained_allLam01_fold2_11-06-2024_07_56-01_PM__epoch2.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 1 BP4D "BP4D_lr0.0001_weight11_AU4_SWIN_TRANSFORMER_1Layers_pretrained_allLam01_fold2_11-06-2024_07_56-01_PM__epoch2.pth"
elif [ "$selected_model_number" == 19 ]; then 
    echo "BP4D_lr0.0001_weight11_AU7_SWIN_TRANSFORMER_3Layers_pretrained_fold1_09-06-2024_03_22-08_PM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight11_AU7_SWIN_TRANSFORMER_3Layers_pretrained_fold1_09-06-2024_03_22-08_PM__epoch1.pth"
elif [ "$selected_model_number" == 20 ]; then 
    echo "BP4D_lr0.0001_weight11_AU8_SWIN_TRANSFORMER_3Layers_pretrained_fold1_09-06-2024_03_56-25_PM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight11_AU8_SWIN_TRANSFORMER_3Layers_pretrained_fold1_09-06-2024_03_56-25_PM__epoch1.pth"
elif [ "$selected_model_number" == 21 ]; then 
    echo "BP4D_lr0.0001_weight11_AU8_SWIN_TRANSFORMER_3Layers_pretrained_fold1_09-06-2024_03_56-25_PM__epoch5.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight11_AU8_SWIN_TRANSFORMER_3Layers_pretrained_fold1_09-06-2024_03_56-25_PM__epoch5.pth"
elif [ "$selected_model_number" == 22 ]; then 
    echo "BP4D_lr0.0001_weight11_AU8_SWIN_TRANSFORMER_3Layers_pretrained_fold3_11-06-2024_09_08-43_AM__epoch3.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight11_AU8_SWIN_TRANSFORMER_3Layers_pretrained_fold3_11-06-2024_09_08-43_AM__epoch3.pth"
elif [ "$selected_model_number" == 23 ]; then 
    echo "BP4D_lr0.0001_weight12_AU0_RESNET50_fold2_15-05-2024_09_41-16_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh resnet50 1 BP4D "BP4D_lr0.0001_weight12_AU0_RESNET50_fold2_15-05-2024_09_41-16_AM__epoch1.pth"
elif [ "$selected_model_number" == 24 ]; then 
    echo "BP4D_lr0.0001_weight12_AU10_RESNET50_fold1_15-05-2024_08_27-19_PM__epoch1.pth"
    ./scripts/runBaseLearner.sh resnet50 1 BP4D "BP4D_lr0.0001_weight12_AU10_RESNET50_fold1_15-05-2024_08_27-19_PM__epoch1.pth"
elif [ "$selected_model_number" == 25 ]; then 
    echo "BP4D_lr0.0001_weight12_AU1_SWIN_TRANSFORMER_3Layers_pretrained_fold1_09-06-2024_04_56-32_AM__epoch4.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight12_AU1_SWIN_TRANSFORMER_3Layers_pretrained_fold1_09-06-2024_04_56-32_AM__epoch4.pth"
elif [ "$selected_model_number" == 26 ]; then 
    echo "BP4D_lr0.0001_weight12_AU2_RESNET50_3Layers_fold3_02-06-2024_12_02-31_PM__epoch4.pth"
    ./scripts/runBaseLearner.sh resnet50 3 BP4D "BP4D_lr0.0001_weight12_AU2_RESNET50_3Layers_fold3_02-06-2024_12_02-31_PM__epoch4.pth"
elif [ "$selected_model_number" == 27 ]; then 
    echo "BP4D_lr0.0001_weight12_AU7_SWIN_TRANSFORMER_3Layers_pretrained_fold1_09-06-2024_03_21-59_PM__epoch2.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight12_AU7_SWIN_TRANSFORMER_3Layers_pretrained_fold1_09-06-2024_03_21-59_PM__epoch2.pth"
elif [ "$selected_model_number" == 28 ]; then 
    echo "BP4D_lr0.0001_weight2_AU11_SWIN_TRANSFORMER_3Layers_pretrained_allLam01_fold1_10-06-2024_10_07-10_PM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight2_AU11_SWIN_TRANSFORMER_3Layers_pretrained_allLam01_fold1_10-06-2024_10_07-10_PM__epoch1.pth"
elif [ "$selected_model_number" == 29 ]; then 
    echo "BP4D_lr0.0001_weight3_AU3_SWIN_TRANSFORMER_3Layers_withMakeHuman_fold1_06-06-2024_12_52-43_PM__epoch3.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight3_AU3_SWIN_TRANSFORMER_3Layers_withMakeHuman_fold1_06-06-2024_12_52-43_PM__epoch3.pth"
elif [ "$selected_model_number" == 30 ]; then 
    echo "BP4D_lr0.0001_weight3_AU3_SWIN_TRANSFORMER_3Layers_withMakeHuman_fold2_06-06-2024_09_48-53_PM__epoch2.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight3_AU3_SWIN_TRANSFORMER_3Layers_withMakeHuman_fold2_06-06-2024_09_48-53_PM__epoch2.pth"
elif [ "$selected_model_number" == 31 ]; then 
    echo "BP4D_lr0.0001_weight3_AU3_SWIN_TRANSFORMER_3Layers_withMakeHuman_fold3_08-06-2024_05_36-09_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight3_AU3_SWIN_TRANSFORMER_3Layers_withMakeHuman_fold3_08-06-2024_05_36-09_AM__epoch1.pth"
elif [ "$selected_model_number" == 32 ]; then 
    echo "BP4D_lr0.0001_weight3_AU4_SWIN_TRANSFORMER_1Layers_pretrained_allLam01_fold1_11-06-2024_08_45-47_PM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 1 BP4D "BP4D_lr0.0001_weight3_AU4_SWIN_TRANSFORMER_1Layers_pretrained_allLam01_fold1_11-06-2024_08_45-47_PM__epoch1.pth"
elif [ "$selected_model_number" == 33 ]; then 
    echo "BP4D_lr0.0001_weight3_AU5_SWIN_TRANSFORMER_3Layers_fold2_21-05-2024_03_49-50_AM__epoch3.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight3_AU5_SWIN_TRANSFORMER_3Layers_fold2_21-05-2024_03_49-50_AM__epoch3.pth"
elif [ "$selected_model_number" == 34 ]; then 
    echo "BP4D_lr0.0001_weight3_AU9_resnet_3Layers_withMakeHuman_fold2_13-07-2024_04_32-28_PM__epoch2.pth"
    ./scripts/runBaseLearner.sh resnet50 3 BP4D "BP4D_lr0.0001_weight3_AU9_resnet_3Layers_withMakeHuman_fold2_13-07-2024_04_32-28_PM__epoch2.pth"
elif [ "$selected_model_number" == 35 ]; then 
    echo "BP4D_lr0.0001_weight3_AU9_resnet_3Layers_withMakeHuman_fold3_14-07-2024_01_00-45_AM__epoch2.pth"
    ./scripts/runBaseLearner.sh resnet50 3 BP4D "BP4D_lr0.0001_weight3_AU9_resnet_3Layers_withMakeHuman_fold3_14-07-2024_01_00-45_AM__epoch2.pth"
elif [ "$selected_model_number" == 36 ]; then 
    echo "BP4D_lr0.0001_weight4_AU1_SWIN_TRANSFORMER_3Layers_pretrained_fold1_09-06-2024_10_04-50_AM__epoch2.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight4_AU1_SWIN_TRANSFORMER_3Layers_pretrained_fold1_09-06-2024_10_04-50_AM__epoch2.pth"
elif [ "$selected_model_number" == 37 ]; then 
    echo "BP4D_lr0.0001_weight4_AU4_SWIN_TRANSFORMER_1Layers_pretrained_allLam01_fold2_12-06-2024_12_05-57_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 1 BP4D "BP4D_lr0.0001_weight4_AU4_SWIN_TRANSFORMER_1Layers_pretrained_allLam01_fold2_12-06-2024_12_05-57_AM__epoch1.pth"
elif [ "$selected_model_number" == 38 ]; then 
    echo "BP4D_lr0.0001_weight4_AU6_SWIN_TRANSFORMER_3Layers_withMakeHuman_fold2_07-06-2024_06_18-52_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight4_AU6_SWIN_TRANSFORMER_3Layers_withMakeHuman_fold2_07-06-2024_06_18-52_AM__epoch1.pth"
elif [ "$selected_model_number" == 39 ]; then 
    echo "BP4D_lr0.0001_weight4_AU6_SWIN_TRANSFORMER_3Layers_withMakeHuman_fold3_08-06-2024_07_23-18_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight4_AU6_SWIN_TRANSFORMER_3Layers_withMakeHuman_fold3_08-06-2024_07_23-18_AM__epoch1.pth"
elif [ "$selected_model_number" == 40 ]; then 
    echo "BP4D_lr0.0001_weight4_AU7_SWIN_TRANSFORMER_3Layers_pretrained_fold3_11-06-2024_09_05-56_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight4_AU7_SWIN_TRANSFORMER_3Layers_pretrained_fold3_11-06-2024_09_05-56_AM__epoch1.pth"
elif [ "$selected_model_number" == 41 ]; then 
    echo "BP4D_lr0.0001_weight4_AU9_resnet_3Layers_withMakeHuman_fold1_13-07-2024_08_11-23_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh resnet50 3 BP4D "BP4D_lr0.0001_weight4_AU9_resnet_3Layers_withMakeHuman_fold1_13-07-2024_08_11-23_AM__epoch1.pth"
elif [ "$selected_model_number" == 42 ]; then 
    echo "BP4D_lr0.0001_weight4_AU9_resnet_3Layers_withMakeHuman_fold2_13-07-2024_04_34-19_PM__epoch1.pth"
    ./scripts/runBaseLearner.sh resnet50 3 BP4D "BP4D_lr0.0001_weight4_AU9_resnet_3Layers_withMakeHuman_fold2_13-07-2024_04_34-19_PM__epoch1.pth"
elif [ "$selected_model_number" == 43 ]; then 
    echo "BP4D_lr0.0001_weight5_AU0_RESNET50_fold1_15-05-2024_09_41-16_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh resnet50 1 BP4D "BP4D_lr0.0001_weight5_AU0_RESNET50_fold1_15-05-2024_09_41-16_AM__epoch1.pth"
elif [ "$selected_model_number" == 44 ]; then 
    echo "BP4D_lr0.0001_weight5_AU0_RESNET50_fold3_15-05-2024_09_41-17_AM__epoch2.pth"
    ./scripts/runBaseLearner.sh resnet50 1 BP4D "BP4D_lr0.0001_weight5_AU0_RESNET50_fold3_15-05-2024_09_41-17_AM__epoch2.pth"
elif [ "$selected_model_number" == 45 ]; then 
    echo "BP4D_lr0.0001_weight5_AU11_SWIN_TRANSFORMER_3Layers_pretrained_allLam01_fold2_11-06-2024_01_38-13_AM__epoch3.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight5_AU11_SWIN_TRANSFORMER_3Layers_pretrained_allLam01_fold2_11-06-2024_01_38-13_AM__epoch3.pth"
elif [ "$selected_model_number" == 46 ]; then 
    echo "BP4D_lr0.0001_weight6_AU11_SWIN_TRANSFORMER_3Layers_pretrained_allLam01_fold3_11-06-2024_05_17-20_AM__epoch2.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight6_AU11_SWIN_TRANSFORMER_3Layers_pretrained_allLam01_fold3_11-06-2024_05_17-20_AM__epoch2.pth"
elif [ "$selected_model_number" == 47 ]; then 
    echo "BP4D_lr0.0001_weight6_AU2_RESNET50_3Layers_fold3_02-06-2024_12_51-26_PM__epoch3.pth"
    ./scripts/runBaseLearner.sh resnet50 3 BP4D "BP4D_lr0.0001_weight6_AU2_RESNET50_3Layers_fold3_02-06-2024_12_51-26_PM__epoch3.pth"
elif [ "$selected_model_number" == 48 ]; then 
    echo "BP4D_lr0.0001_weight6_AU3_SWIN_TRANSFORMER_3Layers_withMakeHuman_fold1_06-06-2024_12_52-44_PM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight6_AU3_SWIN_TRANSFORMER_3Layers_withMakeHuman_fold1_06-06-2024_12_52-44_PM__epoch1.pth"
elif [ "$selected_model_number" == 49 ]; then 
    echo "BP4D_lr0.0001_weight6_AU3_SWIN_TRANSFORMER_3Layers_withMakeHuman_fold3_08-06-2024_12_29-33_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight6_AU3_SWIN_TRANSFORMER_3Layers_withMakeHuman_fold3_08-06-2024_12_29-33_AM__epoch1.pth"
elif [ "$selected_model_number" == 50 ]; then 
    echo "BP4D_lr0.0001_weight6_AU4_SWIN_TRANSFORMER_1Layers_pretrained_allLam01_fold3_12-06-2024_03_51-30_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 1 BP4D "BP4D_lr0.0001_weight6_AU4_SWIN_TRANSFORMER_1Layers_pretrained_allLam01_fold3_12-06-2024_03_51-30_AM__epoch1.pth"
elif [ "$selected_model_number" == 51 ]; then 
    echo "BP4D_lr0.0001_weight6_AU5_SWIN_TRANSFORMER_3Layers_fold3_21-05-2024_01_05-29_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight6_AU5_SWIN_TRANSFORMER_3Layers_fold3_21-05-2024_01_05-29_AM__epoch1.pth"
elif [ "$selected_model_number" == 52 ]; then 
    echo "BP4D_lr0.0001_weight7_AU0_RESNET50_fold1_15-05-2024_09_41-16_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh resnet50 1 BP4D "BP4D_lr0.0001_weight7_AU0_RESNET50_fold1_15-05-2024_09_41-16_AM__epoch1.pth"
elif [ "$selected_model_number" == 53 ]; then 
    echo "BP4D_lr0.0001_weight7_AU0_RESNET50_fold2_15-05-2024_09_41-16_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh resnet50 1 BP4D "BP4D_lr0.0001_weight7_AU0_RESNET50_fold2_15-05-2024_09_41-16_AM__epoch1.pth"
elif [ "$selected_model_number" == 54 ]; then 
    echo "BP4D_lr0.0001_weight7_AU1_SWIN_TRANSFORMER_3Layers_pretrained_fold2_09-06-2024_09_11-29_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight7_AU1_SWIN_TRANSFORMER_3Layers_pretrained_fold2_09-06-2024_09_11-29_AM__epoch1.pth"
elif [ "$selected_model_number" == 55 ]; then 
    echo "BP4D_lr0.0001_weight7_AU2_RESNET50_3Layers_fold1_30-05-2024_03_12-31_PM__epoch4.pth"
    ./scripts/runBaseLearner.sh resnet50 3 BP4D "BP4D_lr0.0001_weight7_AU2_RESNET50_3Layers_fold1_30-05-2024_03_12-31_PM__epoch4.pth"
elif [ "$selected_model_number" == 56 ]; then 
    echo "BP4D_lr0.0001_weight7_AU6_SWIN_TRANSFORMER_3Layers_withMakeHuman_fold1_06-06-2024_09_04-20_PM__epoch4.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight7_AU6_SWIN_TRANSFORMER_3Layers_withMakeHuman_fold1_06-06-2024_09_04-20_PM__epoch4.pth"
elif [ "$selected_model_number" == 57 ]; then 
    echo "BP4D_lr0.0001_weight8_AU11_SWIN_TRANSFORMER_3Layers_pretrained_allLam01_fold1_10-06-2024_10_05-54_PM__epoch2.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight8_AU11_SWIN_TRANSFORMER_3Layers_pretrained_allLam01_fold1_10-06-2024_10_05-54_PM__epoch2.pth"
elif [ "$selected_model_number" == 58 ]; then 
    echo "BP4D_lr0.0001_weight8_AU11_SWIN_TRANSFORMER_3Layers_pretrained_allLam01_fold2_11-06-2024_01_37-08_AM__epoch2.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight8_AU11_SWIN_TRANSFORMER_3Layers_pretrained_allLam01_fold2_11-06-2024_01_37-08_AM__epoch2.pth"
elif [ "$selected_model_number" == 59 ]; then 
    echo "BP4D_lr0.0001_weight8_AU11_SWIN_TRANSFORMER_3Layers_pretrained_allLam01_fold3_11-06-2024_05_17-52_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight8_AU11_SWIN_TRANSFORMER_3Layers_pretrained_allLam01_fold3_11-06-2024_05_17-52_AM__epoch1.pth"
elif [ "$selected_model_number" == 60 ]; then 
    echo "BP4D_lr0.0001_weight8_AU2_RESNET50_3Layers_fold2_02-06-2024_09_18-54_AM__epoch2.pth"
    ./scripts/runBaseLearner.sh resnet50 3 BP4D "BP4D_lr0.0001_weight8_AU2_RESNET50_3Layers_fold2_02-06-2024_09_18-54_AM__epoch2.pth"
elif [ "$selected_model_number" == 61 ]; then 
    echo "BP4D_lr0.0001_weight8_AU4_SWIN_TRANSFORMER_1Layers_pretrained_allLam01_fold3_12-06-2024_02_53-25_AM__epoch2.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 1 BP4D "BP4D_lr0.0001_weight8_AU4_SWIN_TRANSFORMER_1Layers_pretrained_allLam01_fold3_12-06-2024_02_53-25_AM__epoch2.pth"
elif [ "$selected_model_number" == 62 ]; then 
    echo "BP4D_lr0.0001_weight8_AU5_SWIN_TRANSFORMER_3Layers_fold1_20-05-2024_05_59-22_PM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight8_AU5_SWIN_TRANSFORMER_3Layers_fold1_20-05-2024_05_59-22_PM__epoch1.pth"
elif [ "$selected_model_number" == 63 ]; then 
    echo "BP4D_lr0.0001_weight8_AU6_SWIN_TRANSFORMER_3Layers_withMakeHuman_fold1_06-06-2024_07_52-24_PM__epoch2.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight8_AU6_SWIN_TRANSFORMER_3Layers_withMakeHuman_fold1_06-06-2024_07_52-24_PM__epoch2.pth"
elif [ "$selected_model_number" == 64 ]; then 
    echo "BP4D_lr0.0001_weight8_AU9_resnet_3Layers_withMakeHuman_fold1_13-07-2024_08_07-31_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh resnet50 3 BP4D "BP4D_lr0.0001_weight8_AU9_resnet_3Layers_withMakeHuman_fold1_13-07-2024_08_07-31_AM__epoch1.pth"
elif [ "$selected_model_number" == 65 ]; then 
    echo "BP4D_lr0.0001_weight9_AU5_SWIN_TRANSFORMER_3Layers_fold1_20-05-2024_05_59-22_PM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight9_AU5_SWIN_TRANSFORMER_3Layers_fold1_20-05-2024_05_59-22_PM__epoch1.pth"
elif [ "$selected_model_number" == 66 ]; then 
    echo "BP4D_lr0.0001_weight9_AU5_SWIN_TRANSFORMER_3Layers_fold3_21-05-2024_01_18-06_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight9_AU5_SWIN_TRANSFORMER_3Layers_fold3_21-05-2024_01_18-06_AM__epoch1.pth"
elif [ "$selected_model_number" == 67 ]; then 
    echo "BP4D_lr0.0001_weight9_AU6_SWIN_TRANSFORMER_3Layers_withMakeHuman_fold3_08-06-2024_07_23-18_AM__epoch2.pth"
    ./scripts/runBaseLearner.sh swin_transformer_base 3 BP4D "BP4D_lr0.0001_weight9_AU6_SWIN_TRANSFORMER_3Layers_withMakeHuman_fold3_08-06-2024_07_23-18_AM__epoch2.pth"
elif [ "$selected_model_number" == 68 ]; then 
    echo "BP4D_lr0.0001_weight9_AU9_resnet_3Layers_withMakeHuman_fold3_14-07-2024_12_47-32_AM__epoch1.pth"
    ./scripts/runBaseLearner.sh resnet50 3 BP4D "BP4D_lr0.0001_weight9_AU9_resnet_3Layers_withMakeHuman_fold3_14-07-2024_12_47-32_AM__epoch1.pth"
# end of the list of models
else
    echo "Invalid model number"
fi