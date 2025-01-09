# MRIAnatEval
Git repository for MICCAI 2024 <br>
Model Checkpoint: https://drive.google.com/drive/folders/1EYNj_Pz2q7Jw7TuZz3FbCqCwqjOGuuLL?usp=sharing <br>
Please note that we do not have 256, 256, 256 shape so we made slight modification on GAN models!
## Metrics included:
1. FID
2. MMD
3. MS-SSIM
4. PCA-tSNE
5. Segmentation Quality Control

## Models:
1. AE-style GANs: https://github.com/cyclomon/3dbraingen
2. HA-GAN: https://github.com/batmanlab/HA-GAN
3. Conditional DPM: https://arxiv.org/abs/2212.08034
4. MedSyn: https://ieeexplore.ieee.org/document/10566053
5. MONAI: https://github.com/Project-MONAI/GenerativeModels

## Traditional Metrics Evaluation
1. Generate your own samples
2. Run evaluation.py or simply use relevant evaluation functions

## Anatomical-based Evaluation
To run the 2-stage Anatomical-based evaluation, you need to follow these steps:
1. Generate certain number of Brain MRI images, and real MRI images. Use Synthseg+ to do a whole brain segmentation for both data, SynthSeg+: https://github.com/BBillot/Synthseg
2. Run quality control evaluation: Calculate the proportion of each ROI where the quality control exceeds 0.65, and set a threshold to detect the generated results, which we set at 0.95.
3. Run subcortical evaluation and cortical evalution: Replace the CSV path with the segmentation output of your generated data and the generated data of the real image, and perform regression and Cohen's d calculation. The scripts are [run_eval_aseg.py](./run_eval_aseg.py) and [run_eval_aparc.py](./run_eval_aparc.py), respectively, for subcortical and cortical segmentation results.

