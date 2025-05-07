# BYU - Locating Bacterial Flagellar Motors 2025

Kaggle URL: [https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025)

# Project Goal: Help locate flagellar motors in three-dimensional reconstructions of bacteria.

## **Overview**

The goal of this competition is to develop an algorithm to identify the presence and location of flagellar motors in 3D reconstructions of bacteria. Automating this traditionally manual task will accelerate the study of macromolecular complexes, which helps answer fundamental questions in molecular biology, improve drug development, and advance synthetic biology. 

## Description

### Introduction

The flagellar motor is a molecular machine that facilitates the motility of many microorganisms, playing a key role in processes ranging from chemotaxis to pathogenesis. Cryogenic electron tomography (cryo-ET) has enabled us to image these nanomachines in near-native conditions. But identifying flagellar motors in these three-dimensional reconstructions (tomograms) is labor intensive. Factors such as a low signal-to-noise ratio, variable motor orientations, and the complexity of crowded intracellular environments complicate automated identification. Cryo-ET studies become limited by the bottle-neck of a human in the loop. In this contest, your task is to develop an image processing algorithm that identifies the location of a flagellar motor, if it is present.

A **tomogram** is a three-dimensional image that has been *reconstructed* from a series of 2D projection images. The images in this challenge are tomograms of bacteria that have been flash-frozen in ice, which preserves the molecular structure of the bacteria for the imaging process. [This video](https://www.cellstructureatlas.org/6-2-flagellar-motor.html) walks through slices of a tomogram highlighting different features of a bacterial cell, including a flagellar motor. The accompanying text describes the purpose and function of the motor.

## Evaluation

### Evaluation Metric

Submissions will be evaluated using a combination of the F𝛽Fβ-score and Euclidean distance. The goal is to determine whether a tomogram contains a motor and, if it does, to accurately predict its location.

Let the ground truth be 𝑦y and the predicted location be 𝑦¯y¯. The Euclidean distance |𝑦−𝑦¯|2|y−y¯|2 determines classification:

- **True Positive (TP):** If , the prediction is within threshold.
    
    |𝑦−𝑦¯|2≤𝜏
    
    |y−y¯|2≤τ
    
- **False Negative (FN):** If , the prediction is outside of threshold.
    
    |𝑦−𝑦¯|2>𝜏
    
    |y−y¯|2>τ
    

where  𝜏=1000 τ=1000 Angstroms.

### **F𝛽Fβscore**

The F𝛽Fβ-score balances precision and recall, placing greater weight on recall when 𝛽>1β>1 and on precision when 𝛽<1β<1 (in our case we use **𝛽=2β=2**, thus we are weighting recall more than precision). It is defined as:

F𝛽=(1+𝛽2)⋅precision⋅recall(𝛽2⋅precision)+recall=(1+𝛽2)⋅TP(1+𝛽2)⋅TP+𝛽2⋅FN+FPFβ=(1+β2)⋅precision⋅recall(β2⋅precision)+recall=(1+β2)⋅TP(1+β2)⋅TP+β2⋅FN+FP

This metric ensures that both the presence and location accuracy of predicted motors are considered in the final score.

[Link to metric notebook](https://www.kaggle.com/code/metric/byu-biophysics-91249)

## **Submission Format**

Your submission should be a CSV file with one row per tomogram found in the test set. **IMPORTANT:** If you predict that no motor exists, set `Motor axis 0`, `Motor axis 1`, `Motor axis 2` to `-1`

```
tomo_id,Motor axis 0,Motor axis 1,Motor axis 2
tomo_003acc,501.1,22.8,429.8
tomo_00e047,-1,-1,-1
tomo_01a877,395.2,335.4,798.0
etc.
```

## Timeline

- **March 5, 2025** - Start Date.
- **May 28, 2025** - Entry Deadline. You must accept the competition rules before this date in order to compete.
- **May 28, 2025** - Team Merger Deadline. This is the last day participants may join or merge teams.
- **June 4, 2025** - Final Submission Deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Code Requirements

### **This is a Code Competition**

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

- CPU Notebook <= 12 hours run-time
- GPU Notebook <= 12 hours run-time
- Internet access disabled
- Freely & publicly available external data is allowed, including pre-trained models
- Submission file must be named `submission.csv`
- Submission runtimes have been slightly obfuscated. If you repeat the exact same submission you will see up to 15 minutes of variance in the time before you receive your score.

Please see the [Code Competition FAQ](https://www.kaggle.com/docs/competitions#notebooks-only-FAQ) for more information on how to submit. And review the [code debugging doc](https://www.kaggle.com/code-competition-debugging) if you are encountering submission errors.

# **Dataset Description**

In this competition you are tasked with finding flagellar motor centers in 3D tomograms. A tomogram is a 3D volumetric representation of an object. In this competition, each tomogram is provided as a set of 2D image slices (JPEG) stored in a unique directory. You are tasked with predicting points where a flagellar motor is located when one is present.

## **Files and Directories**

**Parent Directory**: byu-locating-bacterial-flagellar-motors-2025

**train/**: Directory of subdirectories each containing a stack of tomogram slices to be used for training. Each tomogram subdirectory comprises JPEGs where each JPEG is a 2D slice of a tomogram.

**train_labels.csv**: Training data labels. Each row represents a unique motor location and not a unique tomogram.

- `row_id` - index of the row
- `tomo_id` - unique identifier of the tomogram. Some tomograms in the train set have multiple motors.
- `Motor axis 0` - the z-coordinate of the motor, i.e., which slice it is located on
- `Motor axis 1` - the y-coordinate of the motor
- `Motor axis 2` - the x-coordinate of the motor
- `Array shape axis 0` - z-axis length, i.e., number of slices in the tomogram
- `Array shape axis 1` - y-axis length, or width of each slice
- `Array shape axis 2` - x-axis length, or height of each slice
- `Voxel spacing` - scaling of the tomogram; angstroms per voxel
- `Number of motors` - Number of motors in the tomogram. Note that each row represents a motor, so tomograms with multiple motors will have several rows to locate each motor.

**test/**: Directory with 3 directories of dummy test tomograms; the rerun test dataset contains approximately 900 tomograms. **The test data only contain tomograms with one or zero motors**.

**sample_submission.csv**: Sample submission file in the correct format. (If you predict that no motor exists, set `Motor axis 0`, `Motor axis 1`, and `Motor axis 2` to '-1')

## Data Analysis and Insights:

### 1. Tomogram Size Variation

The tomograms show significant size variation, particularly in the z-axis (depth):

- Mean dimensions: (415.4, 953.4, 954.8)
- Standard deviation: (183.0, 68.5, 103.6)
- Range: from (300, 924, 924) to (800, 1912, 1847)

This substantial variation means we'll need to implement a dynamic approach for handling different-sized inputs. The z-axis has the most variation (with a standard deviation of 183 voxels), while the x and y dimensions are more consistent but still variable.

### 2. Signal-to-Noise Ratio

The analysis shows poor signal-to-noise characteristics:

- Mean SNR: -0.370
- Negative SNR values indicate that the motor signal is weaker than the surrounding noise

This will be challenging and requires advanced denoising techniques during preprocessing. The negative SNR means that motors aren't immediately distinguishable from background noise using simple intensity thresholding.

### 3. Motor Appearance Consistency

Motors show moderate intensity consistency:

- Motor intensity coefficient of variation: 0.27
- Mean contrast: -0.105 (negative values indicate motors are darker than surroundings)

The coefficient of variation (0.27) suggests reasonable consistency in appearance across different tomograms, but the negative contrast indicates motors appear as darker regions compared to their surroundings, which is important for our detection approach.

### 4. Computational Resources

Memory requirements are substantial:

- Average tomogram: ~1.4 GB in float32 format
- A batch of 10 tomograms would require ~43 GB including gradients

This confirms we'll need to use chunking/cropping strategies rather than processing whole tomograms at once. Our 3D CNN with 64³ input blocks is much more manageable at ~20-30 MB per model.

### 5. Motor Positioning Patterns

Motors show interesting positioning patterns:

- Fairly centered on average (mean relative positions ~0.5 on all axes)
- 47.7% of motors are within 20% of an edge
- 11.5% of motors are very close to edges (within 10%)

This suggests we should pay special attention to edge cases during training and may need special handling for motors near boundaries.

### 6. Class Imbalance

The dataset is relatively balanced:

- 55.9% of tomograms contain motors
- 44.1% have no motors
- Ratio with = 1:0.79

While there's a slight imbalance, it's not severe enough to require extreme measures. However, most tomograms (48.3%) have exactly one motor, with only a small percentage having multiple motors. The test set will only have 0 or 1 motor per tomogram.

### 7. Test Set Characteristics

The test set appears similar to the training set:

- Mean intensity ratio (test/train): 0.890
- Std intensity ratio: 1.314

The test set has slightly lower mean intensity but higher variance than the training set. This suggests our preprocessing needs to normalize intensity across tomograms.

### 8. Voxel Spacing

There's significant voxel spacing variation:

- Mean: 15.066 Å
- Range: 6.5 Å to 19.7 Å
- 10 unique spacing values

This means the 1000 Å threshold in the evaluation metric translates to between ~51 and ~154 voxels depending on the tomogram. We'll need to account for this variable physical scale when designing our model and processing pipeline.

### Downstream Model Implications

Based on these findings, we should focus on:

1. **Implementing a robust preprocessing pipeline** with effective denoising methods to handle the poor SNR
2. **Creating a data generator** that handles variable-sized tomograms and extracts appropriate chunks
3. **Developing a normalization strategy** that accounts for intensity variations across tomograms
4. **Designing a model** that works well with the identified motor characteristics (darker than surroundings)
5. **Implementing distance measurement** that takes voxel spacing into account during evaluation

## Citation

Andrew Darley, Braxton Owens, Bryan Morse, Eben Lonsdale, Gus Hart, Jackson Pond, Joshua Blaser, Matias Gomez Paz, Matthew Ward, Rachel Webb, Andrew Crowther, Nathan Smith, Grant J. Jensen, TJ Hart, Maggie Demkin, Walter Reade, and Elizabeth Park. BYU - Locating Bacterial Flagellar Motors 2025. [https://kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025](https://kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025), 2025. Kaggle.