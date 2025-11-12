# Weakly Supervised Image Segmentation Exercise 2.4

## Introduction
This exercise is the second part of the project you will present for Segmentation.

In this exercise, you will be working with the skin lesion segmentation dataset **PH2** that you used in the first part. Your task is to train a segmentation architecture with different weak supervision types, such as point clicks (i.e., no full segmentation masks during training), and validate your results for segmentation performance.

---

## Tasks

1. **Creating Weak Annotations (Point Clicks):**
    * Create your weak annotations (point clicks) which you will use to train a weakly supervised segmentation model.
    * You do not have to click on the images yourself manually but you should simulate this by using the provided full segmentation masks.
    * You have to think about a sampling strategy that simulates a user annotator behavior that clicks on the object of interest (positive clicks) and on the background (negative clicks). (The weak annotations for the training set are not provided.)

2. **Implementing Point-Level Supervision:**
    * Use the same architecture as the first part.
    * Implement a loss function for point-level supervision, and modify your dataloader accordingly.

3. **Consistency of Splits and Hyperparameters:**
    * Make sure that the train/val/test splits and other hyperparameters are as close as possible to the models that you trained in the first part.
    * This will allow you to validate the segmentation performance and compare the results with the fully supervised counterparts of the first part.

4. **Ablation Study on Clicks and Sampling:**
    * Perform an ablation study where you change the **number of clicks** (positive and negative).
    * Report how the different numbers of clicks affect the segmentation performance.
    * Determine how many clicks you need to achieve a comparable performance with the fully supervised models.
    * Also, experiment with **different sampling strategies** for the user clicks (i.e., random sampling inside and outside the segmentation mask) since this may affect your performance.

5. **Optional: Experiment with Bounding Boxes:**
    * Experiment with other types of weak annotations, such as bounding boxes.
    * For this, you can follow the iterative approach discussed during the previous lecture, where you iteratively:
        * (a) use current segmentation as labels to train a segmentation network
        * (b) use the segmentation network to predict a new set of segmentation

6. **Use of External Tools:**
    * Did you use ChatGPT or similar tools? If yes, please briefly describe how you used it and how they were useful.

---

Your process, performance evaluation, and results should be documented and discussed in project report to be uploaded on DTU Learn. The deadline is on Sunday 16th of November at 20h.
