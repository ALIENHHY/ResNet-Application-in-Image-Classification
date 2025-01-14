ResNet-Based Image Classification for Environmental Action
---

This project provides the implementation of an image classification feature for the WeChat mini-program "Environmental Action Pioneer" (translated from Chinese: “鹅民环保行动派”).

* Installation Guide

  * Libraries: Includes but is not limited to the following: pytorch, PIL and pytesseract

* Usage Instructions

  * Input/Output Format:

    * Object Detection: The input is an image file, and the output is a confidence score indicating the likelihood that the specified object is present in the image.

    * Text Recognition: The input is an image file, and the output is binary (0 or 1), representing whether the target text is detected.

  * Parameters:

    * Threshold: Default is 0.5. When the confidence exceeds the threshold, the object is considered present. This value is adjustable.

    * Training Dataset: The dataset is divided for various tasks and stored in the root directory.

* Testing and Results

  * Testing Methodology:

A test set of 50 images, including bicycles, electric bikes, buses (interior and exterior), is used for evaluation. The test images are stored in the predict folder.

  * Example Results:

Detailed results can be found in the associated documentation.

* Project Structure

  * Root Directory: Contains all the code, datasets, and .pth files with trained model parameters.

  * Key Files:

    * Code files: Python scripts for training, inference, and utility functions.

    * Datasets: Subdivided by classification tasks.

    * Model files: Pretrained weights for ResNet-based models.

* Notes

Due to company policies, the dataset includes only a subset of the images, and the .pth files are provided for reference purposes only.
