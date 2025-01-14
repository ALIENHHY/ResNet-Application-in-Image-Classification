ResNet-Based Image Classification for Environmental Action

This project provides the implementation of an image classification feature for the mini-program "Environmental Action Pioneer" (translated from Chinese: “鹅民环保行动派”).

Project Overview

Project Name: Image Recognition for "Environmental Action Pioneer" Mini-Program

Purpose: To detect specific objects (e.g., bicycles, electric bikes, buses) or perform text recognition tasks in images using ResNet neural networks and OCR algorithms. This project aims to enhance the program’s capabilities in recognizing various items and text for diverse classification tasks.

Installation Guide

Dependencies:

Environment: Python 3.10

Hardware: Tested on an RTX 3060 GPU

Libraries: Includes but is not limited to the following:

pytorch

PIL

pytesseract

Installation Steps:

Ensure your environment meets the requirements.

Install the necessary dependencies listed above. If errors occur, install additional packages as prompted.

Usage Instructions

Input/Output Format:

Object Detection: The input is an image file, and the output is a confidence score indicating the likelihood that the specified object is present in the image.

Text Recognition: The input is an image file, and the output is binary (0 or 1), representing whether the target text is detected.

Parameters:

Threshold: Default is 0.5. When the confidence exceeds the threshold, the object is considered present. This value is adjustable.

Target Text: For text recognition tasks, this parameter defines the text to be detected (e.g., "No disposable cutlery"). It can be customized during execution.

Training Dataset: The dataset is divided for various tasks and stored in the root directory.

Testing and Results

Testing Methodology:

A test set of 50 images, including bicycles, electric bikes, buses (interior and exterior), is used for evaluation. The test images are stored in the predict folder.

Example Results:

Detailed results can be found in the associated documentation.

Project Structure

Root Directory: Contains all the code, datasets, and .pth files with trained model parameters.

Key Files:

Code files: Python scripts for training, inference, and utility functions.

Datasets: Subdivided by classification tasks.

Model files: Pretrained weights for ResNet-based models.

Suggested Workflow

For correctly recognized cases, proceed with standard automation.

For unrecognized or misclassified cases, escalate for manual review.

Known Issues and Recommendations

Unrecognized Scenarios: Photos of unrelated objects (e.g., watches) are currently redirected to manual review.

Misclassification: High confusion rates between bicycles and electric bikes. Consider adjusting the threshold.

Text Recognition Challenges: Difficulties in classifying phrases such as "Bring your own food container" or "Opt out of cutlery." These could be grouped for simplified evaluation.

Dataset Expansion: The training dataset requires further enrichment for better generalization.

Notes

Due to company policies, the dataset includes only a subset of the images, and the .pth files are provided for reference purposes only.
