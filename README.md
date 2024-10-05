# DETECTIFY-CNN-Based-Image-Tampering-Detection
DETECTIFY: CNN-Based Image Tampering Detection is a project focused on identifying whether an image has been tampered with or manipulated. The goal of the project is to detect alterations in images, such as edits, forgeries, or modifications that might not be visible to the human eye.

In this project, a technique called Error Level Analysis (ELA) is used to highlight areas in an image that might have been changed. ELA works by analyzing the compression levels in an image. When an image is edited, different parts may have different error levels, which can indicate tampering.

To make the detection more accurate, a Convolutional Neural Network (CNN) is employed. CNN is a type of deep learning model that is particularly good at recognizing patterns in images. By training the CNN with images that are both original and tampered, it learns to identify the subtle differences between them.

The result is a model that can accurately predict whether an image has been manipulated. This technology is useful for detecting fake images, deepfakes, or any form of image fraud in fields like forensics, journalism, and digital media.
