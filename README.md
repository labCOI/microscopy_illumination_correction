# Crucial Factors Affecting Uneven Illumination Correction Algorithms in Whole Slide Imaging 

## Introduction
Uneven illumination is a common artifact in images obtained from optical microscope devices, causing degradation in whole slide images. This artifact not only deteriorates the visual quality by creating a black plaid pattern but also introduces errors in subsequent quantitative analyses. Therefore, correcting uneven illumination is an essential preprocessing step. Existing effective correction algorithms fall into two main categories: analytical and deep learning-based algorithms. However, each has drawbacks that limit their practical applicability. This study investigates the most effective algorithms and highlights the key factors that influence the estimation of uneven illumination patterns.
## Objective
The objective of this study is to thoroughly investigate both analytical and deep learning algorithms for uneven illumination correction in whole slide imaging. We examine the performance of three effective analytical algorithms—CIDRE, BaSiC, and TAK—by analyzing their merits, demerits, and dependency on input stack size. Additionally, we explore a sparsity-based deep learning algorithm to assess its generalization capabilities and identify its weaknesses. Ultimately, we aim to reveal crucial factors essential for developing robust algorithms tailored for uneven illumination correction in whole slide imaging techniques.
## Methods
This study evaluates both analytical and deep learning-based approaches for correcting uneven illumination in histopathology images. We analyzed three analytical algorithms (CIDRE, BaSiC, and TAK) and highlighted their dependence on input stack size, texture, and density.

We also investigated a recent deep learning method, referred to as Wang, and identified its shortcomings even on the training set. To address this, we designed three training sets (TS1, TS2, and TS3) to evaluate the impact of different augmentation strategies:

TS1: Augmentation of a single sample with one uneven illumination pattern.

TS2: Augmentation of multiple uneven illumination patterns.

TS3: Augmentation of both illumination patterns and textures.

Due to the high computational cost of processing large, high-resolution images, in the original paper the training images are patched  into smaller sizes. However, this approach proved ineffective as it prevented the network from learning the entire illumination pattern. Instead, we cropped smaller images from the database and applied resized, complete illumination patterns. This strategy maintained image quality, reduced computational costs, and improved network training efficiency.
## Results
Analytical algorithms (CIDRE, BaSiC, and TAK) showed that performance depends significantly on the number of input images and the texture's density, deteriorating with fewer than 100 images.

The deep learning method (Wang) initially performed poorly even on the training set. By using three training sets (TS1, TS2, and TS3), we found that incorporating both illumination patterns and textures (TS3) produced the best results. Patching images was ineffective; instead, using smaller crops with complete illumination patterns improved performance and reduced computational costs.
## Conclusion
This study highlights the importance of input stack size, texture, and density in the performance of analytical algorithms for uneven illumination correction. For deep learning approaches, our findings suggest that effective training requires a global view of illumination patterns and textures. The designed training set TS3, which includes comprehensive augmentation, significantly improves performance. These insights are crucial for developing robust and efficient algorithms for uneven illumination correction in whole slide imaging.
