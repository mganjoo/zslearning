Zero-shot learning via cross-modality transfer
==============================================

This package contains the code for the paper Zero-Shot Learning Through
Cross-Modal Transfer presented at NIPS 2013.

Please cite the code as follows:

Zero-Shot Learning Through Cross-Modal Transfer.
Richard Socher, Milind Ganjoo, Christopher D. Manning, Andrew Y. Ng.
Advances in Neural Information Processing Systems (NIPS 2013).

# Getting started

To run our model and generate some of the figures presented in the paper, follow
the following steps.

## Preparing image data

1. Download MATLAB version of dataset from
   http://www.cs.toronto.edu/~kriz/cifar.html.
2. Merge matrices from different batches into one file -- train.mat -- and
   rename the test batch to test.mat and the meta file to meta.mat.
3. Create folder image_data and its subdirectories, images/ and features/.
3. Move all three files to image_data/images/cifar10.
4. Run buildFeatures.m in buildFeatures/ to create image features for use in
   our model. The script will create train.mat and test.mat in
   image_data/features/cifar10. [1]

## Preparing word vectors

We primarily used word vectors from [2], extracting 10 relevant vectors for
CIFAR-10 and 96 vectors for CIFAR-100 (vectors for 4 categories from CIFAR-100
were not present in the vocabulary). You may use these directly. You can also
use a different source of word vectors, saving the extracted word table for
the CIFAR-10 classes under word_data/(dataset_name)/cifar10/wordTable.mat

## Running model

The model can be run by executing main.m in the root directory, which will
tune model parameters and generate a few graphs.

# References

[1] The Importance of Encoding Versus Training with Sparse Coding and Vector
Quantization. A. Coates and A. Ng. In ICML, 2011.

[2] Improving Word Representations via Global Context and Multiple Word
Prototypes. Eric H. Huang, Richard Socher, Christopher D. Manning and Andrew Y.
Ng. Association for Computational Linguistics 2012 Conference (ACL 2012).

