# Deep learning examples for Cancer Analysis

Here you can find two examples for the main inputs data in cancer analysis (images and genomics). These implementations are in pytorch and you can train and run it on both CPU as well as GPU.

The first example for image classification is using the Breast Cancer Histopathological Database (BreakHis) [[1]](#1). This dataset has 9,109 microscopic images of breast tumor tissue from 82 patients. 
The image collection was using different magnifying factors (40X, 100X, 200X and 400X) and has available images from malignant and non-malignant samples. We used different torchvision models for image classification, based on convolutional neural networks.

For the genomic example we used 4,645 samples from 19 patients of single cell RNA-seq of melanoma tumors [[2]](#2). The dataset is available at the Gene Expression Omnibus repository with the GEO accession number GSE72056. The architecure is an autoencoder for dimensionality reduction. Visualizing the data as clusters according to the type of cell fr the malignant samples. 

## References
<a id="1">[1]</a> 
Spanhol, F. A., Oliveira, L. S., Petitjean, C., & Heutte, L. (2016). 
Breast cancer histopathological image classification using convolutional neural networks.
2016 International Joint Conference on Neural Networks (IJCNN), pp. 2560-2567.

<a id="2">[2]</a>
Tirosh, I., Izar, B., Prakadan, S. M., Wadsworth, M. H., Treacy, D., Trombetta, J. J., ... & Fallahi-Sichani, M. (2016). 
Dissecting the multicellular ecosystem of metastatic melanoma by single-cell RNA-seq. 
Science, 352(6282), 189-196.
