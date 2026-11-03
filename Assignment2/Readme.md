## Traffic Sign Recognition Using Machine Learning

This project implements a machine learning pipeline for classifying 43 categories of German traffic signs using the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

The implementation includes image preprocessing, RGB normalization, feature standardization, and Principal Component Analysis (PCA) to reduce dimensionality while preserving 90% of the dataset's variance.

Three Support Vector Machine (SVM) kernels—Linear, Radial Basis Function (RBF), and Polynomial—are trained and evaluated using separate training, validation, and testing datasets.

An optional benchmarking stage compares SVM performance against 1-Nearest Neighbor (1-NN), 3-Nearest Neighbor (3-NN), Nearest Centroid, and an SGD-based linear classifier with hinge loss.

The project generates accuracy comparisons, training-time measurements, PCA visualizations, and JSON reports to evaluate classification performance.

**Technologies:** Python, Scikit-learn, NumPy, Pandas, Matplotlib, Pillow.
