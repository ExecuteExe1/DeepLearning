## Traffic Sign Recognition Using Radial Basis Function Neural Networks

This project implements a custom Radial Basis Function (RBF) Neural Network for classifying 43 categories of German traffic signs using the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

The network uses K-Means clustering to initialize class-specific RBF centers, Gaussian radial basis activation functions with class-wise spread parameters (σ), and regularized least-squares optimization to determine output weights.

Images are resized to 32×32 pixels, normalized, standardized, and processed using Principal Component Analysis (PCA), preserving 90% of the dataset's variance.

The implementation evaluates different RBF center configurations, automatically selecting the optimal number of centers based on validation accuracy. Performance is benchmarked against 1-Nearest Neighbor (1-NN), 3-Nearest Neighbor (3-NN), and Nearest Centroid classifiers.

The project includes training, validation, and test accuracy measurements, confusion matrices, RBF center visualizations, prediction examples, and analysis of the trade-offs between model complexity, accuracy, and training time.

**Technologies:** Python, NumPy, Scikit-learn, SciPy, Pandas, Matplotlib, Pillow.
