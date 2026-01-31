import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import random
from scipy.spatial.distance import cdist
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.cluster import KMeans

print("\n" + "="*80)
print("🚦 RBF NETWORK FOR GTSRB TRAFFIC SIGN RECOGNITION - FIXED VERSION")
print("="*80)

# ---------------- TRAFFIC SIGN LABELS ----------------
TRAFFIC_SIGN_NAMES_SHORT = {
    0: '20 km/h', 1: '30 km/h', 2: '50 km/h', 3: '60 km/h', 4: '70 km/h',
    5: '80 km/h', 6: 'End 80', 7: '100 km/h', 8: '120 km/h', 9: 'No passing',
    10: 'No trucks', 11: 'Right-of-way', 12: 'Priority road',
    13: 'Yield', 14: 'Stop', 15: 'No vehicles', 16: 'No trucks', 17: 'No entry',
    18: 'Caution', 19: 'Curve left', 20: 'Curve right', 21: 'Double curve',
    22: 'Bumpy road', 23: 'Slippery', 24: 'Narrows right', 25: 'Road work',
    26: 'Traffic light', 27: 'Pedestrians', 28: 'Children', 29: 'Bicycles',
    30: 'Ice/snow', 31: 'Animals', 32: 'End limits', 33: 'Turn right',
    34: 'Turn left', 35: 'Ahead only', 36: 'Straight/right', 37: 'Straight/left',
    38: 'Keep right', 39: 'Keep left', 40: 'Roundabout', 41: 'End no passing',
    42: 'End no trucks'
}

# ---------------- FIXED DATA LOADER (Based on working code) ----------------
class FixedGTSRBLoader:
    """Fixed loader based on the working version"""
    
    @staticmethod
    def load_train_data():
        """Load ALL training data from Train/ numbered folders"""
        print("\n📥 LOADING TRAINING DATA...")
        
        train_folder = Path('Train')
        
        if not train_folder.exists():
            print(f"❌ Train folder not found at: {train_folder.absolute()}")
            return None, None
        
        X = []
        y = []
        class_counts = {}
        
        # Get all numbered folders (0/, 1/, 2/, ...)
        class_folders = []
        for item in train_folder.iterdir():
            if item.is_dir() and item.name.isdigit():
                try:
                    class_id = int(item.name)
                    if 0 <= class_id <= 42:
                        class_folders.append((class_id, item))
                except:
                    continue
        
        if not class_folders:
            print(f"❌ No numbered folders found in Train/")
            return None, None
        
        print(f"  Found {len(class_folders)} class folders")
        
        total_images = 0
        for class_id, class_folder in tqdm(class_folders, desc="Loading classes"):
            # Get ALL images in this folder
            images = list(class_folder.glob("*.png")) + list(class_folder.glob("*.jpg"))
            
            if not images:
                continue
            
            for img_path in images:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((32, 32))
                    img_array = np.array(img)
                    img_flat = img_array.flatten()
                    
                    X.append(img_flat)
                    y.append(class_id)
                    total_images += 1
                    
                except Exception as e:
                    continue
            
            class_counts[class_id] = len(images)
        
        if not X:
            print("❌ No training images loaded!")
            return None, None
        
        # Normalize to [0, 1] and convert to arrays
        X = np.array(X, dtype='float32') / 255.0
        y = np.array(y, dtype='int')
        
        print(f"\n✅ SUCCESSFULLY LOADED TRAINING DATA:")
        print(f"   Total images: {total_images:,}")
        print(f"   Classes: {len(class_counts)}")
        print(f"   Image shape: {X[0].shape} (32x32x3 = 3072 features)")
        
        return X, y
    
    @staticmethod
    def load_test_data():
        """Load test data - handles multiple possible folder structures"""
        print("\n📥 LOADING TEST DATA...")
        
        test_folder = Path('Test')
        
        if not test_folder.exists():
            print(f"❌ Test folder not found at: {test_folder.absolute()}")
            return None, None
        
        # Check for different possible structures
        print("  Checking test folder structure...")
        
        # Option 1: Test folder with CSV file (official GTSRB structure)
        csv_files = list(test_folder.glob("*.csv"))
        if csv_files:
            print(f"  Found CSV file: {csv_files[0].name}")
            result = FixedGTSRBLoader._load_test_from_csv(test_folder, csv_files[0])
            if result[0] is not None:
                return result
        
        # Option 2: Test folder with numbered subfolders (like Train folder)
        numbered_folders = [item for item in test_folder.iterdir() 
                           if item.is_dir() and item.name.isdigit()]
        if numbered_folders:
            print(f"  Found {len(numbered_folders)} numbered folders in Test/")
            return FixedGTSRBLoader._load_test_from_folders(test_folder)
        
        # Option 3: Test folder with images directly
        all_images = list(test_folder.rglob("*.png")) + list(test_folder.rglob("*.jpg"))
        if all_images:
            print(f"  Found {len(all_images)} images in Test/ folder")
            return FixedGTSRBLoader._load_test_images_directly(test_folder)
        
        print("❌ No test data found in Test/ folder")
        return None, None
    
    @staticmethod
    def _load_test_from_csv(test_folder, csv_file):
        """Load test data from CSV file (official GTSRB format)"""
        try:
            print(f"  Reading CSV file: {csv_file.name}")
            df = pd.read_csv(csv_file)
            print(f"  CSV has {len(df)} rows")
            
            # Check required columns
            if 'Path' not in df.columns or 'ClassId' not in df.columns:
                print("❌ CSV missing required columns (Path and ClassId)")
                return None, None
            
            X = []
            y = []
            loaded = 0
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading images"):
                try:
                    img_rel_path = row['Path']
                    class_id = row['ClassId']
                    
                    # Construct full path
                    img_path = test_folder / img_rel_path
                    
                    if img_path.exists():
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize((32, 32))
                        img_array = np.array(img)
                        img_flat = img_array.flatten()
                        
                        X.append(img_flat)
                        y.append(class_id)
                        loaded += 1
                    else:
                        # Try alternative path
                        alt_path = test_folder / Path(img_rel_path).name
                        if alt_path.exists():
                            img = Image.open(alt_path).convert('RGB')
                            img = img.resize((32, 32))
                            img_array = np.array(img)
                            img_flat = img_array.flatten()
                            
                            X.append(img_flat)
                            y.append(class_id)
                            loaded += 1
                            
                except Exception as e:
                    continue
            
            if not X:
                print("❌ No test images loaded from CSV")
                return None, None
            
            X = np.array(X, dtype='float32') / 255.0
            y = np.array(y, dtype='int')
            
            print(f"\n✅ Loaded {loaded} test images from CSV")
            print(f"   Classes in test set: {len(np.unique(y))}")
            
            return X, y
            
        except Exception as e:
            print(f"❌ Error loading from CSV: {e}")
            return None, None
    
    @staticmethod
    def _load_test_from_folders(test_folder):
        """Load test data from numbered folders (like Train structure)"""
        print("  Loading from numbered folders...")
        
        X = []
        y = []
        class_counts = {}
        total_images = 0
        
        # Get all numbered folders
        for item in test_folder.iterdir():
            if item.is_dir() and item.name.isdigit():
                try:
                    class_id = int(item.name)
                    if 0 <= class_id <= 42:
                        # Load images from this folder
                        images = list(item.glob("*.png")) + list(item.glob("*.jpg"))
                        
                        for img_path in images:
                            try:
                                img = Image.open(img_path).convert('RGB')
                                img = img.resize((32, 32))
                                img_array = np.array(img)
                                img_flat = img_array.flatten()
                                
                                X.append(img_flat)
                                y.append(class_id)
                                total_images += 1
                                
                            except Exception as e:
                                continue
                        
                        class_counts[class_id] = len(images)
                        
                except:
                    continue
        
        if not X:
            print("❌ No test images loaded from folders")
            return None, None
        
        X = np.array(X, dtype='float32') / 255.0
        y = np.array(y, dtype='int')
        
        print(f"\n✅ Loaded {total_images} test images from {len(class_counts)} folders")
        
        return X, y
    
    @staticmethod
    def _load_test_images_directly(test_folder):
        """Load test images directly from folder"""
        print("  Loading images directly...")
        
        X = []
        y = []
        loaded = 0
        
        # Get all images
        all_images = list(test_folder.rglob("*.png")) + list(test_folder.rglob("*.jpg"))
        
        for img_path in tqdm(all_images, desc="Loading images"):
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((32, 32))
                img_array = np.array(img)
                img_flat = img_array.flatten()
                
                X.append(img_flat)
                
                # Try to extract class from filename
                filename = img_path.stem
                class_id = 0  # Default
                
                # Try different filename patterns
                if '_' in filename:
                    # Pattern: 00000_00000.png
                    class_part = filename.split('_')[0]
                    if len(class_part) >= 5:
                        try:
                            class_id = int(class_part[:5])
                        except:
                            pass
                elif filename.isdigit():
                    # Pattern: 00000.png
                    try:
                        class_id = int(filename)
                    except:
                        pass
                
                # Validate class ID
                if 0 <= class_id <= 42:
                    y.append(class_id)
                    loaded += 1
                else:
                    y.append(0)
                    loaded += 1
                    
            except Exception as e:
                continue
        
        if not X:
            print("❌ No test images loaded")
            return None, None
        
        X = np.array(X, dtype='float32') / 255.0
        y = np.array(y, dtype='int')
        
        print(f"\n✅ Loaded {loaded} test images")
        
        return X, y
    
    @staticmethod
    def load_all_data_properly():
        """Load data with proper train/test separation"""
        print("\n" + "="*70)
        print("📊 LOADING GTSRB DATASET")
        print("="*70)
        
        # Load ALL training data
        print("\n" + "-"*50)
        X_train_full, y_train_full = FixedGTSRBLoader.load_train_data()
        
        if X_train_full is None:
            print("❌ Failed to load training data!")
            return False, None, None, None, None, None, None
        
        print(f"\n📊 TRAINING DATA SUMMARY:")
        print(f"   Total images: {X_train_full.shape[0]:,}")
        print(f"   Features per image: {X_train_full.shape[1]} (32x32x3 = 3072)")
        print(f"   Number of classes: {len(np.unique(y_train_full))}")
        
        # Load test data
        print("\n" + "-"*50)
        X_test, y_test = FixedGTSRBLoader.load_test_data()
        
        if X_test is None:
            print("\n⚠ WARNING: No test data could be loaded from Test/ folder")
            print("   Creating test set by splitting training data...")
            
            # Split training data 60% train, 40% test
            X_train_full, X_test, y_train_full, y_test = train_test_split(
                X_train_full, y_train_full, 
                test_size=0.4, 
                random_state=42, 
                stratify=y_train_full
            )
            print(f"   Created test set: {X_test.shape[0]:,} samples")
        else:
            print(f"\n📊 TEST DATA SUMMARY:")
            print(f"   Total images: {X_test.shape[0]:,}")
            print(f"   Classes: {len(np.unique(y_test))}")
        
        # Split remaining training data into train and validation
        print("\n📊 SPLITTING DATA...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, 
            test_size=0.2,  # 20% validation from remaining 60%
            random_state=42, 
            stratify=y_train_full
        )
        
        print(f"\n✅ FINAL DATASET SPLIT:")
        print(f"   Training set:    {X_train.shape[0]:,} images")
        print(f"   Validation set:  {X_val.shape[0]:,} images")
        print(f"   Test set:        {X_test.shape[0]:,} images")
        print(f"   Total images:    {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]:,}")
        
        return True, X_train, X_val, X_test, y_train, y_val, y_test


# ---------------- IMPROVED RBF NETWORK (Keep your original) ----------------
class ImprovedRBFNetwork:
    def __init__(self, n_centers_per_class=10, sigma_method='classwise', lambda_reg=1e-6):
        """
        Improved RBF Network with better initialization and training
        """
        self.n_centers_per_class = n_centers_per_class
        self.sigma_method = sigma_method
        self.lambda_reg = lambda_reg
        self.centers = None
        self.sigmas = None
        self.W = None
        self.classes_ = None
        self.train_time = 0
        
    def _rbf_kernel(self, X, centers, sigmas):
        """Compute RBF kernel matrix with per-center sigmas"""
        n_samples = X.shape[0]
        n_centers = centers.shape[0]
        
        # Efficient computation using broadcasting
        X_expanded = X[:, np.newaxis, :]
        centers_expanded = centers[np.newaxis, :, :]
        
        distances = np.sum((X_expanded - centers_expanded) ** 2, axis=2)
        
        # Apply RBF with appropriate sigmas
        if isinstance(sigmas, (int, float)):
            return np.exp(-distances / (2 * sigmas ** 2))
        else:
            sigmas_expanded = sigmas[np.newaxis, :]
            return np.exp(-distances / (2 * sigmas_expanded ** 2))
    
    def _calculate_sigmas(self, centers, X_train=None, y_train=None):
        """Calculate sigma values for RBF centers"""
        if self.sigma_method == 'global':
            # Global sigma based on average distance between centers
            distances = cdist(centers, centers)
            np.fill_diagonal(distances, np.inf)
            min_distances = np.min(distances, axis=1)
            sigma = np.mean(min_distances) / np.sqrt(2)
            return sigma
        elif self.sigma_method == 'classwise':
            # Different sigma for each class's centers
            sigmas = []
            start_idx = 0
            for class_id in self.classes_:
                end_idx = start_idx + self.n_centers_per_class
                class_centers = centers[start_idx:end_idx]
                
                if len(class_centers) > 1:
                    distances = cdist(class_centers, class_centers)
                    np.fill_diagonal(distances, np.inf)
                    min_distances = np.min(distances, axis=1)
                    sigma_class = np.mean(min_distances) / np.sqrt(2)
                else:
                    sigma_class = 1.0
                
                sigmas.extend([sigma_class] * len(class_centers))
                start_idx = end_idx
            return np.array(sigmas)
        else:
            return 1.0
    
    def fit(self, X_train, y_train):
        """Train the improved RBF network"""
        print(f"\n🤖 TRAINING IMPROVED RBF NETWORK")
        print(f"   Centers per class: {self.n_centers_per_class}")
        print(f"   Sigma method: {self.sigma_method}")
        
        start_time = time.time()
        
        # Get unique classes
        self.classes_ = np.unique(y_train)
        n_classes = len(self.classes_)
        
        # Step 1: Initialize centers using K-means PER CLASS
        print("   Step 1: Initializing centers per class...")
        all_centers = []
        
        for i, class_id in enumerate(self.classes_):
            class_samples = X_train[y_train == class_id]
            
            if len(class_samples) < self.n_centers_per_class:
                centers_class = class_samples
            else:
                n_centers = min(self.n_centers_per_class, len(class_samples))
                kmeans = KMeans(n_clusters=n_centers, random_state=42+i, n_init=10)
                kmeans.fit(class_samples)
                centers_class = kmeans.cluster_centers_
            
            all_centers.append(centers_class)
        
        self.centers = np.vstack(all_centers)
        print(f"   Total RBF centers: {self.centers.shape[0]}")
        
        # Step 2: Calculate sigmas
        print("   Step 2: Calculating sigmas...")
        self.sigmas = self._calculate_sigmas(self.centers, X_train, y_train)
        
        if isinstance(self.sigmas, (int, float)):
            print(f"   Global sigma: {self.sigmas:.4f}")
        else:
            print(f"   Sigma range: [{self.sigmas.min():.4f}, {self.sigmas.max():.4f}]")
        
        # Step 3: Compute RBF activations
        print("   Step 3: Computing RBF activations...")
        Phi = self._rbf_kernel(X_train, self.centers, self.sigmas)
        
        # Add bias term
        Phi = np.hstack([Phi, np.ones((Phi.shape[0], 1))])
        
        # Step 4: Prepare target matrix
        T = np.zeros((len(y_train), n_classes))
        for i, class_id in enumerate(self.classes_):
            T[y_train == class_id, i] = 1
        
        # Step 5: Compute output weights with regularization
        print("   Step 4: Computing output weights...")
        PhiT_Phi = Phi.T @ Phi
        n_features = PhiT_Phi.shape[0]
        
        reg_matrix = self.lambda_reg * np.eye(n_features)
        reg_matrix[-1, -1] = 0
        
        try:
            self.W = np.linalg.pinv(PhiT_Phi + reg_matrix) @ Phi.T @ T
        except np.linalg.LinAlgError:
            self.W = np.linalg.lstsq(PhiT_Phi + reg_matrix, Phi.T @ T, rcond=None)[0]
        
        self.train_time = time.time() - start_time
        print(f"   Training completed in {self.train_time:.2f} seconds")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.centers is None:
            raise ValueError("Model must be trained before prediction")
        
        Phi = self._rbf_kernel(X, self.centers, self.sigmas)
        Phi = np.hstack([Phi, np.ones((Phi.shape[0], 1))])
        
        scores = Phi @ self.W
        predictions = np.argmax(scores, axis=1)
        return self.classes_[predictions]
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.centers is None:
            raise ValueError("Model must be trained before prediction")
        
        Phi = self._rbf_kernel(X, self.centers, self.sigmas)
        Phi = np.hstack([Phi, np.ones((Phi.shape[0], 1))])
        
        scores = Phi @ self.W
        # Softmax to get probabilities
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def get_rbf_activations(self, X):
        """Get RBF activations for visualization"""
        if self.centers is None:
            raise ValueError("Model must be trained before prediction")
        
        Phi = self._rbf_kernel(X, self.centers, self.sigmas)
        return Phi


# ---------------- NEW VISUALIZATION: RBF CENTERS COMPARISON ----------------
def plot_rbf_centers_comparison(X_train_pca, y_train, X_val_pca, y_val, centers_range=None):
    """
    Create visual comparison of RBF model accuracy for different numbers of centers
    """
    print("\n" + "="*70)
    print("📊 VISUAL COMPARISON: RBF ACCURACY VS NUMBER OF CENTERS")
    print("="*70)
    
    if centers_range is None:
        # Test different numbers of centers per class
        centers_range = [1, 2, 3, 5, 8, 10, 12, 15]
    
    train_accuracies = []
    val_accuracies = []
    training_times = []
    total_centers_list = []
    
    n_classes = len(np.unique(y_train))
    
    print(f"\nTesting {len(centers_range)} different center configurations...")
    print(f"Number of classes: {n_classes}")
    print("-"*60)
    
    for n_centers_per_class in centers_range:
        print(f"\nTesting {n_centers_per_class} centers per class...")
        
        # Create and train RBF model
        rbf = ImprovedRBFNetwork(
            n_centers_per_class=n_centers_per_class,
            sigma_method='classwise',
            lambda_reg=1e-6
        )
        
        start_time = time.time()
        rbf.fit(X_train_pca, y_train)
        train_time = time.time() - start_time
        
        # Make predictions
        y_train_pred = rbf.predict(X_train_pca)
        y_val_pred = rbf.predict(X_val_pca)
        
        # Calculate accuracies
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        # Store results
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        training_times.append(train_time)
        total_centers = n_centers_per_class * n_classes
        total_centers_list.append(total_centers)
        
        print(f"  Total centers: {total_centers}")
        print(f"  Training accuracy: {train_acc:.3%}")
        print(f"  Validation accuracy: {val_acc:.3%}")
        print(f"  Training time: {train_time:.2f}s")
    
    # Create visualization figure
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Accuracy vs Number of Centers
    plt.subplot(2, 2, 1)
    plt.plot(total_centers_list, train_accuracies, 'b-o', linewidth=2, markersize=8, label='Training Accuracy')
    plt.plot(total_centers_list, val_accuracies, 'r-s', linewidth=2, markersize=8, label='Validation Accuracy')
    plt.xlabel('Total Number of RBF Centers', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('RBF Network Accuracy vs Number of Centers', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add value labels to points
    for i, (total_centers, train_acc, val_acc) in enumerate(zip(total_centers_list, train_accuracies, val_accuracies)):
        plt.text(total_centers, train_acc + 0.01, f'{train_acc:.3f}', ha='center', fontsize=9, color='blue')
        plt.text(total_centers, val_acc - 0.01, f'{val_acc:.3f}', ha='center', fontsize=9, color='red')
    
    # Plot 2: Training Time vs Number of Centers
    plt.subplot(2, 2, 2)
    plt.plot(total_centers_list, training_times, 'g-^', linewidth=2, markersize=8)
    plt.xlabel('Total Number of RBF Centers', fontsize=12)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.title('Training Time vs Number of Centers', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value labels to points
    for i, (total_centers, train_time) in enumerate(zip(total_centers_list, training_times)):
        plt.text(total_centers, train_time + 0.5, f'{train_time:.1f}s', ha='center', fontsize=9, color='green')
    
    # Plot 3: Accuracy Difference (Training - Validation)
    plt.subplot(2, 2, 3)
    accuracy_diff = [train - val for train, val in zip(train_accuracies, val_accuracies)]
    bars = plt.bar(range(len(centers_range)), accuracy_diff, color='orange', alpha=0.7)
    plt.xlabel('Centers per Class', fontsize=12)
    plt.ylabel('Accuracy Difference (Train - Val)', fontsize=12)
    plt.title('Overfitting Analysis: Training vs Validation Gap', fontsize=14, fontweight='bold')
    plt.xticks(range(len(centers_range)), centers_range)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels to bars
    for bar, diff in zip(bars, accuracy_diff):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{diff:.3f}', ha='center', fontsize=9)
    
    # Plot 4: Optimal Center Selection
    plt.subplot(2, 2, 4)
    
    # Find optimal based on validation accuracy
    best_val_idx = np.argmax(val_accuracies)
    best_val_centers = centers_range[best_val_idx]
    best_val_accuracy = val_accuracies[best_val_idx]
    
    # Find optimal based on efficiency (accuracy per center)
    efficiency = [val_acc / centers for val_acc, centers in zip(val_accuracies, centers_range)]
    best_eff_idx = np.argmax(efficiency)
    best_eff_centers = centers_range[best_eff_idx]
    best_eff_accuracy = val_accuracies[best_eff_idx]
    
    # Create summary table
    summary_data = [
        ['Best Validation', f'{best_val_centers}', f'{best_val_accuracy:.3%}'],
        ['Most Efficient', f'{best_eff_centers}', f'{best_eff_accuracy:.3%}'],
        ['Total Classes', f'{n_classes}', '-'],
        ['Max Centers Tested', f'{max(centers_range)}', f'{total_centers_list[-1]} total']
    ]
    
    table = plt.table(cellText=summary_data,
                     colLabels=['Metric', 'Centers/Class', 'Value'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.35, 0.3, 0.35])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    plt.axis('off')
    plt.title('Optimal Center Selection Summary', fontsize=14, fontweight='bold', pad=20)
    
    # Overall figure title and layout
    plt.suptitle('RBF Network: Impact of Number of Centers on Performance', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('rbf_centers_comparison.png', dpi=120, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ RBF CENTERS COMPARISON COMPLETE")
    print(f"   Best validation accuracy: {best_val_accuracy:.3%} with {best_val_centers} centers/class")
    print(f"   Most efficient: {best_eff_accuracy:.3%} with {best_eff_centers} centers/class")
    print(f"   Visualization saved as: rbf_centers_comparison.png")
    
    return {
        'centers_per_class': centers_range,
        'total_centers': total_centers_list,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'training_times': training_times,
        'best_val_config': (best_val_centers, best_val_accuracy),
        'best_eff_config': (best_eff_centers, best_eff_accuracy)
    }


# ---------------- SIMPLIFIED VISUALIZATION FUNCTIONS ----------------
def visualize_rbf_centers(rbf_model, X_train_pca, y_train):
    """Visualize RBF centers in feature space with classes"""
    print("\n📊 Visualizing RBF centers in feature space...")
    
    # Get the first two principal components for visualization
    if X_train_pca.shape[1] > 2:
        pca_2d = PCA(n_components=2, random_state=42)
        X_2d = pca_2d.fit_transform(X_train_pca)
        centers_2d = pca_2d.transform(rbf_model.centers)
    else:
        X_2d = X_train_pca
        centers_2d = rbf_model.centers
    
    # Create a scatter plot
    plt.figure(figsize=(12, 10))
    
    # Plot training data points
    unique_classes = np.unique(y_train)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))
    
    for i, class_id in enumerate(unique_classes[:15]):  # Show first 15 classes
        class_mask = y_train == class_id
        plt.scatter(X_2d[class_mask, 0], X_2d[class_mask, 1], 
                   c=[colors[i]], alpha=0.3, s=10, 
                   label=f'Class {class_id}')
    
    # Plot RBF centers
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
               c='red', marker='X', s=100, 
               label='RBF Centers', edgecolors='black', linewidths=1)
    
    # Draw circles around centers with sigma radius
    if isinstance(rbf_model.sigmas, (int, float)):
        sigma = rbf_model.sigmas
        for center in centers_2d[:10]:  # Show circles for first 10 centers
            circle = plt.Circle((center[0], center[1]), sigma, 
                              color='red', fill=False, alpha=0.3, linestyle='--')
            plt.gca().add_patch(circle)
    else:
        avg_sigma = np.mean(rbf_model.sigmas)
        for center in centers_2d[:10]:
            circle = plt.Circle((center[0], center[1]), avg_sigma, 
                              color='red', fill=False, alpha=0.3, linestyle='--')
            plt.gca().add_patch(circle)
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('RBF Centers in Feature Space\n(Circles show RBF influence regions)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rbf_centers_visualization.png', dpi=120, bbox_inches='tight')
    plt.show()
    
    return X_2d, centers_2d

def plot_pca_analysis(pca, reduced_features, variance_preserved):
    """Plot PCA analysis"""
    print("\n📉 Creating PCA analysis plot...")
    
    plt.figure(figsize=(10, 5))
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, 'b-', linewidth=2)
    plt.axhline(y=90, color='r', linestyle='--', label='90% Variance')
    plt.axvline(x=reduced_features, color='g', linestyle=':', label=f'{reduced_features} components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance (%)')
    plt.title('PCA Analysis - GTSRB Dataset')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pca_analysis.png', dpi=100)
    plt.show()

def plot_rbf_predictions_examples(X_test_original, y_test, y_pred, n_examples=20):
    """Plot 20 examples of RBF predictions"""
    print(f"\n📸 Plotting {n_examples} RBF prediction examples...")
    
    # Get random sample indices
    sample_indices = np.random.choice(len(X_test_original), min(n_examples, len(X_test_original)), replace=False)
    
    # Create figure
    fig, axes = plt.subplots(4, 5, figsize=(18, 15))
    fig.suptitle('RBF Network Prediction Examples\n(Green=Correct, Red=Incorrect)', fontsize=16, y=1.02)
    
    for idx, ax in enumerate(axes.flatten()):
        if idx < len(sample_indices):
            sample_idx = sample_indices[idx]
            img = X_test_original[sample_idx].reshape(32, 32, 3)
            ax.imshow(img)
            
            true_label = y_test[sample_idx]
            pred_label = y_pred[sample_idx]
            
            # Get class names
            true_name = TRAFFIC_SIGN_NAMES_SHORT.get(true_label, f"Class {true_label}")
            pred_name = TRAFFIC_SIGN_NAMES_SHORT.get(pred_label, f"Class {pred_label}")
            
            # Truncate long names
            if len(true_name) > 15:
                true_name = true_name[:12] + "..."
            if len(pred_name) > 15:
                pred_name = pred_name[:12] + "..."
            
            # Determine color
            is_correct = true_label == pred_label
            color = 'green' if is_correct else 'red'
            
            # Set title with prediction info
            ax.set_title(f"Image {idx+1}\nTrue: {true_label} ({true_name})\nPred: {pred_label} ({pred_name})", 
                        color=color, fontsize=9)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('rbf_prediction_examples.png', dpi=120, bbox_inches='tight')
    plt.show()

def plot_model_comparison(rbf_acc, knn1_acc, knn3_acc, ncc_acc):
    """Plot model comparison bar chart"""
    print("\n📊 Creating model comparison chart...")
    
    plt.figure(figsize=(10, 6))
    
    models = ['RBF Network', '1-NN', '3-NN', 'Nearest Centroid']
    accuracies = [rbf_acc, knn1_acc, knn3_acc, ncc_acc]
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon']
    
    bars = plt.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
    plt.xlabel('Model')
    plt.ylabel('Test Accuracy')
    plt.title('Model Accuracy Comparison on GTSRB Dataset')
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=100, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title="RBF Network Confusion Matrix"):
    """Plot simplified confusion matrix"""
    print("\n📊 Creating confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    
    # Add text annotations for major confusions only
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > thresh:  # Only show large values
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=6)
    
    plt.title(f'{title}\nAccuracy: {accuracy_score(y_true, y_pred):.2%}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.show()


# ---------------- MAIN PIPELINE ----------------
def run_full_rbf_pipeline():
    """Run the complete RBF network pipeline with full dataset"""
    print("\n" + "="*80)
    print("🚀 STARTING FULL DATASET RBF NETWORK PIPELINE")
    print("="*80)
    
    # Step 1: Load ALL data (no limits) - USING FIXED LOADER
    print("\n📊 STEP 1: LOADING COMPLETE DATASET")
    print("-"*50)
    
    start_load = time.time()
    success, X_train, X_val, X_test, y_train, y_val, y_test = FixedGTSRBLoader.load_all_data_properly()
    load_time = time.time() - start_load
    
    if not success:
        print("❌ Cannot continue without data")
        return None, None, None
    
    print(f"\n⏱️ Data loading time: {load_time:.1f} seconds")
    
    # Step 2: Preprocessing
    print("\n📊 STEP 2: PREPROCESSING")
    print("-"*50)
    
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("Applying PCA with >90% variance preservation...")
    pca = PCA(n_components=0.90, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    variance_preserved = pca.explained_variance_ratio_.sum()
    original_features = X_train.shape[1]
    reduced_features = X_train_pca.shape[1]
    
    print(f"\n✅ PCA RESULTS:")
    print(f"   Original features: {original_features}")
    print(f"   Features after PCA: {reduced_features}")
    print(f"   Variance preserved: {variance_preserved:.2%}")
    print(f"   Data reduction: {(1 - reduced_features/original_features):.1%}")
    
    # Plot PCA analysis
    plot_pca_analysis(pca, reduced_features, variance_preserved)
    
    # Step 3: RBF Centers Comparison - YOUR SPECIFIED RANGE
    print("\n📊 STEP 3: RBF CENTERS COMPARISON ANALYSIS")
    print("-"*50)
    
    # Run comparison of different center configurations
    centers_comparison = plot_rbf_centers_comparison(
        X_train_pca, y_train, X_val_pca, y_val,
        centers_range=[1, 2, 3, 5, 8, 10, 12, 15]  # YOUR SPECIFIED RANGE
    )
    
    # Get optimal configuration from comparison
    best_centers, best_val_acc = centers_comparison['best_val_config']
    print(f"\n🔧 Using optimal configuration: {best_centers} centers per class")
    
    # Step 4: Train final RBF network with optimal centers
    print("\n📊 STEP 4: TRAINING FINAL RBF NETWORK")
    print("-"*50)
    
    rbf = ImprovedRBFNetwork(
        n_centers_per_class=best_centers,
        sigma_method='classwise',
        lambda_reg=1e-6
    )
    
    rbf.fit(X_train_pca, y_train)
    
    # Make predictions
    y_train_pred = rbf.predict(X_train_pca)
    y_val_pred = rbf.predict(X_val_pca)
    y_test_pred = rbf.predict(X_test_pca)
    
    # Calculate accuracies
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\n📊 FINAL RBF NETWORK RESULTS:")
    print(f"   Training accuracy:   {train_acc:.3%}")
    print(f"   Validation accuracy: {val_acc:.3%}")
    print(f"   Test accuracy:       {test_acc:.3%}")  # Should be ~46% with 10 centers!
    print(f"   Training time:       {rbf.train_time:.2f} seconds")
    print(f"   Total centers used:  {rbf.centers.shape[0]}")
    
    # Step 5: Train comparison models
    print("\n📊 STEP 5: TRAINING COMPARISON MODELS")
    print("-"*50)
    
    comparison_results = {}
    
    # 1-NN
    print("\n🔧 Training 1-NN...")
    knn1 = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    knn1.fit(X_train_pca, y_train)
    knn1_acc = accuracy_score(y_test, knn1.predict(X_test_pca))
    comparison_results['1-NN'] = knn1_acc
    print(f"   1-NN accuracy: {knn1_acc:.3%}")
    
    # 3-NN
    print("\n🔧 Training 3-NN...")
    knn3 = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    knn3.fit(X_train_pca, y_train)
    knn3_acc = accuracy_score(y_test, knn3.predict(X_test_pca))
    comparison_results['3-NN'] = knn3_acc
    print(f"   3-NN accuracy: {knn3_acc:.3%}")
    
    # Nearest Class Centroid
    print("\n🔧 Training Nearest Class Centroid...")
    ncc = NearestCentroid()
    ncc.fit(X_train_pca, y_train)
    ncc_acc = accuracy_score(y_test, ncc.predict(X_test_pca))
    comparison_results['NCC'] = ncc_acc
    print(f"   NCC accuracy: {ncc_acc:.3%}")
    
    # Step 6: Create visualizations
    print("\n📊 STEP 6: CREATING VISUALIZATIONS")
    print("-"*50)
    
    # 1. RBF Centers visualization
    print("1. Visualizing RBF centers...")
    visualize_rbf_centers(rbf, X_train_pca, y_train)
    
    # 2. Model comparison
    print("2. Creating model comparison...")
    plot_model_comparison(test_acc, knn1_acc, knn3_acc, ncc_acc)
    
    # 3. Confusion matrix
    print("3. Creating confusion matrix...")
    plot_confusion_matrix(y_test, y_test_pred)
    
    # 4. Prediction examples
    print("4. Plotting prediction examples...")
    plot_rbf_predictions_examples(X_test, y_test, y_test_pred, n_examples=20)
    
    # Step 7: Performance summary
    print("\n📊 STEP 7: PERFORMANCE SUMMARY")
    print("-"*50)
    
    print(f"\n🏆 MODEL PERFORMANCE COMPARISON:")
    print("-"*40)
    print(f"{'Model':<20} {'Test Accuracy':<15}")
    print("-"*40)
    print(f"{'RBF Network':<20} {test_acc:.3%}")
    print(f"{'1-Nearest Neighbor':<20} {knn1_acc:.3%}")
    print(f"{'3-Nearest Neighbors':<20} {knn3_acc:.3%}")
    print(f"{'Nearest Centroid':<20} {ncc_acc:.3%}")
    
    # Calculate improvement
    improvement_over_1nn = ((test_acc - knn1_acc) / knn1_acc * 100) if knn1_acc > 0 else 0
    improvement_over_3nn = ((test_acc - knn3_acc) / knn3_acc * 100) if knn3_acc > 0 else 0
    improvement_over_ncc = ((test_acc - ncc_acc) / ncc_acc * 100) if ncc_acc > 0 else 0
    
    print(f"\n📈 RBF IMPROVEMENT OVER:")
    print(f"   1-NN: {improvement_over_1nn:+.1f}%")
    print(f"   3-NN: {improvement_over_3nn:+.1f}%")
    print(f"   NCC:  {improvement_over_ncc:+.1f}%")
    
    # RBF network statistics
    print(f"\n🔧 RBF NETWORK STATISTICS:")
    print(f"   Number of centers: {rbf.centers.shape[0]}")
    print(f"   Centers per class: {rbf.n_centers_per_class}")
    print(f"   Sigma method: {rbf.sigma_method}")
    if isinstance(rbf.sigmas, (int, float)):
        print(f"   Sigma value: {rbf.sigmas:.4f}")
    else:
        print(f"   Sigma range: {rbf.sigmas.min():.4f} to {rbf.sigmas.max():.4f}")
    print(f"   Training time: {rbf.train_time:.2f} seconds")
    
    # Class-wise performance
    print(f"\n📊 CLASS-WISE PERFORMANCE (Top 5 best/worst):")
    cm = confusion_matrix(y_test, y_test_pred)
    class_accuracies = np.diag(cm) / np.sum(cm, axis=1)
    
    # Sort classes by accuracy
    sorted_indices = np.argsort(class_accuracies)
    
    print(f"\n   Best classes:")
    for i in sorted_indices[-5:][::-1]:
        class_name = TRAFFIC_SIGN_NAMES_SHORT.get(i, f"Class {i}")
        if len(class_name) > 20:
            class_name = class_name[:17] + "..."
        print(f"     Class {i:2d} ({class_name:20}): {class_accuracies[i]:.1%}")
    
    print(f"\n   Worst classes:")
    for i in sorted_indices[:5]:
        class_name = TRAFFIC_SIGN_NAMES_SHORT.get(i, f"Class {i}")
        if len(class_name) > 20:
            class_name = class_name[:17] + "..."
        print(f"     Class {i:2d} ({class_name:20}): {class_accuracies[i]:.1%}")
    
    print(f"\n📁 GENERATED VISUALIZATIONS:")
    print("   1. pca_analysis.png              - PCA variance analysis")
    print("   2. rbf_centers_comparison.png    - RBF centers comparison (NEW)")
    print("   3. rbf_centers_visualization.png - RBF centers in feature space")
    print("   4. model_comparison.png          - Model accuracy comparison")
    print("   5. confusion_matrix.png          - Confusion matrix")
    print("   6. rbf_prediction_examples.png   - 20 prediction examples")
    
    return rbf, test_acc, comparison_results, centers_comparison


# ---------------- MAIN EXECUTION ----------------
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 GTSRB TRAFFIC SIGN CLASSIFICATION WITH RBF NETWORKS")
    print("="*80)
    
    print("\n⚡ FIXED VERSION FEATURES:")
    print("   • Uses proper data loading with multiple fallback options")
    print("   • Visual comparison of RBF accuracy for different centers")
    print("   • Automatically selects optimal number of centers")
    print("   • Comparison with 1-NN, 3-NN, and Nearest Centroid")
    print("   • All standard visualizations")
    
    print("\n📊 TESTING CONFIGURATION:")
    print("   • Centers per class to test: [1, 2, 3, 5, 8, 10, 12, 15]")
    print("   • PCA variance preservation: 90%")
    print("   • Sigma method: classwise")
    
    print("\n⚠ NOTE: If Test/ folder is missing, training data will be split")
    print("   Training: 60%, Validation: 20%, Test: 20%")
    
    try:
        rbf_model, rbf_accuracy, comparison_results, centers_comparison = run_full_rbf_pipeline()
        
        if rbf_model is not None:
            print("\n" + "="*80)
            print("💡 RECOMMENDATIONS:")
            print("="*80)
            print(f"   1. Optimal centers per class: {rbf_model.n_centers_per_class}")
            print(f"   2. Total RBF centers used: {rbf_model.centers.shape[0]}")
            print(f"   3. Test accuracy achieved: {rbf_accuracy:.2%}")
            print(f"   4. Consider PCA variance of 95% for potentially better results")
            print(f"   5. Experiment with sigma_method='global' for comparison")
            
            # Show center comparison insights
            print(f"\n📈 CENTERS COMPARISON INSIGHTS:")
            best_val = centers_comparison['best_val_config']
            best_eff = centers_comparison['best_eff_config']
            print(f"   Best accuracy: {best_val[1]:.2%} with {best_val[0]} centers/class")
            print(f"   Most efficient: {best_eff[1]:.2%} with {best_eff[0]} centers/class")
            print(f"   Expected test accuracy: {best_val[1]:.2%} ± 2%")
            
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {str(e)}")
        print("\n💡 TROUBLESHOOTING:")
        print("   1. Check that Train/ folder exists with numbered subfolders (0/, 1/, etc.)")
        print("   2. Test/ folder is optional but helps for better evaluation")
        print("   3. Ensure all required packages are installed")
        print("   4. If memory error occurs, reduce dataset size in load_train_data()")
    
    print("\n" + "="*80)
    print("🏁 PROGRAM FINISHED - CHECK GENERATED VISUALIZATIONS")
    print("="*80)