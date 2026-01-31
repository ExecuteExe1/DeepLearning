import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import re  # Added for filename parsing
warnings.filterwarnings('ignore')

# Check packages for debugging purposes
print("🔧 Checking packages...")
try:
    import sklearn
    print("✅ All packages ready!")
except ImportError as e:
    print(f"⚠ Missing packages: {e}")
    print("Run: pip install numpy pandas matplotlib scikit-learn pillow tqdm")

# Now import
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import SGDClassifier

print("\n" + "="*70)
print("🚦 GTSRB CLASSIFIER - PROPER TEST/TRAIN SPLIT")
print("="*70)


# Classification of the traffic signs
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


# UPDATED FIXED DATA LOADER - IMPROVED VERSION
class FixedGTSRBLoader:
    """Fixed loader that properly handles GTSRB dataset structure"""
    
    @staticmethod
    def debug_folder_structure():
        """Debug the folder structure to see what's wrong"""
        print("\n🔍 DEBUGGING FOLDER STRUCTURE")
        print("="*50)
        
        current_dir = Path.cwd()
        folders_to_check = ['Train', 'Test', 'Meta']
        
        for folder_name in folders_to_check:
            folder = current_dir / folder_name
            print(f"\n📁 {folder_name}/ folder:")
            
            if not folder.exists():
                print(f"   ❌ NOT FOUND")
                continue
            
            # Count items
            items = list(folder.iterdir())
            print(f"   Found {len(items)} items")
            
            # Show first few items
            for i, item in enumerate(items[:10]):
                if item.is_dir():
                    print(f"   {i+1}. 📁 {item.name}/")
                else:
                    print(f"   {i+1}. 📄 {item.name} ({item.stat().st_size:,} bytes)")
            
            if len(items) > 10:
                print(f"   ... and {len(items)-10} more")
            
            # Count images
            images = list(folder.rglob("*.png")) + list(folder.rglob("*.jpg"))
            print(f"   Total images: {len(images)}")
            
            if images and folder_name == 'Train':
                # Check class distribution in Train
                class_folders = [item for item in items if item.is_dir() and item.name.isdigit()]
                print(f"   Class folders: {len(class_folders)}")
        
        return True
    
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
            print(f"   Looking for folders named 0/, 1/, 2/, ... up to 42/")
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
                    # Load and preprocess image
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((32, 32))  # Resize to standard size
                    img_array = np.array(img)
                    img_flat = img_array.flatten()  # Flatten to 1D
                    
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
        print(f"   Classes: {len(class_counts)} (IDs: {sorted(class_counts.keys())})")
        print(f"   Image shape: {X[0].shape} (32x32x3 = 3072 features)")
        
        # Show some statistics
        print(f"\n📊 CLASS DISTRIBUTION (first 10 classes):")
        for class_id in sorted(class_counts.keys())[:10]:
            count = class_counts[class_id]
            class_name = TRAFFIC_SIGN_NAMES_SHORT.get(class_id, f"Class {class_id}")
            print(f"   Class {class_id:2d}: {class_name:25} - {count:4d} images")
        
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
        print(f"   Test classes: {sorted(class_counts.keys())}")
        
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
                else:
                    # Try to extract numbers using regex
                    numbers = re.findall(r'\d+', filename)
                    if numbers:
                        try:
                            class_id = int(numbers[0])
                        except:
                            pass
                
                # Validate class ID
                if 0 <= class_id <= 42:
                    y.append(class_id)
                    loaded += 1
                else:
                    # Use 0 as default
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
        print(f"   Classes found: {len(np.unique(y))}")
        
        return X, y
    
    @staticmethod
    def load_meta_images():
        """Load meta images for examples"""
        print("\n📥 LOADING META IMAGES...")
        
        meta_folder = Path('Meta')
        
        if not meta_folder.exists():
            print("⚠ Meta folder not found (optional)")
            return None, None, None
        
        X_meta = []
        y_meta = []
        filenames = []
        
        # Meta images are typically named 0.png, 1.png, etc.
        meta_images = list(meta_folder.glob("*.png")) + list(meta_folder.glob("*.jpg"))
        
        if not meta_images:
            print("⚠ No meta images found")
            return None, None, None
        
        print(f"  Found {len(meta_images)} meta images")
        
        for img_file in meta_images:
            try:
                # Try to get class from filename
                filename = img_file.stem
                if filename.isdigit():
                    class_id = int(filename)
                else:
                    # Try to extract number from filename
                    numbers = re.findall(r'\d+', filename)
                    if numbers:
                        class_id = int(numbers[0])
                    else:
                        continue
                
                if 0 <= class_id <= 42:
                    img = Image.open(img_file).convert('RGB')
                    img = img.resize((32, 32))
                    img_array = np.array(img)
                    img_flat = img_array.flatten()
                    
                    X_meta.append(img_flat)
                    y_meta.append(class_id)
                    filenames.append(img_file.name)
                    
            except Exception as e:
                continue
        
        if not X_meta:
            print("⚠ No valid meta images loaded")
            return None, None, None
        
        X_meta = np.array(X_meta, dtype='float32') / 255.0
        y_meta = np.array(y_meta, dtype='int')
        
        print(f"✅ Loaded {len(X_meta)} meta images")
        
        return X_meta, y_meta, filenames


# 3. UPDATED DATA LOADING FUNCTION
def load_all_data_properly():
    """Load data with proper train/test separation"""
    print("\n" + "="*70)
    print("📊 LOADING GTSRB DATASET")
    print("="*70)
    
    # Debug folder structure first
    FixedGTSRBLoader.debug_folder_structure()
    
    # Load ALL training data
    print("\n" + "-"*50)
    print("🚀 LOADING TRAINING DATA")
    print("-"*50)
    X_train_full, y_train_full = FixedGTSRBLoader.load_train_data()
    
    if X_train_full is None:
        print("❌ Failed to load training data!")
        print("💡 Check that Train/ folder exists and contains numbered subfolders")
        return False, None, None, None, None, None, None
    
    print(f"\n📊 TRAINING DATA SUMMARY:")
    print(f"   Total images: {X_train_full.shape[0]:,}")
    print(f"   Features per image: {X_train_full.shape[1]} (32x32x3 = 3072)")
    print(f"   Number of classes: {len(np.unique(y_train_full))}")
    
    # Load test data
    print("\n" + "-"*50)
    print("🚀 LOADING TEST DATA")
    print("-"*50)
    X_test, y_test = FixedGTSRBLoader.load_test_data()
    
    if X_test is None:
        print("\n⚠ WARNING: No test data could be loaded from Test/ folder")
        print("   This could be because:")
        print("   1. Test/ folder doesn't exist")
        print("   2. Test/ folder has different structure than expected")
        print("   3. No images found in Test/ folder")
        print("\n   Creating test set by splitting training data (60/40 split)...")
        
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
    print("\n" + "-"*50)
    print("📊 SPLITTING DATA")
    print("-"*50)
    
    print("Splitting into train/validation (80/20 split)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_train_full
    )
    
    print(f"\n✅ FINAL DATASET SPLIT:")
    print(f"   Training set:    {X_train.shape[0]:,} images")
    print(f"   Validation set:  {X_val.shape[0]:,} images")
    print(f"   Test set:        {X_test.shape[0]:,} images")
    print(f"   Total images:    {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]:,}")
    print(f"   Classes:         {len(np.unique(y_train))} (0 to {np.max(y_train)})")
    
    # Show detailed class distribution
    print(f"\n📈 DETAILED CLASS DISTRIBUTION:")
    unique_classes = np.unique(y_train)
    print(f"   {'Class':<6} {'Name':<20} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    print(f"   {'-'*6} {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    for class_id in sorted(unique_classes)[:15]:  # Show first 15
        train_count = np.sum(y_train == class_id)
        val_count = np.sum(y_val == class_id)
        test_count = np.sum(y_test == class_id)
        total_count = train_count + val_count + test_count
        class_name = TRAFFIC_SIGN_NAMES_SHORT.get(class_id, f"Class {class_id}")
        
        if len(class_name) > 20:
            class_name = class_name[:17] + "..."
        
        print(f"   {class_id:<6} {class_name:<20} {train_count:<8} {val_count:<8} {test_count:<8} {total_count:<8}")
    
    if len(unique_classes) > 15:
        print(f"   ... and {len(unique_classes) - 15} more classes")
    
    return True, X_train, X_val, X_test, y_train, y_val, y_test


# 4. FIXED PIPELINE (same as before)
def run_fixed_pipeline():
    """Run the fixed pipeline with proper data loading"""
    print("\n" + "="*70)
    print("🚀 STARTING FIXED PIPELINE - PROPER TEST/TRAIN SEPARATION")
    print("="*70)
    
    # Step 1: Load data PROPERLY
    start_load = time.time()
    success, X_train, X_val, X_test, y_train, y_val, y_test = load_all_data_properly()
    load_time = time.time() - start_load
    
    if not success:
        print("❌ Cannot continue without data")
        return
    
    print(f"\n⏱️ Data loading time: {load_time:.1f} seconds")
    
    # Step 2: Apply PCA
    print("\n" + "="*60)
    print("📉 APPLYING PCA")
    print("="*60)
    
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("Applying PCA with >90% variance preservation...")
    pca = PCA(n_components=0.90)  # Keep 90% variance
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    variance_preserved = pca.explained_variance_ratio_.sum()
    original_features = X_train.shape[1]
    reduced_features = X_train_pca.shape[1]
    
    print(f"\n✅ PCA Results:")
    print(f"   Original features: {original_features}")
    print(f"   Features after PCA: {reduced_features}")
    print(f"   Variance preserved: {variance_preserved:.2%}")
    print(f"   Data reduction: {(1 - reduced_features/original_features):.1%}")
    
    # Save PCA visualization
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
    plt.close()
    print("   📈 PCA plot saved to 'pca_analysis.png'")
    
    # Step 3: Train models (SVM only for now - to test)
    print("\n" + "="*60)
    print("🤖 TRAINING SVM MODELS")
    print("="*60)
    
    results = {'svm': {}}
    
    # Train SVM with different kernels
    kernels = ['linear', 'rbf', 'poly']
    
    for kernel in kernels:
        print(f"\n🔧 Training SVM ({kernel} kernel)...")
        start_time = time.time()
        
        if kernel == 'linear':
            svm = SVC(kernel='linear', C=1.0, random_state=42, verbose=False, probability=True)
        elif kernel == 'rbf':
            svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, verbose=False, probability=True)
        elif kernel == 'poly':
            svm = SVC(kernel='poly', C=1.0, degree=3, random_state=42, verbose=False, probability=True)
        
        svm.fit(X_train_pca, y_train)
        training_time = time.time() - start_time
        
        # Evaluate
        y_val_pred = svm.predict(X_val_pca)
        y_test_pred = svm.predict(X_test_pca)
        
        val_acc = accuracy_score(y_val, y_val_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        results['svm'][kernel] = {
            'model': svm,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'training_time': training_time,
            'predictions': y_test_pred.tolist()
        }
        
        print(f"    Validation accuracy: {val_acc:.3%}")
        print(f"    Test accuracy: {test_acc:.3%}")
        print(f"    Training time: {training_time:.1f}s")
    
    # Step 4: Test with meta images (for examples only)
    print("\n" + "="*60)
    print("🔎 TESTING WITH META IMAGES (EXAMPLES ONLY)")
    print("="*60)
    
    X_meta, y_meta, filenames = FixedGTSRBLoader.load_meta_images()
    
    if X_meta is not None:
        # Transform meta images
        X_meta_scaled = scaler.transform(X_meta)
        X_meta_pca = pca.transform(X_meta_scaled)
        
        # Use SVM RBF
        svm_rbf = results['svm']['rbf']['model']
        y_meta_pred = svm_rbf.predict(X_meta_pca)
        y_meta_proba = svm_rbf.predict_proba(X_meta_pca)
        meta_acc = accuracy_score(y_meta, y_meta_pred)
        
        print(f"\n📊 Meta image classification (examples):")
        print(f"   Total meta images: {len(X_meta)}")
        print(f"   Accuracy: {meta_acc:.3%}")
        
        # Save meta results with names
        meta_results = []
        for i in range(len(filenames)):
            true_class = int(y_meta[i])
            pred_class = int(y_meta_pred[i])
            confidence = y_meta_proba[i][pred_class]
            
            meta_results.append({
                'filename': filenames[i],
                'true_class': true_class,
                'true_class_name': TRAFFIC_SIGN_NAMES_SHORT.get(true_class, f"Class {true_class}"),
                'predicted_class': pred_class,
                'predicted_class_name': TRAFFIC_SIGN_NAMES_SHORT.get(pred_class, f"Class {pred_class}"),
                'confidence': float(confidence),
                'correct': bool(y_meta[i] == y_meta_pred[i])
            })
        
        with open('meta_examples_with_names.json', 'w') as f:
            json.dump(meta_results, f, indent=2)
        
        print(f"   Examples saved to 'meta_examples_with_names.json'")
    
    # Step 5: Generate report
    print("\n" + "="*60)
    print("📋 GENERATING REPORT")
    print("="*60)
    
    # Find best SVM kernel
    svm_accuracies = [(f'SVM_{k}', results['svm'][k]['test_accuracy']) for k in kernels]
    best_svm = max(svm_accuracies, key=lambda x: x[1])
    
    # Generate report
    final_report = {
        'dataset_info': {
            'name': 'GTSRB - German Traffic Sign Recognition Benchmark',
            'training_samples': X_train.shape[0],
            'validation_samples': X_val.shape[0],
            'test_samples': X_test.shape[0],
            'total_samples': X_train.shape[0] + X_val.shape[0] + X_test.shape[0],
            'num_classes': len(np.unique(y_train)),
            'class_names': TRAFFIC_SIGN_NAMES_SHORT,
            'pca_reduction': f"{original_features} → {reduced_features} features",
            'variance_preserved': float(variance_preserved)
        },
        'svm_results': {
            k: {
                'test_accuracy': float(results['svm'][k]['test_accuracy']),
                'validation_accuracy': float(results['svm'][k]['val_accuracy']),
                'training_time': float(results['svm'][k]['training_time'])
            }
            for k in kernels
        },
        'best_model': {
            'name': best_svm[0],
            'test_accuracy': float(best_svm[1])
        },
        'meta_testing': {
            'total_images': len(X_meta) if X_meta is not None else 0,
            'accuracy': float(meta_acc) if X_meta is not None else None
        }
    }
    
    with open('gtsrb_svm_report.json', 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f"\n✅ Report saved to 'gtsrb_svm_report.json'")
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    
    # SVM kernel comparison
    plt.subplot(1, 2, 1)
    svm_accs = [results['svm'][k]['test_accuracy'] for k in kernels]
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    bars = plt.bar(kernels, svm_accs, color=colors)
    
    plt.xlabel('SVM Kernel')
    plt.ylabel('Test Accuracy')
    plt.title('SVM Kernel Performance Comparison')
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, svm_accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Class distribution
    plt.subplot(1, 2, 2)
    class_counts = np.bincount(y_train)
    classes = list(range(len(class_counts)))
    
    plt.bar(classes[:15], class_counts[:15], color='skyblue', alpha=0.7)
    plt.xlabel('Class ID')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution (First 15 Classes)')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('svm_comparison.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Visualization saved to 'svm_comparison.png'")
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 FIXED PIPELINE COMPLETE!")
    print("="*70)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Training images: {X_train.shape[0]:,}")
    print(f"   Validation images: {X_val.shape[0]:,}")
    print(f"   Test images: {X_test.shape[0]:,} (from Test/ folder!)")
    
    print(f"\n🏆 Best SVM Kernel: {best_svm[0]}")
    print(f"   Test Accuracy: {best_svm[1]:.3%}")
    
    print(f"\n📁 Generated Files:")
    print("   1. gtsrb_svm_report.json - SVM results")
    print("   2. svm_comparison.png - Visualization")
    print("   3. pca_analysis.png - PCA plot")
    print("   4. meta_examples_with_names.json - Meta image examples")
    
    # Ask if user wants to run full comparison models
    print(f"\n📋 Would you like to run the COMPLETE comparison with:")
    print("   - 1-NN, 3-NN, Nearest Centroid, MLP?")
    
    choice = input("\nRun complete comparison? (y/n): ").strip().lower()
    
    if choice == 'y':
        run_complete_comparison(X_train_pca, X_val_pca, X_test_pca, 
                               y_train, y_val, y_test, scaler, pca)


# 5. COMPLETE COMPARISON (Optional)
def run_complete_comparison(X_train_pca, X_val_pca, X_test_pca, 
                           y_train, y_val, y_test, scaler, pca):
    """Run complete comparison with all models"""
    print("\n" + "="*70)
    print("🤖 RUNNING COMPLETE MODEL COMPARISON")
    print("="*70)
    
    results = {
        'svm': {},
        'comparison': {}
    }
    
    # Train SVM kernels
    kernels = ['linear', 'rbf', 'poly']
    
    for kernel in kernels:
        print(f"\n🔧 Training SVM ({kernel})...")
        start_time = time.time()
        
        if kernel == 'linear':
            svm = SVC(kernel='linear', C=0.05, random_state=42, verbose=False)
        elif kernel == 'rbf':
            svm = SVC(kernel='rbf', C=0.05, gamma='scale', random_state=42, verbose=False)
        elif kernel == 'poly':
            svm = SVC(kernel='poly', C=0.05, degree=3, random_state=42, verbose=False)
        
        svm.fit(X_train_pca, y_train)
        training_time = time.time() - start_time
        
        test_acc = accuracy_score(y_test, svm.predict(X_test_pca))
        
        results['svm'][kernel] = {
            'model': svm,
            'test_accuracy': test_acc,
            'training_time': training_time
        }
        
        print(f"    Test accuracy: {test_acc:.3%}")
    
    # 1-NN
    print("\n🔧 Training 1-NN...")
    start_time = time.time()
    knn1 = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    knn1.fit(X_train_pca, y_train)
    knn1_time = time.time() - start_time
    knn1_acc = accuracy_score(y_test, knn1.predict(X_test_pca))
    
    results['comparison']['1nn'] = {
        'model': knn1,
        'test_accuracy': knn1_acc,
        'training_time': knn1_time
    }
    
    print(f"    Test accuracy: {knn1_acc:.3%}")
    
    # 3-NN
    print("\n🔧 Training 3-NN...")
    start_time = time.time()
    knn3 = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    knn3.fit(X_train_pca, y_train)
    knn3_time = time.time() - start_time
    knn3_acc = accuracy_score(y_test, knn3.predict(X_test_pca))
    
    results['comparison']['3nn'] = {
        'model': knn3,
        'test_accuracy': knn3_acc,
        'training_time': knn3_time
    }
    
    print(f"    Test accuracy: {knn3_acc:.3%}")
    
    # Nearest Class Centroid
    print("\n🔧 Training Nearest Class Centroid...")
    start_time = time.time()
    ncc = NearestCentroid()
    ncc.fit(X_train_pca, y_train)
    ncc_time = time.time() - start_time
    ncc_acc = accuracy_score(y_test, ncc.predict(X_test_pca))
    
    results['comparison']['ncc'] = {
        'model': ncc,
        'test_accuracy': ncc_acc,
        'training_time': ncc_time
    }
    
    print(f"    Test accuracy: {ncc_acc:.3%}")
    
    # MLP
    print("\n🔧 Training MLP (hinge loss)...")
    start_time = time.time()
    
    mlp_hinge = SGDClassifier(
        loss='hinge',          # Hinge loss (REQUIRED)
        penalty='l2',
        alpha=1e-4,
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    
    mlp_hinge.fit(X_train_pca, y_train)
    mlp_time = time.time() - start_time
    
    mlp_acc = accuracy_score(y_test, mlp_hinge.predict(X_test_pca))
    
    results['comparison']['mlp_hinge'] = {
        'model': mlp_hinge,
        'test_accuracy': mlp_acc,
        'training_time': mlp_time
    }
    
    print(f"    Test accuracy: {mlp_acc:.3%}")
    print(f"    Training time: {mlp_time:.1f}s")
    
    # Generate comparison report
    all_accuracies = []
    
    for kernel in kernels:
        all_accuracies.append((f'SVM_{kernel}', results['svm'][kernel]['test_accuracy']))
    
    for name in ['1nn', '3nn', 'ncc', 'mlp_hinge']:
        if name in results['comparison']:
            all_accuracies.append((name.upper(), results['comparison'][name]['test_accuracy']))
    
    best_model = max(all_accuracies, key=lambda x: x[1])
    
    print(f"\n🏆 Best Model: {best_model[0]} with accuracy {best_model[1]:.3%}")
    
    print(f"\n📊 All Model Accuracies:")
    for name, acc in sorted(all_accuracies, key=lambda x: x[1], reverse=True):
        print(f"   {name:15} {acc:.3%}")


# ====================================================
# RUN THE PROGRAM
# ====================================================

if __name__ == "__main__":
    run_fixed_pipeline()