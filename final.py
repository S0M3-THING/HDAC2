import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, accuracy_score, f1_score, 
                           confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.regularizers import l1_l2
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Force CPU usage if needed
os.environ["CUDA_VISIBLE_DEVICES"] = ""

class CancerStageClassifier:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_encoder = LabelEncoder()
        self.class_weights = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("Loading and preprocessing data...")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        
        # Check for missing values
        print(f"Missing values:\n{self.df.isnull().sum()}")
        
        # Handle missing values
        self.df = self.df.dropna()
        
        # Fix column name issue - looks like you have 'mean.x' but use 'mean2' in original code
        if 'mean.x' in self.df.columns:
            self.df['mean2'] = self.df['mean.x']
        
        # Encode categorical variables
        categorical_cols = ['race', 'gender']
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[f'{col}s'] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
        
        # Define features and target
        self.features = ['mean2', 'races', 'genders', 'age_at_index']
        self.X = self.df[self.features]
        self.y = self.target_encoder.fit_transform(self.df['ajcc_pathologic_stage'])
        
        print(f"Dataset shape: {self.X.shape}")
        print(f"Target distribution:\n{pd.Series(self.y).value_counts().sort_index()}")
        
        return self.X, self.y
    
    def handle_class_imbalance(self, X, y):
        """Handle class imbalance using SMOTE"""
        print("Handling class imbalance with SMOTE...")
        
        # Use SMOTE instead of RandomOverSampler for better synthetic samples
        smote = SMOTE(random_state=42, k_neighbors=min(3, min(np.bincount(y)) - 1))
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"Original distribution: {np.bincount(y)}")
        print(f"Resampled distribution: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def split_and_scale_data(self, X, y):
        """Split data and apply scaling"""
        print("Splitting and scaling data...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Compute class weights
        self.class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        self.class_weights_dict = dict(zip(np.unique(y_train), self.class_weights))
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def build_model(self, input_dim, num_classes):
        """Build an improved neural network model"""
        print("Building model...")
        
        model = Sequential([
            # Input layer with batch normalization
            Dense(256, input_dim=input_dim, activation='relu', 
                  kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layers
            Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            # Output layer
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile with custom optimizer
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model with callbacks"""
        print("Training model...")
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train model
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        history = self.model.fit(
            X_train, y_train,
            epochs=200,
            batch_size=32,
            validation_split=0.2 if validation_data is None else 0.0,
            validation_data=validation_data,
            class_weight=self.class_weights_dict,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        print("Evaluating model...")
        
        # Predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Weighted F1 Score: {f1:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.target_encoder.classes_))
        
        # Per-class accuracy
        print("\nPer-class Accuracy:")
        for i, class_label in enumerate(self.target_encoder.classes_):
            class_indices = (y_test == i)
            if np.sum(class_indices) > 0:
                class_accuracy = accuracy_score(y_test[class_indices], y_pred[class_indices])
                print(f"{class_label}: {class_accuracy:.4f}")
        
        return y_pred, y_pred_prob
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.target_encoder.classes_,
                   yticklabels=self.target_encoder.classes_)
        plt.title('Confusion Matrix - HDAC2 Cancer Stage Classification')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, y_test, y_pred_prob):
        """Plot ROC curves for multiclass classification"""
        # Binarize the output
        y_test_bin = label_binarize(y_test, classes=range(len(self.target_encoder.classes_)))
        n_classes = y_test_bin.shape[1]
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curves
        plt.figure(figsize=(12, 8))
        colors = ['aqua', 'darkorange', 'cornflowerblue', 'red']
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                    label=f'{self.target_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - HDAC2 Cancer Stage Classification')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot training & validation loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def cross_validate(self, X, y, cv_folds=5):
        """Perform cross-validation"""
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        cv_scores = []
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"Training fold {fold+1}/{cv_folds}")
            
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            # Scale data
            scaler_cv = StandardScaler()
            X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
            X_val_cv_scaled = scaler_cv.transform(X_val_cv)
            
            # Build and train model
            model_cv = self.build_model(X_train_cv_scaled.shape[1], len(np.unique(y)))
            
            # Compute class weights for this fold
            class_weights_cv = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train_cv),
                y=y_train_cv
            )
            class_weights_dict_cv = dict(zip(np.unique(y_train_cv), class_weights_cv))
            
            # Train
            model_cv.fit(
                X_train_cv_scaled, y_train_cv,
                epochs=100,
                batch_size=32,
                class_weight=class_weights_dict_cv,
                validation_data=(X_val_cv_scaled, y_val_cv),
                verbose=0
            )
            
            # Evaluate
            y_pred_cv = model_cv.predict(X_val_cv_scaled)
            y_pred_cv = np.argmax(y_pred_cv, axis=1)
            score = accuracy_score(y_val_cv, y_pred_cv)
            cv_scores.append(score)
            
            print(f"Fold {fold+1} accuracy: {score:.4f}")
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
        
        return cv_scores
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        print("="*50)
        print("HDAC2 Cancer Stage Classification Pipeline")
        print("="*50)
        
        # Load and preprocess data
        X, y = self.load_and_preprocess_data()
        
        # Handle class imbalance
        X_resampled, y_resampled = self.handle_class_imbalance(X, y)
        
        # Split and scale data
        X_train, X_test, y_train, y_test = self.split_and_scale_data(X_resampled, y_resampled)
        
        # Build model
        self.model = self.build_model(X_train.shape[1], len(np.unique(y)))
        
        # Train model
        history = self.train_model(X_train, y_train)
        
        # Evaluate model
        y_pred, y_pred_prob = self.evaluate_model(X_test, y_test)
        
        # Plot results
        self.plot_training_history(history)
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_roc_curves(y_test, y_pred_prob)
        
        # Cross-validation on original data (without resampling)
        X_scaled = self.scaler.fit_transform(X)
        cv_scores = self.cross_validate(X_scaled, y)
        
        print("="*50)
        print("Pipeline completed successfully!")
        print("="*50)
        
        return self.model, history
    
    def predict_single_sample(self, mean2, race, gender, age_at_index):
        """Predict cancer stage for a single sample"""
        if self.model is None:
            raise ValueError("Model not trained yet. Run the pipeline first.")
        
        # Encode categorical variables
        race_encoded = self.label_encoders['race'].transform([race])[0]
        gender_encoded = self.label_encoders['gender'].transform([gender])[0]
        
        # Create feature array
        sample = np.array([[mean2, race_encoded, gender_encoded, age_at_index]])
        
        # Scale features
        sample_scaled = self.scaler.transform(sample)
        
        # Predict
        pred_prob = self.model.predict(sample_scaled)[0]
        pred_class = np.argmax(pred_prob)
        pred_stage = self.target_encoder.inverse_transform([pred_class])[0]
        
        return pred_stage, pred_prob

# Usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = CancerStageClassifier('HDAC2.csv')
    
    # Run the complete pipeline
    model, history = classifier.run_full_pipeline()
    
    # Example prediction
    print("\nExample Prediction:")
    try:
        pred_stage, pred_prob = classifier.predict_single_sample(
            mean2=-0.2, race='white', gender='male', age_at_index=65
        )
        print(f"Predicted stage: {pred_stage}")
        print(f"Prediction probabilities: {pred_prob}")
    except Exception as e:
        print(f"Prediction error: {e}")