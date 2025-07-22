import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.regularizers import l1_l2

# Set randomness for numpy and tensorflow
np.random.seed(42)
tf.random.set_seed(42)

# If you don't have gpu like me, this lets it runs on cpu
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Read and combine both datasets
df_hdac2 = pd.read_csv('HDAC2.csv')
df_hdac3 = pd.read_csv('HDAC3.csv')

# Add identifier columns to distinguish the datasets
df_hdac2['hdac_type'] = 'HDAC2'
df_hdac3['hdac_type'] = 'HDAC3'

# Rename mean columns to be more specific
df_hdac2.rename(columns={'mean.x': 'hdac2_mean'}, inplace=True)
df_hdac3.rename(columns={'mean.x': 'hdac3_mean'}, inplace=True)

# Merge the datasets on common columns (assuming same patients)
# If they don't have common patient IDs, we'll concatenate them
try:
    # Try to merge if there's a common identifier (adjust column name as needed)
    df = pd.merge(df_hdac2, df_hdac3, on=['race', 'gender', 'age_at_index', 'ajcc_pathologic_stage'], 
                  how='inner', suffixes=('_hdac2', '_hdac3'))
    print("Successfully merged datasets based on patient characteristics")
except:
    # If merge fails, create combined features by concatenating datasets
    print("Merging failed, combining datasets with averaged features per patient group")
    
    # For demonstration, we'll create a combined dataset
    # In practice, you'd want proper patient matching
    df_hdac2['hdac3_mean'] = np.nan
    df_hdac3['hdac2_mean'] = np.nan
    
    # Fill missing values with mean of the respective HDAC type
    hdac2_mean_avg = df_hdac2['hdac2_mean'].mean()
    hdac3_mean_avg = df_hdac3['hdac3_mean'].mean()
    
    df_hdac2['hdac3_mean'] = hdac3_mean_avg
    df_hdac3['hdac2_mean'] = hdac2_mean_avg
    
    # Combine datasets
    df = pd.concat([df_hdac2, df_hdac3], ignore_index=True)

# Encode categorical variables to numbers
label_encoders = {}
catcols = ['race', 'gender']
for col in catcols:
    le = LabelEncoder()
    df[f'{col}s'] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define inputs and outputs - now including both HDAC2 and HDAC3 means
features = ['hdac2_mean', 'hdac3_mean', 'races', 'genders', 'age_at_index']
X = df[features].fillna(X.mean())  # Handle any remaining NaN values
cancerstage = LabelEncoder()
y = cancerstage.fit_transform(df['ajcc_pathologic_stage'])

# Fixing the class imbalance using SMOTE
smote = SMOTE(random_state=42, k_neighbors=min(3, min(np.bincount(y)) - 1))
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, 
                                                    test_size=0.2, random_state=42, 
                                                    stratify=y_resampled)

# Scale the features so the model can learn the true importance of each input
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Get the class weights to further fix the class imbalance
class_weights = compute_class_weight(class_weight='balanced', 
                                   classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(zip(np.unique(y_train), class_weights))

# Enhanced model architecture for combined features
model = Sequential([
    # Input layer with more nodes to handle combined features
    Dense(512, input_dim=X_train_scaled.shape[1], activation='relu', 
          kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
    BatchNormalization(),
    Dropout(0.3),
    
    # Hidden layers
    Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.005, l2=0.005)),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    Dropout(0.2),
    
    # Output layer
    Dense(len(np.unique(y)), activation='softmax')
])

# Compile model
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model with callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=25,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=15,
    min_lr=1e-7,
    verbose=1
)

print("Training combined HDAC2 & HDAC3 model...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=300,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Model evaluation
y_pred_prob = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n=== COMBINED HDAC2 & HDAC3 MODEL RESULTS ===")
print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Weighted F1 Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=cancerstage.classes_))

print("\nAccuracy per class:")
for i, class_label in enumerate(cancerstage.classes_):
    classindex = (y_test == i)
    if np.sum(classindex) > 0:  # Check if class exists in test set
        class_accuracy = accuracy_score(y_test[classindex], y_pred[classindex])
        print(f"{class_label}: {class_accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=cancerstage.classes_, 
           yticklabels=cancerstage.classes_)
plt.title('Confusion Matrix - Combined HDAC2 & HDAC3 Cancer Stage Classification')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# ROC Curves
y_test_roc = label_binarize(y_test, classes=range(len(cancerstage.classes_)))
classes = y_test_roc.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_roc[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(12, 8))
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
for i in range(classes):
    plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
            label=f'{cancerstage.classes_[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Combined HDAC2 & HDAC3 Cancer Stage Classification')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()