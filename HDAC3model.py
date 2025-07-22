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

# Read data
df = pd.read_csv('HDAC3.csv')

# Encode categorical variables to numbers
label_encoders = {}
catcols = ['race', 'gender']
for col in catcols:
    l = LabelEncoder()
    df[f'{col}s'] = l.fit_transform(df[col])
    label_encoders[col] = l

# Define inputs and outputs
features = ['mean.x', 'races', 'genders', 'age_at_index']
X = df[features]
cancerstage = LabelEncoder()
y = cancerstage.fit_transform(df['ajcc_pathologic_stage'])



# Fixing the class imbalance using SMOTE this time instead of ROS
smote = SMOTE(random_state=42, k_neighbors=min(3, min(np.bincount(y)) - 1))
X_resampled, y_resampled = smote.fit_resample(X, y)



# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Scale the features so the model can learn the true importance of each input
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Get the class weights to further fix the class imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(zip(np.unique(y_train), class_weights))

# Actual model
model = Sequential([
    # Input layer. Nodes, input dimension, activation function, regularization to reduce overfitting, batch normalization, and dropout to reduce overfitting
    Dense(256, input_dim=X_train_scaled.shape[1], activation='relu', 
          kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
    BatchNormalization(),
    Dropout(0.3),
    
    # Hidden layers
    # Number of nodes, activation function, regularization
    Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    Dropout(0.2),
    
    # Output
    Dense(len(np.unique(y)), activation='softmax')
])

# Compile model using Adam optimizer with hyperparameters to affect how fast the model learns in order to stop overfitting
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model with earlystopping and reducing learning rate to stop overfitting
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

# Test more epochs if u have a gpu
history = model.fit(
    X_train_scaled, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evalution of model from here to the end
y_pred_prob = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Weighted F1 Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=cancerstage.classes_))

print("\nAccuracy per class:")
for i, class_label in enumerate(cancerstage.classes_):
    classindex = (y_test == i)
    class_accuracy = accuracy_score(y_test[classindex], y_pred[classindex])
    print(f"{class_label}: {class_accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cancerstage.classes_, yticklabels=cancerstage.classes_)
plt.title('Confusion Matrix - HDAC2 Cancer Stage Classification')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

y_test_roc = label_binarize(y_test, classes=range(len(cancerstage.classes_)))
classes = y_test_roc.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_roc[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(12, 8))
colors = ['blue', 'orange', 'green', 'red']
for i in range(classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
            label=f'{cancerstage.classes_[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - HDAC2 Cancer Stage Classification')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
