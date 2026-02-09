"""
Tier B: The Semanticist
Feedforward Neural Network using Averaged Pre-trained Embeddings (GloVe)

This script implements a semantic-based classifier for distinguishing
Human, AI-Generic, and AI-Styled text using word embeddings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support
)
from sklearn.manifold import TSNE
import re
import warnings
import os  # ADD THIS
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class LabelSmoothingCrossEntropy(nn.Module):
        def __init__(self, epsilon=0.1):
            super().__init__()
            self.epsilon = epsilon
        
        def forward(self, preds, target):
            n_classes = preds.size(-1)
            log_preds = nn.functional.log_softmax(preds, dim=-1)
            loss = -log_preds.sum(dim=-1).mean()
            nll = nn.functional.nll_loss(log_preds, target)
            return (1 - self.epsilon) * nll + self.epsilon * loss / n_classes



class TierBSemanticist:
    def __init__(self, embedding_path='glove.6B.100d.txt', embedding_dim=100, output_dir='outputs'):
        self.embedding_path = embedding_path
        self.embedding_dim = embedding_dim
        self.output_dir = output_dir
        self.embeddings_index = {}
        self.label_encoder = LabelEncoder()
        
        # CREATE OUTPUT DIRECTORY IF IT DOESN'T EXIST
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"✓ Output directory: {os.path.abspath(self.output_dir)}")
        
        # Data
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Model
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = None
        
    def load_glove_embeddings(self):
        """Load GloVe embeddings from file"""
        print("=" * 70)
        print("TIER B: THE SEMANTICIST")
        print("=" * 70)
        print(f"\n Loading GloVe embeddings from {self.embedding_path}...")
        print("   This may take a few minutes...")
        
        try:
            with open(self.embedding_path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.array(values[1:], dtype='float32')
                    self.embeddings_index[word] = vector
            
            print(f"✓ Loaded {len(self.embeddings_index):,} word vectors")
            print(f"✓ Embedding dimension: {self.embedding_dim}")
            
            # Show some example words
            sample_words = list(self.embeddings_index.keys())[:10]
            print(f"\n Sample words in vocabulary: {', '.join(sample_words)}")
            
        except FileNotFoundError:
            raise
        
        return self
    
    def preprocess_text(self, text):
        if pd.isna(text):
            return []
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove special characters but keep spaces and hyphens
        text = re.sub(r'[^a-z\s\-]', ' ', text)
        
        # Split into tokens
        tokens = text.split()
        
        # Remove very short tokens and standalone hyphens
        tokens = [t for t in tokens if len(t) > 1 and t != '-']
        
        return tokens
    
    def text_to_embedding(self, text):
        # Tokenize
        tokens = self.preprocess_text(text)
        
        # Get embeddings for each word
        word_vectors = []
        found_words = 0
        
        for token in tokens:
            if token in self.embeddings_index:
                word_vectors.append(self.embeddings_index[token])
                found_words += 1
        
        # Handle case where no words have embeddings
        if len(word_vectors) == 0:
            return np.zeros(self.embedding_dim)
        
        # Average all word vectors
        paragraph_vector = np.mean(word_vectors, axis=0)
        
        return paragraph_vector
    
    def load_and_prepare_data(self, data_path, text_column='text'):
        print("\n" + "=" * 70)
        print("DATA PREPARATION")
        print("=" * 70)
        
        print(f"\n📊 Loading data from {data_path}...")
        self.df = pd.read_csv(data_path)
        
        # Check for required columns
        if text_column not in self.df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV. Available columns: {list(self.df.columns)}")
        
        if 'class' not in self.df.columns:
            raise ValueError("Column 'class' not found in CSV")
        
        print(f"✓ Loaded {len(self.df)} samples")
        
        # Display class distribution
        print("\n📈 Class Distribution:")
        class_dist = self.df['class'].value_counts()
        for class_name, count in class_dist.items():
            print(f"   {class_name}: {count} samples ({count/len(self.df)*100:.1f}%)")
        
        # Convert texts to embeddings
        print(f"\n🔄 Converting texts to {self.embedding_dim}D embeddings...")
        
        embeddings = []
        coverage_stats = {'found': 0, 'total': 0}
        
        for idx, row in self.df.iterrows():
            embedding = self.text_to_embedding(row[text_column])
            embeddings.append(embedding)
            
            # Track coverage
            tokens = self.preprocess_text(row[text_column])
            coverage_stats['total'] += len(tokens)
            coverage_stats['found'] += sum(1 for t in tokens if t in self.embeddings_index)
            
            if (idx + 1) % 500 == 0:
                print(f"   Processed {idx + 1}/{len(self.df)} samples...", end='\r')
        
        print(f"   Processed {len(self.df)}/{len(self.df)} samples... ✓")
        
        # Calculate vocabulary coverage
        if coverage_stats['total'] > 0:
            coverage = 100 * coverage_stats['found'] / coverage_stats['total']
            print(f"\n📊 Vocabulary Coverage: {coverage:.2f}%")
            print(f"   ({coverage_stats['found']:,} / {coverage_stats['total']:,} words found in GloVe)")
        
        # Convert to numpy array
        X = np.array(embeddings)
        y = self.df['class'].values
        
        print(f"\n✓ Created embedding matrix: {X.shape}")
        
        return X, y
    
    def prepare_train_test_split(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """Split data into train, validation, and test sets"""
        print("\n" + "=" * 70)
        print("TRAIN-VAL-TEST SPLIT")
        print("=" * 70)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"\n📊 Label Encoding:")
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"   {label} → {i}")
        
        # First split: train+val vs test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=y_encoded
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )
        
        print(f"\n✓ Train set: {len(self.X_train)} samples ({len(self.X_train)/len(X)*100:.1f}%)")
        print(f"✓ Val set:   {len(self.X_val)} samples ({len(self.X_val)/len(X)*100:.1f}%)")
        print(f"✓ Test set:  {len(self.X_test)} samples ({len(self.X_test)/len(X)*100:.1f}%)")
        
        return self
    
    def diagnose_data_leakage(self):
        print("\n" + "=" * 70)
        print("DATA LEAKAGE DIAGNOSIS")
        print("=" * 70)
        
        # Check if embeddings are too similar within classes
        print("\n1. Checking embedding similarity within/between classes...")
        
        from scipy.spatial.distance import cosine
        
        # Sample some embeddings from each class
        for class_idx, class_name in enumerate(self.label_encoder.classes_):
            # Get samples from this class
            mask = self.y_train == class_idx
            class_embeddings = self.X_train[mask][:100]  # Take first 100
            
            if len(class_embeddings) < 2:
                continue
                
            # Calculate average within-class similarity
            similarities = []
            for i in range(min(50, len(class_embeddings))):
                for j in range(i+1, min(50, len(class_embeddings))):
                    sim = 1 - cosine(class_embeddings[i], class_embeddings[j])
                    similarities.append(sim)
            
            avg_sim = np.mean(similarities) if similarities else 0
            print(f"   {class_name}: Avg within-class similarity = {avg_sim:.4f}")
    
        print("\n2. Checking between-class similarity...")
        for i, class1 in enumerate(self.label_encoder.classes_):
            for j, class2 in enumerate(self.label_encoder.classes_):
                if i >= j:
                    continue
                
                mask1 = self.y_train == i
                mask2 = self.y_train == j
                
                emb1 = self.X_train[mask1][:50]
                emb2 = self.X_train[mask2][:50]
                
                similarities = []
                for e1 in emb1[:20]:
                    for e2 in emb2[:20]:
                        sim = 1 - cosine(e1, e2)
                        similarities.append(sim)
                
                avg_sim = np.mean(similarities) if similarities else 0
                print(f"   {class1} vs {class2}: Avg similarity = {avg_sim:.4f}")
    

        print("\n3. Checking for duplicate embeddings...")
        unique_embeddings = np.unique(self.X_train, axis=0)
        duplicate_ratio = 1 - (len(unique_embeddings) / len(self.X_train))
        print(f"   Duplicate embeddings: {duplicate_ratio*100:.2f}%")
        
        # Check embedding variance
        print("\n4. Checking embedding statistics...")
        print(f"   Mean embedding norm: {np.linalg.norm(self.X_train, axis=1).mean():.4f}")
        print(f"   Std embedding norm: {np.linalg.norm(self.X_train, axis=1).std():.4f}")
        print(f"   Min embedding value: {self.X_train.min():.4f}")
        print(f"   Max embedding value: {self.X_train.max():.4f}")
        
        # Check if embeddings are all zeros or very similar
        zero_embeddings = np.all(self.X_train == 0, axis=1).sum()
        print(f"   Zero embeddings: {zero_embeddings} ({zero_embeddings/len(self.X_train)*100:.2f}%)")
        
        # Sample some actual texts and their embeddings
        print("\n5. Sample texts and embeddings:")
        for class_idx, class_name in enumerate(self.label_encoder.classes_):
            print(f"\n   {class_name}:")
            mask = self.y_train == class_idx
            indices = np.where(mask)[0][:2]
            
            for idx in indices:
                # Get original text
                original_idx = self.df.index[indices[0]]
                text_preview = self.df.iloc[original_idx]['text'][:100]
                embedding_norm = np.linalg.norm(self.X_train[idx])
                print(f"      Text: {text_preview}...")
                print(f"      Embedding norm: {embedding_norm:.4f}")
    def analyze_class_discriminators(self):
        print("\n" + "=" * 70)
        print("ANALYZING WHAT DISTINGUISHES CLASSES")
        print("=" * 70)
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance
        
        # Train a simple random forest to see feature importance
        print("\n1. Training Random Forest to identify discriminative embedding dimensions...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(self.X_train, self.y_train)
        
        rf_acc = rf.score(self.X_test, self.y_test)
        print(f"   Random Forest test accuracy: {rf_acc:.4f}")
        
        # Get feature importance
        importances = rf.feature_importances_
        top_features = np.argsort(importances)[-10:][::-1]
        
        print(f"\n   Top 10 most important embedding dimensions:")
        for i, feat_idx in enumerate(top_features):
            print(f"      Dimension {feat_idx}: importance = {importances[feat_idx]:.4f}")
        
        # 2. Analyze which words contribute most to each dimension
        print("\n2. Analyzing vocabulary differences...")
        
        # Get most distinctive words for each class
        from collections import Counter, defaultdict
        
        class_words = defaultdict(Counter)
        class_total_words = defaultdict(int)
        
        for idx, row in self.df.iterrows():
            tokens = self.preprocess_text(row['text'])
            class_name = row['class']
            class_words[class_name].update(tokens)
            class_total_words[class_name] += len(tokens)
        
        print("\n   Most frequent words by class:")
        for class_name in sorted(class_words.keys()):
            total = class_total_words[class_name]
            print(f"\n   {class_name} (total words: {total}):")
            top_words = class_words[class_name].most_common(20)
            print(f"      {', '.join([f'{w}({c})' for w, c in top_words])}")
        
        # 3. Find words unique or heavily skewed to each class
        print("\n3. Finding distinctive vocabulary for each class...")
        
        # Calculate TF-IDF-like score for each word in each class
        all_words = set()
        for counter in class_words.values():
            all_words.update(counter.keys())
        
        for class_name in sorted(class_words.keys()):
            distinctive_scores = {}
            
            for word in all_words:
                if word not in self.embeddings_index:  # Only consider words we have embeddings for
                    continue
                    
                # Frequency in this class
                freq_in_class = class_words[class_name][word] / max(class_total_words[class_name], 1)
                
                # Average frequency in other classes
                other_classes = [c for c in class_words.keys() if c != class_name]
                freq_in_others = sum(class_words[c][word] / max(class_total_words[c], 1) 
                                    for c in other_classes) / len(other_classes)
                
                # Distinctive score (higher = more unique to this class)
                if freq_in_others > 0:
                    distinctive_scores[word] = freq_in_class / freq_in_others
                elif freq_in_class > 0:
                    distinctive_scores[word] = float('inf')
            
            # Show top distinctive words
            top_distinctive = sorted(distinctive_scores.items(), 
                                    key=lambda x: x[1], reverse=True)[:15]
            
            print(f"\n   Most distinctive words for '{class_name}':")
            for word, score in top_distinctive:
                if score != float('inf'):
                    count = class_words[class_name][word]
                    print(f"      {word}: {score:.2f}x more common (appears {count} times)")
        
        # 4. Check sentence structure patterns
        print("\n4. Analyzing sentence structure patterns...")
        
        for class_name in sorted(self.df['class'].unique()):
            class_texts = self.df[self.df['class'] == class_name]['text']
            
            avg_words_per_text = class_texts.str.split().str.len().mean()
            avg_sentence_length = class_texts.str.split('.').str.len().mean()
            avg_commas = class_texts.str.count(',').mean()
            avg_semicolons = class_texts.str.count(';').mean()
            
            print(f"\n   {class_name}:")
            print(f"      Avg words per paragraph: {avg_words_per_text:.1f}")
            print(f"      Avg sentences per paragraph: {avg_sentence_length:.1f}")
            print(f"      Avg commas per paragraph: {avg_commas:.1f}")
            print(f"      Avg semicolons per paragraph: {avg_semicolons:.1f}")
        
        print("\n" + "=" * 70)


    def check_actual_text_differences(self):
        """Compare actual text samples side by side"""
        print("\n" + "=" * 70)
        print("SIDE-BY-SIDE TEXT COMPARISON")
        print("=" * 70)
        
        # Get one sample from each class
        samples = {}
        for class_name in self.df['class'].unique():
            samples[class_name] = self.df[self.df['class'] == class_name].iloc[0]['text']
        
        print("\n📖 SAMPLE FROM EACH CLASS:\n")
        
        for class_name in ['human', 'ai_neutral', 'ai_styled']:
            if class_name in samples:
                print(f"\n{'='*70}")
                print(f"CLASS: {class_name}")
                print('='*70)
                print(samples[class_name])
        
        print("\n\n❓ HUMAN EVALUATION:")
        print("   Can YOU tell these apart just by reading?")
        print("   If not, the task might be unrealistically hard.\n")
        
        
    def build_model(self, hidden_layers=[64, 32], dropout_rates=[0.7, 0.6]):  # Even smaller, more dropout
        print("\n" + "=" * 70)
        print("MODEL ARCHITECTURE")
        print("=" * 70)
        
        num_classes = len(self.label_encoder.classes_)
        
        layers = []
        input_dim = self.embedding_dim
        
        print(f"\n🏗️  Building Feedforward Neural Network:")
        print(f"   Input: {input_dim} (embedding dimension)")
        
        for i, (hidden_size, dropout) in enumerate(zip(hidden_layers, dropout_rates)):
            layers.extend([
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout)
            ])
            print(f"   Layer {i+1}: {input_dim} → {hidden_size} (ReLU, BatchNorm, Dropout={dropout})")
            input_dim = hidden_size
        
        # Add final dropout before output
        layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(input_dim, num_classes))
        print(f"   Output: {input_dim} → {num_classes} (classes)")
        
        self.model = nn.Sequential(*layers)
        self.model = self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n📊 Model Summary:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Device: {self.device}")
        
        return self

    '''def build_model(self, hidden_layers=[128, 64], dropout_rates=[0.5, 0.4]):
        print("\n" + "=" * 70)
        print("MODEL ARCHITECTURE")
        print("=" * 70)
        
        num_classes = len(self.label_encoder.classes_)
        
        layers = []
        input_dim = self.embedding_dim
        
        print(f"\n🏗️  Building Feedforward Neural Network:")
        print(f"   Input: {input_dim} (embedding dimension)")
        
        for i, (hidden_size, dropout) in enumerate(zip(hidden_layers, dropout_rates)):
            layers.extend([
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout)
            ])
            print(f"   Layer {i+1}: {input_dim} → {hidden_size} (ReLU, BatchNorm, Dropout={dropout})")
            input_dim = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_dim, num_classes))
        print(f"   Output: {input_dim} → {num_classes} (classes)")
        
        self.model = nn.Sequential(*layers)
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n📊 Model Summary:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Device: {self.device}")
        
        return self'''
    
    
    def train(self, epochs=50, batch_size=64, learning_rate=0.0005, patience=10, weight_decay=0.01):
        print("\n" + "=" * 70)
        print("TRAINING")
        print("=" * 70)
                
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(self.X_train),
            torch.LongTensor(self.y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(self.X_val),
            torch.LongTensor(self.y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizer with L2 REGULARIZATION
        criterion = LabelSmoothingCrossEntropy(epsilon=0.1)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"\n🚀 Starting training...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Weight decay (L2): {weight_decay}")
        print(f"   Early stopping patience: {patience}")
        print()
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # GRADIENT CLIPPING to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f'Epoch [{epoch+1:3d}/{epochs}] '
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:6.2f}% | '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:6.2f}%')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping triggered at epoch {epoch + 1}")
                print(f"   Best validation loss: {best_val_loss:.4f}")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print("\n✓ Restored best model weights")
        
        print(f"\n✅ Training complete!")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        print(f"   Final validation accuracy: {max(self.history['val_acc']):.2f}%")
        
        return self

    def evaluate(self):
        """Evaluate model on test set"""
        print("\n" + "=" * 70)
        print("EVALUATION ON TEST SET")
        print("=" * 70)
        
        self.model.eval()
        
        # Get predictions
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(self.X_test).to(self.device)
            outputs = self.model(X_test_tensor)
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.cpu().numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions, average='weighted')
        
        print(f"\n🎯 Test Accuracy: {accuracy:.4f}")
        print(f"📊 Weighted F1 Score: {f1:.4f}")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print("-" * 70)
        target_names = self.label_encoder.classes_
        print(classification_report(self.y_test, predictions, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, predictions)
        self._plot_confusion_matrix(cm, target_names)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': predictions,
            'confusion_matrix': cm
        }
    
    def _plot_confusion_matrix(self, cm, labels):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8}
        )
        
        plt.title('Confusion Matrix - Tier B (Semantic)\n(Normalized by True Label)',
                 fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # USE self.output_dir
        save_path = os.path.join(self.output_dir, 'tier_b_confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved confusion matrix to {save_path}")
        plt.close()
    
    def plot_training_history(self):
        """Plot training and validation metrics"""
        print("\n📊 Generating training history plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(self.history['train_acc'], label='Train Accuracy', linewidth=2)
        axes[1].plot(self.history['val_acc'], label='Val Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # USE self.output_dir
        save_path = os.path.join(self.output_dir, 'tier_b_training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training history to {save_path}")
        plt.close()
    
    def visualize_embeddings_tsne(self):
        """Visualize embeddings in 2D using t-SNE"""
        print("\n🔍 Generating t-SNE visualization...")
        print("   This may take a few minutes for large datasets...")
        
        # Combine all data for visualization
        X_all = np.vstack([self.X_train, self.X_val, self.X_test])
        y_all = np.concatenate([self.y_train, self.y_val, self.y_test])
        
        # Sample if too large
        max_samples = 2000
        if len(X_all) > max_samples:
            indices = np.random.choice(len(X_all), max_samples, replace=False)
            X_sample = X_all[indices]
            y_sample = y_all[indices]
            print(f"   Sampling {max_samples} points for visualization")
        else:
            X_sample = X_all
            y_sample = y_all
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        X_2d = tsne.fit_transform(X_sample)
        
        # Plot
        plt.figure(figsize=(12, 8))
        class_names = self.label_encoder.classes_
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        
        for i, class_name in enumerate(class_names):
            mask = y_sample == i
            plt.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                label=class_name,
                alpha=0.6,
                s=50,
                c=colors[i],
                edgecolors='black',
                linewidth=0.5
            )
        
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.title('Embedding Space Visualization (t-SNE)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # USE self.output_dir
        save_path = os.path.join(self.output_dir, 'tier_b_embeddings_tsne.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved t-SNE visualization to {save_path}")
        plt.close()
    
    def save_model(self, filename='tier_b_model.pth'):
        """Save model weights"""
        save_path = os.path.join(self.output_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_encoder': self.label_encoder,
            'embedding_dim': self.embedding_dim
        }, save_path)
        print(f"\n💾 Model saved to {save_path}")
    
    def generate_report(self, results):
        """Generate final analysis report"""
        print("\n" + "=" * 70)
        print("TIER B: FINAL ANALYSIS REPORT")
        print("=" * 70)
        
        print(f"\n TEST SET PERFORMANCE:")
        print(f"   Accuracy: {results['accuracy']:.4f}")
        print(f"   F1 Score: {results['f1_score']:.4f}")
        
        print("\n📝 KEY FINDINGS:")
        print("-" * 70)
        
        if results['f1_score'] > 0.8:
            print("✓ Model shows STRONG semantic discriminative power")
            print("  → Word meanings differ significantly between classes")
            print("  → Semantic features are highly effective for this task")
        elif results['f1_score'] > 0.6:
            print("⚠ Model shows MODERATE semantic discriminative power")
            print("  → Some semantic overlap exists between classes")
            print("  → Consider combining with Tier A features")
        else:
            print("✗ Model shows WEAK semantic discriminative power")
            print("  → Semantic features alone may not be sufficient")
            print("  → Classes may use similar vocabulary/topics")
            print("  → Move to Tier C for contextual understanding")
        
        print("\n" + "=" * 70)


def main():
    """Main execution function"""
    # Configuration - UPDATED PATHS
    EMBEDDING_PATH = r'D:\precog_task\glove.6B.100d.txt'  # Use raw string
    DATA_PATH = r'D:\precog_task\paragraphs_dataset_FIXED1.csv'  # Use raw string
    OUTPUT_DIR = r'D:\precog_task\outputs'  # Use raw string
    TEXT_COLUMN = 'text'
    
    print("\n" + "=" * 70)
    print("TIER B: THE SEMANTICIST")
    print("Feedforward NN with Pre-trained Embeddings")
    print("=" * 70)
    
    # Initialize with output_dir
    classifier = TierBSemanticist(
        embedding_path=EMBEDDING_PATH,
        embedding_dim=100,
        output_dir=OUTPUT_DIR
    )
    
    # Load embeddings
    classifier.load_glove_embeddings()
    
    # Load and prepare data
    X, y = classifier.load_and_prepare_data(DATA_PATH, TEXT_COLUMN)
    
    # Train-test split
    classifier.prepare_train_test_split(X, y, test_size=0.2, val_size=0.1)
    classifier.check_actual_text_differences()
    classifier.analyze_class_discriminators()

    classifier.diagnose_data_leakage()
    
    # Build model with REDUCED COMPLEXITY
    classifier.build_model(
        hidden_layers=[128, 64],  # REDUCED from [256, 128, 64]
        dropout_rates=[0.5, 0.4]  # INCREASED dropout
    )
    
    # Train with ANTI-OVERFITTING measures
    classifier.train(
        epochs=50,              # REDUCED from 100
        batch_size=64,          # INCREASED from 32
        learning_rate=0.0005,   # REDUCED from 0.001
        patience=10,            # REDUCED from 15
        weight_decay=0.01       # ADDED L2 regularization
    )
    
    # Plot training history
    classifier.plot_training_history()
    
    # Evaluate
    results = classifier.evaluate()
    
    # Visualizations
    classifier.visualize_embeddings_tsne()
    
    # Save model
    classifier.save_model()
    
    # Generate report
    classifier.generate_report(results)
    
    print("\n✅ Tier B analysis complete!")
    print(f"📁 All outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()