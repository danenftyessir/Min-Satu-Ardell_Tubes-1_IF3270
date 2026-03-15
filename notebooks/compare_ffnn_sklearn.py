import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.models.ffnn import FFNN
from src.optimizers import Adam


def load_and_preprocess_data(data_path='data/datasetml_2026.csv'):
    print("="*60)
    print("LOAD DAN PREPROCESS DATA")
    print("="*60)

    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if 'placement_status' in categorical_cols:
        categorical_cols.remove('placement_status')

    print(f"\nCategorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")

    df_encoded = df.copy()
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    le_target = LabelEncoder()
    df_encoded['placement_status'] = le_target.fit_transform(df_encoded['placement_status'])
    print(f"\nTarget encoding: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

    X = df_encoded.drop('placement_status', axis=1).values
    y = df_encoded['placement_status'].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\nData split:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'label_encoder': le_target,
        'n_features': X_train_scaled.shape[1],
        'n_classes': len(np.unique(y))
    }


def train_ffnn_custom(data, config=None):
    """Training FFNN custom."""
    print("\n" + "="*60)
    print("TRAINING FFNN CUSTOM")
    print("="*60)

    if config is None:
        config = {
            'layer_sizes': [64, 32, 16, 2],
            'activations': ['relu', 'relu', 'softmax'],
            'loss_function': 'categorical_cross_entropy',
            'initializer': 'xavier',
            'learning_rate': 0.001,
            'dropout_rate': 0.2
        }

    model = FFNN(
        layer_sizes=config['layer_sizes'],
        activations=config['activations'],
        loss_function=config['loss_function'],
        initializer=config['initializer'],
        learning_rate=config['learning_rate'],
        dropout_rate=config['dropout_rate']
    )

    optimizer = Adam(learning_rate=config['learning_rate'], weight_decay=0.01)

    history = model.train(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_val=data['X_val'],
        y_val=data['y_val'],
        batch_size=32,
        epochs=100,
        verbose=1,
        optimizer=optimizer,
        patience=15
    )

    return model, history


def train_sklearn_mlp(data, config=None):
    """Training scikit-learn MLPClassifier."""
    print("\n" + "="*60)
    print("TRAINING SKLEARN MLPClassifier")
    print("="*60)

    if config is None:
        config = {
            'hidden_layer_sizes': (64, 32, 16),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.01,
            'learning_rate_init': 0.001,
            'batch_size': 32,
            'max_iter': 100,
            'early_stopping': True,
            'validation_fraction': 0.2,
            'n_iter_no_change': 15,
            'random_state': 42
        }

    model = MLPClassifier(**config)
    model.fit(data['X_train'], data['y_train'])

    print(f"Training selesai! Iterations: {model.n_iter_}")

    return model


def evaluate_models(ffnn_model, mlp_model, data):
    """Evaluate both models."""
    print("\n" + "="*60)
    print("EVALUASI MODEL")
    print("="*60)

    # FFNN predictions
    y_pred_ffnn_train = ffnn_model.predict(data['X_train'])
    y_pred_ffnn_val = ffnn_model.predict(data['X_val'])
    y_pred_ffnn_test = ffnn_model.predict(data['X_test'])

    # MLP predictions
    y_pred_mlp_train = mlp_model.predict(data['X_train'])
    y_pred_mlp_val = mlp_model.predict(data['X_val'])
    y_pred_mlp_test = mlp_model.predict(data['X_test'])

    results = {
        'FFNN Custom': {
            'train_acc': accuracy_score(data['y_train'], y_pred_ffnn_train),
            'val_acc': accuracy_score(data['y_val'], y_pred_ffnn_val),
            'test_acc': accuracy_score(data['y_test'], y_pred_ffnn_test),
            'precision': precision_score(data['y_test'], y_pred_ffnn_test),
            'recall': recall_score(data['y_test'], y_pred_ffnn_test),
            'f1': f1_score(data['y_test'], y_pred_ffnn_test),
            'y_pred_test': y_pred_ffnn_test
        },
        'sklearn MLP': {
            'train_acc': accuracy_score(data['y_train'], y_pred_mlp_train),
            'val_acc': accuracy_score(data['y_val'], y_pred_mlp_val),
            'test_acc': accuracy_score(data['y_test'], y_pred_mlp_test),
            'precision': precision_score(data['y_test'], y_pred_mlp_test),
            'recall': recall_score(data['y_test'], y_pred_mlp_test),
            'f1': f1_score(data['y_test'], y_pred_mlp_test),
            'y_pred_test': y_pred_mlp_test
        }
    }

    print(f"\n{'Metric':<20} {'FFNN Custom':>15} {'sklearn MLP':>15}")
    print("-"*50)
    print(f"{'Train Accuracy':<20} {results['FFNN Custom']['train_acc']:>15.4f} {results['sklearn MLP']['train_acc']:>15.4f}")
    print(f"{'Val Accuracy':<20} {results['FFNN Custom']['val_acc']:>15.4f} {results['sklearn MLP']['val_acc']:>15.4f}")
    print(f"{'Test Accuracy':<20} {results['FFNN Custom']['test_acc']:>15.4f} {results['sklearn MLP']['test_acc']:>15.4f}")
    print(f"{'Precision':<20} {results['FFNN Custom']['precision']:>15.4f} {results['sklearn MLP']['precision']:>15.4f}")
    print(f"{'Recall':<20} {results['FFNN Custom']['recall']:>15.4f} {results['sklearn MLP']['recall']:>15.4f}")
    print(f"{'F1-Score':<20} {results['FFNN Custom']['f1']:>15.4f} {results['sklearn MLP']['f1']:>15.4f}")

    return results


def plot_comparisons(results, data, output_dir='results'):
    """Plot comparison visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    # Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm_ffnn = confusion_matrix(data['y_test'], results['FFNN Custom']['y_pred_test'])
    cm_mlp = confusion_matrix(data['y_test'], results['sklearn MLP']['y_pred_test'])

    sns.heatmap(cm_ffnn, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Placed', 'Placed'],
                yticklabels=['Not Placed', 'Placed'],
                ax=axes[0])
    axes[0].set_title('FFNN Custom\nConfusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Not Placed', 'Placed'],
                yticklabels=['Not Placed', 'Placed'],
                ax=axes[1])
    axes[1].set_title('sklearn MLP\nConfusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['Train Acc', 'Val Acc', 'Test Acc', 'Precision', 'Recall', 'F1-Score']
    ffnn_scores = [
        results['FFNN Custom']['train_acc'],
        results['FFNN Custom']['val_acc'],
        results['FFNN Custom']['test_acc'],
        results['FFNN Custom']['precision'],
        results['FFNN Custom']['recall'],
        results['FFNN Custom']['f1']
    ]
    mlp_scores = [
        results['sklearn MLP']['train_acc'],
        results['sklearn MLP']['val_acc'],
        results['sklearn MLP']['test_acc'],
        results['sklearn MLP']['precision'],
        results['sklearn MLP']['recall'],
        results['sklearn MLP']['f1']
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, ffnn_scores, width, label='FFNN Custom', color='steelblue')
    bars2 = ax.bar(x + width/2, mlp_scores, width, label='sklearn MLP', color='coral')

    ax.set_ylabel('Score')
    ax.set_title('Perbandingan Performa: FFNN Custom vs sklearn MLP')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nVisualizations saved to {output_dir}/")


def save_results(results, output_dir='results'):
    """Save results to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    comparison_df = pd.DataFrame({
        'Metric': ['Train Accuracy', 'Val Accuracy', 'Test Accuracy',
                   'Precision', 'Recall', 'F1-Score'],
        'FFNN Custom': [
            results['FFNN Custom']['train_acc'],
            results['FFNN Custom']['val_acc'],
            results['FFNN Custom']['test_acc'],
            results['FFNN Custom']['precision'],
            results['FFNN Custom']['recall'],
            results['FFNN Custom']['f1']
        ],
        'sklearn MLP': [
            results['sklearn MLP']['train_acc'],
            results['sklearn MLP']['val_acc'],
            results['sklearn MLP']['test_acc'],
            results['sklearn MLP']['precision'],
            results['sklearn MLP']['recall'],
            results['sklearn MLP']['f1']
        ]
    })

    comparison_df['Difference'] = comparison_df['FFNN Custom'] - comparison_df['sklearn MLP']
    comparison_df['Better'] = comparison_df.apply(
        lambda row: 'FFNN' if row['FFNN Custom'] > row['sklearn MLP']
                   else ('MLP' if row['sklearn MLP'] > row['FFNN Custom'] else 'Tie'),
        axis=1
    )

    comparison_df.to_csv(os.path.join(output_dir, 'comparison_results.csv'), index=False)
    print(f"Results saved to {output_dir}/comparison_results.csv")

    return comparison_df


def main():
    """Main function."""
    print("="*60)
    print("PERBANDINGAN FFNN CUSTOM vs SKLEARN MLP")
    print("Tugas Besar 1 IF3270 Pembelajaran Mesin")
    print("="*60)

    data = load_and_preprocess_data()

    ffnn_model, history_ffnn = train_ffnn_custom(data)

    mlp_model = train_sklearn_mlp(data)

    results = evaluate_models(ffnn_model, mlp_model, data)

    plot_comparisons(results, data)

    comparison_df = save_results(results)
    
    print("\n" + "="*60)
    print("KESIMPULAN")
    print("="*60)

    if results['FFNN Custom']['test_acc'] > results['sklearn MLP']['test_acc']:
        winner = "FFNN Custom"
        score = results['FFNN Custom']['test_acc']
    elif results['sklearn MLP']['test_acc'] > results['FFNN Custom']['test_acc']:
        winner = "sklearn MLP"
        score = results['sklearn MLP']['test_acc']
    else:
        winner = "Seri"
        score = results['FFNN Custom']['test_acc']

    print(f"\nBest Model: {winner}")
    print(f"Test Accuracy: {score:.4f} ({score*100:.2f}%)")

    print("\n--- KELEBIHAN FFNN CUSTOM ---")
    print("- Implementasi dari nol (full control)")
    print("- Dapat dikustomisasi sepenuhnya")
    print("- Memudahkan pembelajaran konsep deep learning")

    print("\n--- KELEBIHAN SKLEARN MLP ---")
    print("- Lebih mudah digunakan")
    print("- Optimizer yang sudah dioptimasi")
    print("- Lebih cepat untuk dataset besar")


if __name__ == "__main__":
    main()
