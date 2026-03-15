import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import os
from pathlib import Path

class DataPreprocessor:
    """
    class untuk preprocessing dan eda dataset.
    """

    def __init__(self, data_path: str):
        """
        inisialisasi DataPreprocessor.

        argumen:
            data_path: path ke file dataset csv
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []

    def load_data(self) -> pd.DataFrame:
        """
        load dataset dari file csv.

        kembali:
            dataframe yang berisi dataset
        """
        print("memuat dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"dataset berhasil dimuat! shape: {self.df.shape}")
        return self.df

    def explore_data(self) -> None:
        """
        exploratory data analysis (eda).
        """
        if self.df is None:
            self.load_data()

        print("\n" + "="*50)
        print("exploratory data analysis")
        print("="*50)

        # 1. info dasar
        print("\n1. info dataset:")
        print(f"   - total baris: {self.df.shape[0]}")
        print(f"   - total kolom: {self.df.shape[1]}")

        # 2. tipe kolom
        print("\n2. tipe kolom:")
        print(self.df.dtypes)

        # 3. nilai yang hilang
        print("\n3. nilai yang hilang:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "   tidak ada nilai yang hilang")

        # 4. ringkasan statistik
        print("\n4. ringkasan statistik untuk kolom numerik:")
        print(self.df.describe())

        # 5. nilai unik untuk kolom kategorikal
        print("\n5. nilai unik untuk kolom kategorikal:")
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            print(f"   - {col}: {self.df[col].nunique()} nilai unik")
            print(f"     nilai: {self.df[col].unique()}")

        # 6. distribusi target
        if 'placement_status' in self.df.columns:
            print("\n6. distribusi target (placement_status):")
            print(self.df['placement_status'].value_counts())
            print(f"\n   persentase:")
            print(self.df['placement_status'].value_counts(normalize=True) * 100)

    def visualize_data(self, save_path: str = None) -> None:
        """
        visualisasi data untuk eda (individual plots).

        argumen:
            save_path: path untuk menyimpan visualisasi. jika None, hanya menampilkan.
                      akan digunakan sebagai base directory untuk menyimpan multiple plots.
        """
        if self.df is None:
            self.load_data()

        print("\n" + "="*50)
        print("membuat visualisasi")
        print("="*50)

        # Tentukan output directory
        if save_path:
            output_dir = os.path.dirname(save_path)
            if not output_dir:
                output_dir = '.'
            os.makedirs(output_dir, exist_ok=True)
            save_mode = True
        else:
            save_mode = False

        # 1. Distribusi fitur numerik (individual plots)
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print(f"\n1. Membuat {len(numerical_cols)} plot distribusi numerik...")
            for col in numerical_cols:
                plt.figure(figsize=(10, 6))
                self.df[col].hist(bins=30, color='skyblue', edgecolor='black', alpha=0.7)
                plt.title(f'Distribution of {col}', fontsize=14, fontweight='bold')
                plt.xlabel(col, fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.grid(True, alpha=0.3)

                if save_mode:
                    filename = f"{col.replace('/', '_')}_distribution.png"
                    filepath = os.path.join(output_dir, filename)
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()
                    plt.close()

        # 2. Distribusi fitur kategorikal (individual plots)
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"2. Membuat {len(categorical_cols)} plot distribusi kategorikal...")
            for col in categorical_cols:
                plt.figure(figsize=(10, 6))
                self.df[col].value_counts().plot(kind='bar', color='lightcoral', alpha=0.7)
                plt.title(f'Distribution of {col}', fontsize=14, fontweight='bold')
                plt.xlabel(col, fontsize=12)
                plt.ylabel('Count', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3, axis='y')

                if save_mode:
                    filename = f"{col.replace('/', '_')}_distribution.png"
                    filepath = os.path.join(output_dir, filename)
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()
                    plt.close()

        # 3. Correlation heatmap (separate plot)
        if len(numerical_cols) > 1:
            print("3. Membuat correlation matrix heatmap...")
            plt.figure(figsize=(12, 10))
            corr_matrix = self.df[numerical_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                       fmt='.2f', annot_kws={'size': 8})
            plt.title('Correlation Matrix - Numerical Features', fontsize=14, fontweight='bold')
            plt.tight_layout()

            if save_mode:
                filepath = os.path.join(output_dir, 'correlation_matrix.png')
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                plt.close()

        # 4. Target distribution (separate plot)
        if 'placement_status' in self.df.columns:
            print("4. Membuat target distribution plot...")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Count plot
            self.df['placement_status'].value_counts().plot(kind='bar', ax=ax1, color=['#2ecc71', '#e74c3c'])
            ax1.set_title('Placement Status Count', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Placement Status', fontsize=12)
            ax1.set_ylabel('Count', fontsize=12)
            ax1.tick_params(axis='x', rotation=0)
            ax1.grid(True, alpha=0.3, axis='y')

            # Percentage plot
            self.df['placement_status'].value_counts(normalize=True).plot(kind='bar', ax=ax2, color=['#2ecc71', '#e74c3c'])
            ax2.set_title('Placement Status Percentage', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Placement Status', fontsize=12)
            ax2.set_ylabel('Percentage', fontsize=12)
            ax2.tick_params(axis='x', rotation=0)
            ax2.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()

            if save_mode:
                filepath = os.path.join(output_dir, 'target_distribution.png')
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"\n[OK] Semua visualisasi disimpan ke: {output_dir}")
            else:
                plt.show()
                plt.close()

    def preprocess_data(
        self,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42
    ) -> tuple:
        """
        preprocess dataset untuk training neural network.

        argumen:
            test_size: proporsi data untuk testing
            val_size: proporsi dari data latih untuk validasi
            random_state: random seed untuk reproduktibilitas

        kembali:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if self.df is None:
            self.load_data()

        print("\n" + "="*50)
        print("preprocessing data")
        print("="*50)

        # 1. tangani nilai yang hilang
        print("\n1. menangani nilai yang hilang...")
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if self.df[col].dtype in ['int64', 'float64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                else:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)

        # 2. encode variabel kategorikal dengan ONE-HOT ENCODING
        print("\n2. encoding variabel kategorikal (One-Hot Encoding)...")
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col != 'placement_status']

        # Identify numerical and categorical columns
        numerical_cols = [col for col in self.df.columns if col != 'placement_status' and col not in categorical_cols]

        print(f"   - numerical columns: {numerical_cols}")
        print(f"   - categorical columns: {categorical_cols}")

        # Store original categorical values before encoding
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols

        # One-Hot Encoding
        self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        # Fit one-hot encoder on categorical columns
        cat_data = self.df[categorical_cols]
        self.onehot_encoder.fit(cat_data)

        # Transform categorical columns to one-hot
        cat_encoded = self.onehot_encoder.transform(cat_data)
        cat_feature_names = self.onehot_encoder.get_feature_names_out(categorical_cols)

        print(f"   - One-hot encoded features: {len(cat_feature_names)}")

        # 2b. FEATURE ENGINEERING - Tambah interaction features
        print("\n2b. feature engineering (interaction features)...")

        # Create a new dataframe with numerical columns
        df_features = self.df[numerical_cols].copy()

        # Add one-hot encoded features
        for i, name in enumerate(cat_feature_names):
            df_features[name] = cat_encoded[:, i]

        # Add interaction features
        # cgpa × aptitude_score
        df_features['cgpa_x_aptitude'] = self.df['cgpa'] * self.df['aptitude_score']

        # cgpa × communication_score
        df_features['cgpa_x_communication'] = self.df['cgpa'] * self.df['communication_score']

        # aptitude × communication
        df_features['aptitude_x_communication'] = self.df['aptitude_score'] * self.df['communication_score']

        # cgpa × internship_quality
        df_features['cgpa_x_internship_quality'] = self.df['cgpa'] * self.df['internship_quality_score']

        # Total experience score
        df_features['total_experience'] = self.df['internship_count'] + self.df['internship_quality_score']

        # Skill score (normalized backlogs + other metrics)
        df_features['skill_score'] = (self.df['aptitude_score'] + self.df['communication_score'] + self.df['internship_quality_score']) / 3

        # High CGPA indicator
        df_features['high_cgpa'] = (self.df['cgpa'] >= 7.5).astype(int)

        print(f"   - Added interaction features: cgpa_x_aptitude, cgpa_x_communication, aptitude_x_communication, etc.")
        print(f"   - Total features after engineering: {df_features.shape[1]}")

        # 3. siapkan fitur dan target
        print("\n3. menyiapkan fitur dan target...")
        # Use df_features which now includes one-hot encoded + interaction features
        self.feature_columns = list(df_features.columns)
        X = df_features.values
        y = self.df['placement_status'].values

        # encode target
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        self.label_encoders['placement_status'] = le_target

        print(f"   - shape fitur: {X.shape}")
        print(f"   - shape target: {y.shape}")
        print(f"   - jumlah kelas: {len(np.unique(y))}")

        # 4. split data: train -> (val + test)
        print("\n4. membagi data...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size),
            random_state=random_state, stratify=y
        )

        # sesuaikan test_size untuk split dari X_temp
        adjusted_test_size = test_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=adjusted_test_size,
            random_state=random_state, stratify=y_temp
        )

        print(f"   - set latih: {X_train.shape[0]} sampel")
        print(f"   - set validasi: {X_val.shape[0]} sampel")
        print(f"   - set uji: {X_test.shape[0]} sampel")

        # 5. scale fitur
        print("\n5. scaling fitur...")
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_val = self.scaler.transform(X_val)
        self.X_test = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        print("   - fitur di-scale menggunakan StandardScaler")
        print(f"   - mean (train): {self.X_train.mean(axis=0)[:3]}... (3 fitur pertama)")
        print(f"   - std (train): {self.X_train.std(axis=0)[:3]}... (3 fitur pertama)")

        return (self.X_train, self.X_val, self.X_test,
                self.y_train, self.y_val, self.y_test)

    def get_data_info(self) -> dict:
        """
        mendapatkan informasi tentang data yang sudah dipreprocess.

        kembali:
            dict: informasi data
        """
        if self.X_train is None:
            raise ValueError("data belum dipreprocess. jalankan preprocess_data() terlebih dahulu.")

        return {
            'n_features': self.X_train.shape[1],
            'n_classes': len(np.unique(self.y_train)),
            'n_train_samples': self.X_train.shape[0],
            'n_val_samples': self.X_val.shape[0],
            'n_test_samples': self.X_test.shape[0],
            'feature_columns': self.feature_columns,
            'class_distribution': {
                'train': np.bincount(self.y_train),
                'val': np.bincount(self.y_val),
                'test': np.bincount(self.y_test)
            }
        }

    def save_processed_data(self, output_dir: str = 'data/processed') -> None:
        """
        menyimpan data yang sudah dipreprocess.

        argumen:
            output_dir: directory untuk menyimpan processed data
        """
        if self.X_train is None:
            raise ValueError("data belum dipreprocess. jalankan preprocess_data() terlebih dahulu.")

        os.makedirs(output_dir, exist_ok=True)

        np.save(os.path.join(output_dir, 'X_train.npy'), self.X_train)
        np.save(os.path.join(output_dir, 'X_val.npy'), self.X_val)
        np.save(os.path.join(output_dir, 'X_test.npy'), self.X_test)
        np.save(os.path.join(output_dir, 'y_train.npy'), self.y_train)
        np.save(os.path.join(output_dir, 'y_val.npy'), self.y_val)
        np.save(os.path.join(output_dir, 'y_test.npy'), self.y_test)

        print(f"\ndata yang dipreprocess disimpan ke: {output_dir}")


def main():
    """
    main function untuk menjalankan preprocessing dan eda.
    """
    # path ke dataset
    data_path = 'data/datasetml_2026.csv'

    # buat output directory
    output_dir = 'data/processed'
    visualization_path = 'data/eda_visualization.png'

    # inisialisasi preprocessor
    preprocessor = DataPreprocessor(data_path)

    # load dan explore data
    preprocessor.load_data()
    preprocessor.explore_data()

    # visualisasi data
    preprocessor.visualize_data(save_path=visualization_path)

    # preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_data()

    # tampilkan info data
    info = preprocessor.get_data_info()
    print("\n" + "="*50)
    print("informasi data")
    print("="*50)
    for key, value in info.items():
        if key != 'feature_columns' and key != 'class_distribution':
            print(f"{key}: {value}")

    print(f"\nkolom fitur: {info['feature_columns']}")
    print(f"\ndistribusi kelas:")
    for split, counts in info['class_distribution'].items():
        print(f"  {split}: {counts}")

    # simpan processed data
    preprocessor.save_processed_data(output_dir)

    print("\n" + "="*50)
    print("preprocessing berhasil diselesaikan!")
    print("="*50)


if __name__ == "__main__":
    main()