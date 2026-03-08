import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
        visualisasi data untuk eda.

        argumen:
            save_path: path untuk menyimpan visualisasi. jika None, hanya menampilkan.
        """
        if self.df is None:
            self.load_data()

        print("\n" + "="*50)
        print("membuat visualisasi")
        print("="*50)

        # buat figure dengan subplots
        fig = plt.figure(figsize=(20, 15))

        # 1. distribusi fitur numerik
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        n_num = len(numerical_cols)

        if n_num > 0:
            for i, col in enumerate(numerical_cols[:6], 1):  # maksimal 6 plot numerik
                ax = fig.add_subplot(4, 3, i)
                self.df[col].hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')

        # 2. fitur kategorikal
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        n_cat = len(categorical_cols)

        start_idx = 7
        for i, col in enumerate(categorical_cols[:6], start_idx):
            if i > 12:
                break
            ax = fig.add_subplot(4, 3, i)
            self.df[col].value_counts().plot(kind='bar', ax=ax, color='lightcoral')
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # 3. correlation heatmap (untuk kolom numerik)
        if n_num > 1:
            ax = fig.add_subplot(4, 3, 12)
            corr_matrix = self.df[numerical_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.5}, ax=ax)
            ax.set_title('Correlation Matrix')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nvisualisasi disimpan ke: {save_path}")
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

        # 2. encode variabel kategorikal
        print("\n2. encoding variabel kategorikal...")
        categorical_cols = self.df.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            if col != 'placement_status':  # jangan encode target di sini
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le

        # 3. siapkan fitur dan target
        print("\n3. menyiapkan fitur dan target...")
        self.feature_columns = [col for col in self.df.columns if col != 'placement_status']
        X = self.df[self.feature_columns].values
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