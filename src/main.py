import argparse
import os
import sys
import numpy as np

# setup project path untuk imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.models.ffnn import FFNN
from src.models.autodiff_ffnn import AutodiffFFNN
from src.optimizers import Adam
from src.utils.plotting import plot_training_history
from src.utils.pipeline import prepare_dataset, evaluate_model, save_training_artifacts
from src.utils.io import save_training_history_to_csv, save_predictions_to_csv
import matplotlib.pyplot as plt


def parse_args():
    """parse argumen baris perintah."""
    parser = argparse.ArgumentParser(description='implementasi Feedforward Neural Network')

    # argumen data
    parser.add_argument('--data', type=str, default='data/datasetml_2026.csv',
                        help='path ke file dataset CSV')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='proporsi data untuk testing')
    parser.add_argument('--val-size', type=float, default=0.2,
                        help='proporsi data training untuk validasi')

    # arsitektur model
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                        help='ukuran layer (contoh: --layers 64 32 16 2 untuk 3 hidden layers)')
    parser.add_argument('--activations', type=str, nargs='+', default=None,
                        help='fungsi aktivasi untuk setiap layer (contoh: --activations relu relu softmax)')

    # parameter training
    parser.add_argument('--loss', type=str, default='categorical_cross_entropy',
                        choices=['mse', 'binary_cross_entropy', 'categorical_cross_entropy'],
                        help='fungsi loss')
    parser.add_argument('--initializer', type=str, default='uniform',
                        choices=['zero', 'uniform', 'normal', 'xavier', 'he'],
                        help='metode inisialisasi bobot')
    parser.add_argument('--optimizer', type=str, default='gd',
                        choices=['gd'],
                        help='optimizer (gd=gradient descent)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size untuk training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='jumlah training epochs')
    parser.add_argument('--verbose', type=int, default=1,
                        choices=[0, 1],
                        help='tingkat verbositas (0=silent, 1=progress)')

    # regularisasi
    parser.add_argument('--regularizer', type=str, default=None,
                        choices=['l1', 'l2'],
                        help='tipe regularisasi')
    parser.add_argument('--lambda', type=float, default=0.01,
                        dest='lambda_param',
                        help='parameter lambda regularisasi')

    # normalization
    parser.add_argument('--normalization', type=str, nargs='+', default=None,
                        choices=['none', 'rmsnorm', 'layernorm'],
                        help='normalization untuk setiap layer (contoh: --normalization rmsnorm layernorm none)')

    # output
    parser.add_argument('--output-dir', type=str, default='results',
                        help='direktori untuk menyimpan hasil')
    parser.add_argument('--model-name', type=str, default='ffnn_model',
                        help='nama untuk menyimpan model')

    # aksi
    parser.add_argument('--skip-train', action='store_true',
                        help='lewati training dan hanya evaluasi')
    parser.add_argument('--model-path', type=str, default=None,
                        help='path untuk memuat model yang sudah dilatih')
    parser.add_argument('--use-autodiff', action='store_true',
                        help='gunakan autodiff.Value untuk automatic differentiation')

    return parser.parse_args()


def main():
    """fungsi utama."""
    args = parse_args()


    # buat direktori output
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("FEEDFORWARD NEURAL NETWORK - PROGRAM UTAMA")
    print("="*70)

    # 1. preprocessing
    print("\n[1] PREPROCESSING DATA")
    print("-" * 70)
    _, X_train, X_val, X_test, y_train, y_val, y_test, info = prepare_dataset(
        data_path=args.data,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=42
    )

    print(f"\ninfo dataset:")
    print(f"  fitur: {info['n_features']}")
    print(f"  kelas: {info['n_classes']}")
    print(f"  train: {info['n_train_samples']}, val: {info['n_val_samples']}, test: {info['n_test_samples']}")

    # 2. bangun atau muat model
    print("\n[2] SETUP MODEL")
    print("-" * 70)

    # Tampilkan info bonus yang tersedia
    print("\n>>> FITUR BONUS YANG TERSEDIA:")
    print("  [40%] Autodiff      : --use-autodiff")
    print("  [5%]  Aktivasi Bonus: leakyrelu, elu")
    print("  [5%]  Inisialisasi  : xavier, he")
    print("  [10%] Normalisasi  : rmsnorm, layernorm (--normalization rmsnorm)")
    print("  [40%] Adam Optimizer: Pilih opsi 2 atau 3 di menu")
    print("-" * 70)

    if args.model_path and args.skip_train:
        # muat model yang sudah ada
        print(f"memuat model dari: {args.model_path}")
        model = FFNN(
            layer_sizes=[],
            activations=[],
            loss_function=args.loss,
            initializer=args.initializer
        )
        model.load(args.model_path)
    elif args.use_autodiff:
        # gunakan AutodiffFFNN (40% bonus)
        print("menggunakan AutodiffFFNN dengan autodiff.Value")

        # cek keterbatasan autodiff
        if args.regularizer:
            print("[WARNING] Regularizer tidak didukung dalam mode autodiff, diabaikan.")
        if args.normalization and any(n is not None for n in args.normalization):
            print("[WARNING] Normalization tidak didukung dalam mode autodiff, diabaikan.")

        # ukuran layer default jika tidak ditentukan
        if args.layers is None:
            n_hidden = 2
            hidden_neurons = 32
            args.layers = [info['n_features']] + [hidden_neurons] * n_hidden + [info['n_classes']]

        # aktivasi default jika tidak ditentukan
        if args.activations is None:
            n_activations = len(args.layers) - 1
            args.activations = ['relu'] * (n_activations - 1) + ['softmax']

        print(f"arsitektur:")
        print(f"  ukuran layer: {args.layers}")
        print(f"  aktivasi: {args.activations}")
        print(f"  fungsi loss: {args.loss}")
        print(f"  inisialisasi: {args.initializer}")
        print(f"  mode: AUTODIFF (40% bonus)")

        model = AutodiffFFNN(
            layer_sizes=args.layers,
            activations=args.activations,
            loss_function=args.loss,
            initializer=args.initializer,
            learning_rate=args.learning_rate,
            use_autodiff=True
        )
    else:
        # buat model baru
        # ukuran layer default jika tidak ditentukan
        if args.layers is None:
            n_hidden = 2
            hidden_neurons = 32
            args.layers = [info['n_features']] + [hidden_neurons] * n_hidden + [info['n_classes']]

        # aktivasi default jika tidak ditentukan
        if args.activations is None:
            n_activations = len(args.layers) - 1
            args.activations = ['relu'] * (n_activations - 1) + ['softmax']

        print(f"arsitektur:")
        print(f"  ukuran layer: {args.layers}")
        print(f"  aktivasi: {args.activations}")
        print(f"  fungsi loss: {args.loss}")
        print(f"  inisialisasi: {args.initializer}")

        # Check for saved hyperparameter tuning results
        from src.utils.tune_hyperparams import load_saved_params
        saved_params = load_saved_params()

        # Interactive tuning selection
        print("\n" + "="*70)
        print("HYPERPARAMETER TUNING")
        print("="*70)
        if saved_params:
            print(f"[1] Gunakan parameter optimal (tersimpan)")
            print(f"    GD: lr={saved_params['gd']['learning_rate']}, batch={saved_params['gd']['batch_size']}, acc={saved_params['gd']['accuracy']:.4f}")
            print(f"    Adam: lr={saved_params['adam']['learning_rate']}, wd={saved_params['adam']['weight_decay']}, batch={saved_params['adam']['batch_size']}, acc={saved_params['adam']['accuracy']:.4f}")
            print("[2] Jalankan hyperparameter tuning baru")
        else:
            print("[1] Jalankan hyperparameter tuning baru")
            print("    (Belum ada hasil tuning tersimpan)")
        print("="*70)

        tuning_choice = input("Masukkan pilihan [1/2]: ").strip()

        if tuning_choice == '2' or not saved_params:
            # Run new tuning
            print("\nMenjalankan hyperparameter tuning baru...")
            from src.utils.tune_hyperparams import main as run_tuning
            run_tuning()
            saved_params = load_saved_params()
            print("\nTuning selesai! Lanjutkan dengan training...")
        else:
            print("\nMenggunakan parameter tersimpan...")

        # Now select which optimizer to use
        print("\n" + "="*70)
        print("PILIH OPTIMIZER")
        print("="*70)
        print("[1] Gradient Descent (GD)")
        print("[2] Adam Optimizer")
        print("[3] Bandingkan GD dan Adam")
        print("="*70)

        pilihan = input("Masukkan pilihan [1/2/3]: ").strip()

        if pilihan == '1':
            optimizer_mode = 'gd'
        elif pilihan == '2':
            optimizer_mode = 'adam'
        else:
            optimizer_mode = 'both'

        print(f"  optimizer: {optimizer_mode}")

        # Use saved params if available - adjust architecture to match current features
        n_features = info['n_features']
        if saved_params:
            # Adjust saved architecture to match current number of features
            saved_arch_gd = saved_params['gd']['layer_sizes']
            saved_arch_adam = saved_params['adam']['layer_sizes']

            # Adjust first layer to current feature count, keep rest
            arch_gd = [n_features] + saved_arch_gd[1:]
            arch_adam = [n_features] + saved_arch_adam[1:]

            print("\n>> Menggunakan parameter optimal tersimpan:")
            print(f"   Arsitektur GD: {arch_gd}")
            print(f"   Arsitektur Adam: {arch_adam}")
            print(f"   GD: lr={saved_params['gd']['learning_rate']}, batch={saved_params['gd']['batch_size']}")
            print(f"   Adam: lr={saved_params['adam']['learning_rate']}, wd={saved_params['adam']['weight_decay']}, batch={saved_params['adam']['batch_size']}")

            # Store adjusted architectures for later use
            args.saved_arch_gd = arch_gd
            args.saved_arch_adam = arch_adam
            args.layers = arch_adam  # Default to Adam's architecture
            args.activations = ['relu'] * (len(arch_adam) - 2) + ['softmax']
        else:
            print(f"\n>> Menggunakan arsitektur default ({n_features} features):")
            print(f"   GD: lr=0.01, batch=32")
            print(f"   Adam: lr=0.001, wd=0.01, batch=16")
            args.layers = [n_features, 64, 32, 2]
            args.activations = ['relu', 'relu', 'softmax']

        # setup regularizer
        regularizer = None
        if args.regularizer:
            regularizer = {
                'type': args.regularizer,
                'lambda_param': args.lambda_param
            }
            print(f"  regularizer: {args.regularizer} (lambda={args.lambda_param})")

        # Setup normalization based on current architecture
        # Will be adjusted per model in training loop
        default_normalization = None

        # Use saved normalization or default
        if saved_params:
            model_normalization = saved_params.get('adam', saved_params.get('gd', {})).get('normalization', [None] * (len(args.layers) - 1))
            model_dropout = saved_params.get('adam', saved_params.get('gd', {})).get('dropout_rate', 0.0)
        else:
            model_normalization = [None] * (len(args.layers) - 1)
            model_dropout = 0.0

        model = FFNN(
            layer_sizes=args.layers,
            activations=args.activations,
            loss_function=args.loss,
            initializer=args.initializer,
            learning_rate=args.learning_rate,
            regularizer=regularizer,
            normalization=model_normalization,
            dropout_rate=model_dropout
        )

    # 3. training
    if not args.skip_train:
        print("\n[3] TRAINING")
        print("-" * 70)
        print(f"konfigurasi training:")
        print(f"  epochs: {args.epochs}")
        print(f"  batch size: {args.batch_size}")
        print(f"  learning rate: {args.learning_rate}")
        print(f"  verbose: {args.verbose}")

        # training berbeda untuk AutodiffFFNN vs FFNN
        if args.use_autodiff:
            # training loop untuk AutodiffFFNN
            print("\n menggunakan training loop autodiff...")

            history = {
                'train_loss': [],
                'val_loss': [],
                'train_accuracy': [],
                'val_accuracy': []
            }

            n_samples = X_train.shape[0]
            n_batches = max(1, n_samples // args.batch_size)

            for epoch in range(args.epochs):
                epoch_train_loss = 0.0

                # mini-batch training
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * args.batch_size
                    end_idx = min(start_idx + args.batch_size, n_samples)

                    X_batch = X_train[start_idx:end_idx]
                    y_batch = y_train[start_idx:end_idx]

                    # training step
                    result = model.train_step(X_batch, y_batch, X_val, y_val)
                    epoch_train_loss += result['loss']

                # average metrics
                epoch_train_loss /= n_batches

                # validation
                val_pred = model.forward(X_val)
                val_loss = -np.mean(y_val * np.log(val_pred + 1e-15))

                # accuracy
                train_pred = model.forward(X_train)
                train_acc = np.mean(np.argmax(train_pred, axis=1) == np.argmax(y_train, axis=1))
                val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))

                history['train_loss'].append(epoch_train_loss)
                history['val_loss'].append(val_loss)
                history['train_accuracy'].append(train_acc)
                history['val_accuracy'].append(val_acc)

                if args.verbose == 1 and (epoch % 10 == 0 or epoch == args.epochs - 1):
                    print(f"epoch {epoch+1}/{args.epochs} - "
                          f"loss: {epoch_train_loss:.4f} - val_loss: {val_loss:.4f} - "
                          f"train_acc: {train_acc:.4f} - val_acc: {val_acc:.4f}")
        else:
            # training loop standard untuk FFNN
            if optimizer_mode == 'both':
                # ============ Training dengan GD ============
                print("\n" + "="*70)
                print("TRAINING DENGAN GRADIENT DESCENT")
                print("="*70)

                # Use saved params or args
                gd_lr = saved_params['gd']['learning_rate'] if saved_params else args.learning_rate
                gd_bs = saved_params['gd']['batch_size'] if saved_params else args.batch_size
                gd_arch = getattr(args, 'saved_arch_gd', args.layers)
                gd_activations = ['relu'] * (len(gd_arch) - 2) + ['softmax']
                # Use saved normalization or default
                if saved_params:
                    gd_normalization = saved_params['gd'].get('normalization', [None] * (len(gd_arch) - 1))
                    gd_dropout = saved_params['gd'].get('dropout_rate', 0.0)
                else:
                    gd_normalization = [None] * (len(gd_arch) - 1)
                    gd_dropout = 0.0

                model_gd = FFNN(
                    layer_sizes=gd_arch,
                    activations=gd_activations,
                    loss_function=args.loss,
                    initializer=args.initializer,
                    learning_rate=gd_lr,
                    regularizer=regularizer,
                    normalization=gd_normalization,
                    dropout_rate=gd_dropout
                )

                history_gd = model_gd.train(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    batch_size=gd_bs,
                    learning_rate=gd_lr,
                    epochs=args.epochs,
                    verbose=args.verbose,
                    optimizer=None,
                    patience=5
                )

                # Evaluate GD
                y_pred_gd = model_gd.predict(X_test)
                y_test_labels = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
                acc_gd = np.mean(y_pred_gd == y_test_labels)

                # ============ Training dengan Adam ============
                print("\n" + "="*70)
                print("TRAINING DENGAN ADAM OPTIMIZER")
                print("="*70)

                # Use saved params or defaults
                adam_lr = saved_params['adam']['learning_rate'] if saved_params else 0.0005
                adam_wd = saved_params['adam']['weight_decay'] if saved_params else 0.01
                adam_bs = saved_params['adam']['batch_size'] if saved_params else 16
                adam_arch = getattr(args, 'saved_arch_adam', args.layers)
                adam_activations = ['relu'] * (len(adam_arch) - 2) + ['softmax']
                # Use saved normalization or default
                if saved_params:
                    adam_normalization = saved_params['adam'].get('normalization', [None] * (len(adam_arch) - 1))
                    adam_dropout = saved_params['adam'].get('dropout_rate', 0.0)
                else:
                    adam_normalization = [None] * (len(adam_arch) - 1)
                    adam_dropout = 0.0

                model_adam = FFNN(
                    layer_sizes=adam_arch,
                    activations=adam_activations,
                    loss_function=args.loss,
                    initializer=args.initializer,
                    learning_rate=adam_lr,
                    regularizer=regularizer,
                    normalization=adam_normalization,
                    dropout_rate=adam_dropout
                )

                adam_optimizer = Adam(learning_rate=adam_lr, weight_decay=adam_wd)
                history_adam = model_adam.train(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    batch_size=adam_bs,
                    learning_rate=0.0005,
                    epochs=args.epochs,
                    verbose=args.verbose,
                    optimizer=adam_optimizer,
                    patience=5
                )

                # Evaluate Adam
                y_pred_adam = model_adam.predict(X_test)
                acc_adam = np.mean(y_pred_adam == y_test_labels)

                # Display Adam Learning Rate Info
                adam_stats = adam_optimizer.get_stats()
                print("\n" + "="*70)
                print("ADAM LEARNING RATE INFO")
                print("="*70)
                print(f"Base Learning Rate: {adam_stats['base_learning_rate']}")
                print(f"Beta1 (momentum): {adam_stats['beta1']}")
                print(f"Beta2 (variance): {adam_stats['beta2']}")
                print(f"Current timestep: {adam_stats['timestep']}")
                print(f"Effective LR per layer:")
                for i, eff_lr in enumerate(adam_stats['effective_learning_rates']):
                    print(f"  Layer {i+1}: {eff_lr:.6f} (base: {adam_stats['base_learning_rate']}, multiplier: {eff_lr/adam_stats['base_learning_rate']:.2f}x)")
                print("="*70)

                # ============ Comparison ============
                print("\n" + "="*70)
                print("HASIL PERBANDINGAN")
                print("="*70)
                print(f"Gradient Descent: {acc_gd:.4f} ({acc_gd*100:.2f}%)")
                print(f"Adam Optimizer:  {acc_adam:.4f} ({acc_adam*100:.2f}%)")
                if acc_adam > acc_gd:
                    print(f"\n==> Adam lebih baik dari GD sebesar {(acc_adam - acc_gd)*100:.2f}%")
                elif acc_gd > acc_adam:
                    print(f"\n==> GD lebih baik dari Adam sebesar {(acc_gd - acc_adam)*100:.2f}%")
                else:
                    print("\n==> Keduanya memiliki performa yang sama")
                print("="*70)

                # Use Adam as the main model for saving
                model = model_adam
                history = history_adam

            elif optimizer_mode == 'adam':
                # Training dengan Adam (use saved params if available)
                adam_lr = saved_params['adam']['learning_rate'] if saved_params else 0.0005
                adam_wd = saved_params['adam']['weight_decay'] if saved_params else 0.01
                adam_bs = saved_params['adam']['batch_size'] if saved_params else 16

                adam_optimizer = Adam(learning_rate=adam_lr, weight_decay=adam_wd)
                history = model.train(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    batch_size=adam_bs,
                    learning_rate=adam_lr,
                    epochs=args.epochs,
                    verbose=args.verbose,
                    optimizer=adam_optimizer,
                    patience=5
                )

                # Display Adam Learning Rate Info
                adam_stats = adam_optimizer.get_stats()
                print("\n" + "="*70)
                print("ADAM LEARNING RATE INFO")
                print("="*70)
                print(f"Base Learning Rate: {adam_stats['base_learning_rate']}")
                print(f"Beta1 (momentum): {adam_stats['beta1']}")
                print(f"Beta2 (variance): {adam_stats['beta2']}")
                print(f"Current timestep: {adam_stats['timestep']}")
                print(f"Effective LR per layer:")
                for i, eff_lr in enumerate(adam_stats['effective_learning_rates']):
                    print(f"  Layer {i+1}: {eff_lr:.6f} (base: {adam_stats['base_learning_rate']}, multiplier: {eff_lr/adam_stats['base_learning_rate']:.2f}x)")
                print("="*70)
            else:
                # Training dengan GD (manual update)
                history = model.train(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    epochs=args.epochs,
                    verbose=args.verbose,
                    optimizer=None,
                    patience=5
                )

        # plot training history
        fig = plot_training_history(history)
        plt.savefig(os.path.join(args.output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        print(f"\ntraining history disimpan ke: {os.path.join(args.output_dir, 'training_history.png')}")

        # simpan model dan training history
        if args.use_autodiff:
            # AutodiffFFNN menggunakan save() method
            model_path = os.path.join(args.output_dir, f'{args.model_name}.pkl')
            model.save(model_path)

            # save history separately
            import pickle
            history_path = os.path.join(args.output_dir, f'{args.model_name}_history.pkl')
            with open(history_path, 'wb') as f:
                pickle.dump(history, f)

            print(f"model disimpan ke: {model_path}")
            print(f"training history disimpan ke: {history_path}")
        else:
            # FFNN menggunakan utility function
            model_path, history_path = save_training_artifacts(
                model=model,
                history=history,
                output_dir=args.output_dir,
                model_name=args.model_name
            )
            print(f"model disimpan ke: {model_path}")
            print(f"training history disimpan ke: {history_path}")

    # 4. evaluasi
    print("\n[4] EVALUASI")
    print("-" * 70)

    # evaluasi model
    results = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)

    train_acc = results['train_accuracy']
    val_acc = results['val_accuracy']
    test_acc = results['test_accuracy']

    print(f"train accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"val accuracy:   {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"test accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")

    # metrik tambahan
    print(f"\nmetrik tambahan (test set):")
    print(f"  precision: {results['precision']:.4f}")
    print(f"  recall:    {results['recall']:.4f}")
    print(f"  f1 score:  {results['f1_score']:.4f}")

    # confusion matrix
    print(f"\nconfusion matrix:")
    print(results['confusion_matrix'])

    # Export ke CSV (selalu di data/)
    print("\n[4b] EXPORT CSV")
    print("-" * 70)
    history_csv_path = save_training_history_to_csv(history, 'data')

    test_predictions = model.predict(X_test)
    predictions_csv_path = save_predictions_to_csv(
        test_predictions, y_test, 'data', args.model_name
    )
    print(f"training history CSV: {history_csv_path}")
    print(f"predictions CSV: {predictions_csv_path}")

    # 5. visualisasi
    print("\n[5] VISUALISASI")
    print("-" * 70)

    # plot distribusi bobot
    try:
        model.plot_weight_distribution()
        plt.savefig(os.path.join(args.output_dir, 'weight_distribution.png'), dpi=300, bbox_inches='tight')
        print(f"distribusi bobot disimpan")
    except Exception as e:
        print(f"tidak bisa plot distribusi bobot: {e}")

    # plot distribusi gradien
    try:
        if hasattr(model, 'weight_gradients') and model.weight_gradients:
            model.plot_gradient_distribution()
            plt.savefig(os.path.join(args.output_dir, 'gradient_distribution.png'), dpi=300, bbox_inches='tight')
            print(f"distribusi gradien disimpan")
    except Exception as e:
        print(f"tidak bisa plot distribusi gradien: {e}")

    # 6. ringkasan
    print("\n" + "="*70)
    print("RINGKASAN")
    print("="*70)
    print(f"\nhasil disimpan ke: {args.output_dir}")
    print(f"test accuracy akhir: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"\nfile yang dihasilkan:")
    print(f"  - eda_visualization.png")
    print(f"  - training_history.png")
    print(f"  - weight_distribution.png")
    print(f"  - gradient_distribution.png")
    if not args.skip_train:
        print(f"  - {args.model_name}.pkl (model)")
        print(f"  - {args.model_name}_history.pkl (training history)")

if __name__ == "__main__":
    main()