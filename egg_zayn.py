import os
import sys
import numpy as np
import mne
from mne.datasets import eegbci
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
import threading
import time
import psutil
import datetime
import warnings

# Uyarıları bastır
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Bağımlılık kontrolü
required_libs = ['mne', 'numpy', 'sklearn', 'tensorflow', 'matplotlib', 'seaborn', 'psutil']
for lib in required_libs:
    try:
        __import__(lib)
    except ImportError:
        print(f"Hata: {lib} kütüphanesi eksik. Lütfen kurun: pip install {lib}")
        sys.exit(1)

# Mixed precision optimizasyonu
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Veri artırma ve contrastive learning için çiftler
def augment_data(X, noise_factor=0.01):
    X_aug = X.copy()
    noise = np.random.normal(0, noise_factor, X.shape)
    X_aug += noise
    return X_aug

def create_contrastive_pairs(X):
    X_pos = augment_data(X)
    X_neg = np.roll(X, shift=1, axis=0)
    return X_pos, X_neg

# EggZayn v9.4 Modeli
class EggZaynModel:
    def __init__(self):
        self.model = None
        self.class_names = ['Left Fist', 'Right Fist', 'Both Fists', 'Both Feet']
        self.history = None

    def prepare_eegmmidb_data(self, epoch_duration=1.0, target_sfreq=160):
        """EEGMMIDB verisini hatasız ve boyut uyumlu şekilde işler."""
        data_dir = './eeg_data'
        subjects = range(1, 110)
        runs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        
        print("EggZayn: EEGMMIDB verisi hazırlanıyor...")
        os.makedirs(data_dir, exist_ok=True)
        total_files = len(subjects) * len(runs)
        processed_files = 0
        
        raw_list = []
        motor_channels = ['Fc3', 'Fc4', 'C3', 'C4', 'Cz', 'Cp3', 'Cp4']
        for subject in subjects:
            for run in runs:
                file_path = f"{data_dir}/S{subject:03d}/S{subject:03d}R{run:02d}.edf"
                if not os.path.exists(file_path):
                    continue
                
                try:
                    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
                    raw.resample(target_sfreq, npad='auto', verbose=False)
                    
                    raw.notch_filter(60, verbose=False)
                    raw.filter(8, 30, fir_design='firwin', verbose=False)
                    
                    available_channels = [ch for ch in raw.ch_names if any(mc.upper() in ch.upper() for mc in motor_channels)]
                    if len(available_channels) < 1:
                        raise ValueError(f"Denek {subject}, Run {run}: Hiç motor kanal bulunamadı.")
                    
                    raw.pick(available_channels)
                    if len(available_channels) < 7:
                        raw.set_montage('standard_1020')
                        missing_channels = [ch for ch in motor_channels if ch not in available_channels]
                        raw.interpolate_bads(reset_bads=True, mode='accurate', exclude=missing_channels)
                        raw.pick(motor_channels)
                    
                    events = mne.make_fixed_length_events(raw, duration=epoch_duration)
                    labels = self.assign_labels(run, len(events))
                    raw_list.append((raw, events, labels))
                    processed_files += 1
                    print(f"İlerleme: {processed_files}/{total_files}")
                except Exception as e:
                    print(f"Hata: Denek {subject}, Run {run} işlenemedi: {e}")
                    continue
        
        if not raw_list:
            raise ValueError("EggZayn: Hiçbir veri işlenemedi, veri setinde ciddi bir sorun var.")
        
        X_all, Y_all = [], []
        expected_samples = int(target_sfreq * epoch_duration)  # 160 Hz * 1 sn = 160 örnek
        for raw, events, labels in raw_list:
            epochs = mne.Epochs(raw, events, tmin=0, tmax=epoch_duration, baseline=None, preload=True, verbose=False)
            X = epochs.get_data(picks='eeg')
            if X.shape[2] != expected_samples:
                X_resampled = np.zeros((X.shape[0], X.shape[1], expected_samples))
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        X_resampled[i, j, :] = np.interp(
                            np.linspace(0, 1, expected_samples),
                            np.linspace(0, 1, X.shape[2]),
                            X[i, j, :]
                        )
                X = X_resampled
            X = (X - X.min(axis=2, keepdims=True)) / (X.max(axis=2, keepdims=True) - X.min(axis=2, keepdims=True))
            
            # Veri ve etiket eşitleme
            if X.shape[0] != len(labels):
                min_len = min(X.shape[0], len(labels))
                X = X[:min_len]
                labels = labels[:min_len]
                print(f"Uyarı: Veri ve etiket eşitlemesi yapıldı. Yeni boyut: {min_len}")
            
            X_all.append(X)
            Y_all.append(labels)
        
        X = np.concatenate(X_all, axis=0)
        Y = np.concatenate(Y_all, axis=0)
        
        # Son eşitleme kontrolü
        if X.shape[0] != len(Y):
            min_len = min(X.shape[0], len(Y))
            X = X[:min_len]
            Y = Y[:min_len]
            print(f"Uyarı: Son eşitleme yapıldı. Yeni boyut: {min_len}")
        
        unique, counts = np.unique(Y, return_counts=True)
        print(f"EggZayn: Sınıf dağılımı: {dict(zip(unique, counts))}")
        
        X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
        X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp)
        
        np.save('X_train.npy', X_train)
        np.save('Y_train.npy', Y_train)
        np.save('X_val.npy', X_val)
        np.save('Y_val.npy', Y_val)
        np.save('X_test.npy', X_test)
        np.save('Y_test.npy', Y_test)
        
        print(f"EggZayn: Veri hazır: {X.shape[0]} örnek, Şekil: {X.shape}")
        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    def assign_labels(self, run, num_events):
        """Optimize edilmiş etiket atama."""
        label_map = {
            (1, 2): 0,  # Baseline
            (3, 5, 7): [0, 1],  # Sol/Sağ yumruk
            (4, 6, 8): [0, 1],  # Sol/Sağ imagery
            (9, 11, 13): [2, 3],  # Her iki yumruk/ayak
            (10, 12, 14): [2, 3]  # Her iki yumruk/ayak imagery
        }
        for runs, labels in label_map.items():
            if run in runs:
                if isinstance(labels, int):
                    return np.full(num_events, labels, dtype=int)
                return np.array([labels[i % 2] for i in range(num_events)])
        raise ValueError(f"Geçersiz run numarası: {run}")

    def process_signal(self, signal_data, epoch_duration=1.0, target_sfreq=160):
        """Anlık sinyal veya dosya girişini hatasız ve ultra gelişmiş yöntemlerle işler."""
        if isinstance(signal_data, str):
            raw = mne.io.read_raw(signal_data, preload=True, verbose=False)
        else:
            if not isinstance(signal_data, np.ndarray):
                raise ValueError("EggZayn: Anlık sinyal numpy array olmalı.")
            info = mne.create_info(ch_names=['Fc3', 'Fc4', 'C3', 'C4', 'Cz', 'Cp3', 'Cp4'], sfreq=target_sfreq, ch_types='eeg')
            raw = mne.io.RawArray(signal_data, info)
        
        if raw.info['sfreq'] != target_sfreq:
            raw.resample(target_sfreq, npad='auto', verbose=False)
        
        raw.notch_filter(60, verbose=False)
        raw.filter(8, 30, fir_design='firwin', verbose=False)
        
        available_channels = [ch for ch in raw.ch_names if ch.upper() in ['FC3', 'FC4', 'C3', 'C4', 'CZ', 'CP3', 'CP4']]
        if len(available_channels) < 1:
            raise ValueError("EggZayn: Hiç motor kanal bulunamadı.")
        
        raw.pick(available_channels)
        if len(available_channels) < 7:
            raw.set_montage('standard_1020')
            missing_channels = [ch for ch in ['Fc3', 'Fc4', 'C3', 'C4', 'Cz', 'Cp3', 'Cp4'] if ch not in available_channels]
            raw.interpolate_bads(reset_bads=True, mode='accurate', exclude=missing_channels)
            raw.pick(['Fc3', 'Fc4', 'C3', 'C4', 'Cz', 'Cp3', 'Cp4'])
        
        events = mne.make_fixed_length_events(raw, duration=epoch_duration)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=epoch_duration, baseline=None, preload=True, verbose=False)
        X = epochs.get_data(picks='eeg')
        expected_samples = int(target_sfreq * epoch_duration)
        if X.shape[2] != expected_samples:
            X_resampled = np.zeros((X.shape[0], X.shape[1], expected_samples))
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    X_resampled[i, j, :] = np.interp(
                        np.linspace(0, 1, expected_samples),
                        np.linspace(0, 1, X.shape[2]),
                        X[i, j, :]
                    )
            X = X_resampled
        X = (X - X.min(axis=2, keepdims=True)) / (X.max(axis=2, keepdims=True) - X.min(axis=2, keepdims=True))
        return X

    def build_transformer_block(self, x, num_heads=4, key_dim=32, ff_dim=64):
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        ffn = layers.Dense(ff_dim, activation='gelu')(x)
        ffn = layers.Dense(x.shape[-1])(ffn)
        x = layers.Add()([x, ffn])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x

    def build_encoder(self, input_shape):
        """Geliştirilmiş encoder for contrastive learning."""
        inputs = layers.Input(shape=input_shape)
        x = layers.Dense(32, activation='gelu')(inputs)
        x = layers.Dropout(0.05)(x)
        
        for _ in range(4):
            x = self.build_transformer_block(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation='gelu')(x)
        outputs = layers.Dense(64)(x)
        
        return models.Model(inputs, outputs)

    def contrastive_loss(self, labels, z1, z2, margin=1.0):
        """Kendi contrastive loss fonksiyonumuz."""
        # Türleri float32'ye çevir
        labels = tf.cast(labels, tf.float32)
        z1 = tf.cast(z1, tf.float32)
        z2 = tf.cast(z2, tf.float32)
        margin = tf.cast(margin, tf.float32)

        # Mesafeleri hesapla
        squared_distance = tf.reduce_sum(tf.square(z1 - z2), axis=-1)
        distance = tf.sqrt(squared_distance + tf.keras.backend.epsilon())

        # Pozitif ve negatif çiftler için kayıp
        positive_loss = labels * squared_distance
        negative_loss = (1 - labels) * tf.square(tf.maximum(margin - distance, 0))
        loss = 0.5 * (positive_loss + negative_loss)
        return tf.reduce_mean(loss)

    def pretrain(self, X_train, epochs=3):
        """Contrastive learning ile pretraining."""
        encoder = self.build_encoder(X_train.shape[1:])
        X_pos, X_neg = create_contrastive_pairs(X_train)
        
        inputs1 = layers.Input(shape=X_train.shape[1:])
        inputs2 = layers.Input(shape=X_train.shape[1:])
        z1 = encoder(inputs1)
        z2 = encoder(inputs2)
        model = models.Model([inputs1, inputs2], [z1, z2])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # Kendi loss fonksiyonumuzu kullanarak modeli derle
        @tf.function
        def train_step(X1, X2, labels):
            with tf.GradientTape() as tape:
                z1, z2 = model([X1, X2], training=True)
                loss = self.contrastive_loss(labels, z1, z2)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        # Eğitim döngüsü
        batch_size = 128
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            for i in range(0, len(X_pos), batch_size):
                X1_batch = X_pos[i:i+batch_size]
                X2_batch = X_neg[i:i+batch_size]
                labels_batch = np.ones(len(X1_batch))
                loss = train_step(X1_batch, X2_batch, labels_batch)
                print(f"Batch {i//batch_size+1}: Loss = {loss.numpy():.4f}")
        
        return encoder

    def train(self, X_train, Y_train, X_val, Y_val, save_path='EggZayn_final.h9_4'):
        """Ultra gelişmiş ve hatasız eğitim."""
        encoder = self.pretrain(X_train)
        inputs = layers.Input(shape=X_train.shape[1:])
        x = encoder(inputs)
        x = layers.Dense(256, activation='gelu')(x)
        x = layers.Dropout(0.05)(x)
        outputs = layers.Dense(4, activation='softmax', dtype='float32')(x)
        self.model = models.Model(inputs, outputs)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(save_path, save_best_only=True, monitor='val_accuracy', mode='max'),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6)
        ]
        
        self.history = self.model.fit(X_train, Y_train, epochs=10, batch_size=128, 
                                      validation_data=(X_val, Y_val), callbacks=callbacks, verbose=1)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        with open(save_path.replace('.h9_4', '.tflite'), 'wb') as f:
            f.write(tflite_model)
        
        print(f"EggZayn: Model {save_path} ve {save_path.replace('.h9_4', '.tflite')} olarak kaydedildi.")
        return self.history

    def evaluate(self, X_test, Y_test):
        if self.model is None:
            raise ValueError("EggZayn: Model eğitilmedi veya yüklenmedi.")
        
        loss, accuracy = self.model.evaluate(X_test, Y_test, verbose=0)
        Y_pred = np.argmax(self.model.predict(X_test, verbose=0), axis=1)
        report = classification_report(Y_test, Y_pred, target_names=self.class_names)
        cm = confusion_matrix(Y_test, Y_pred)
        return loss, accuracy, report, cm

    def predict(self, signal_input):
        """Prompt tabanlı, ultra gelişmiş tahmin ve raporlama."""
        if self.model is None:
            if os.path.exists('EggZayn_final.h9_4'):
                self.model = tf.keras.models.load_model('EggZayn_final.h9_4')
            else:
                raise ValueError("EggZayn: Model bulunamadı.")
        
        start_time = time.time()
        X_processed = self.process_signal(signal_input)
        predictions = self.model.predict(X_processed, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        probabilities = [max(prob) for prob in predictions]
        analysis_time = time.time() - start_time
        
        results = [(self.class_names[pred], prob) for pred, prob in zip(predicted_classes, probabilities)]
        
        # Profesyonel raporlama
        report = f"Analiz Raporu - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Toplam Örnek: {len(results)}\n"
        report += f"Analiz Süresi: {analysis_time:.3f} saniye\n"
        report += "Sonuçlar:\n"
        for i, (label, prob) in enumerate(results):
            report += f"Örnek {i+1}: {label} (Güven: {prob*100:.2f}%)\n"
        
        return results, report

    def load_model(self, model_path='EggZayn_final.h9_4'):
        self.model = tf.keras.models.load_model(model_path)
        print(f"EggZayn: Model {model_path} yüklendi.")

# GUI: EggZaynGUI
class EggZaynGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EggZayn v9.4")
        self.root.geometry("1200x900")
        self.root.configure(bg='#1A2526')
        
        self.model = EggZaynModel()
        self.X_train, self.Y_train = None, None
        self.X_val, self.Y_val = None, None
        self.X_test, self.Y_test = None, None
        
        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style()
        style.configure('TButton', font=('Arial', 14, 'bold'), background='#00A8E8', foreground='white')
        style.configure('TLabel', font=('Arial', 12), background='#1A2526', foreground='#ECF0F1')
        
        top_frame = ttk.Frame(self.root)
        top_frame.pack(pady=20)
        
        ttk.Button(top_frame, text="EggZayn'ı Eğit (EEGMMIDB)", command=self.run_full_process_thread).pack(side=tk.LEFT, padx=10)
        ttk.Button(top_frame, text="Kendi Veri Setimle Eğit", command=self.run_custom_train_thread).pack(side=tk.LEFT, padx=10)
        ttk.Button(top_frame, text="Sinyal Analiz Et", command=self.predict_new_data).pack(side=tk.LEFT, padx=10)
        
        self.status_label = ttk.Label(self.root, text="Durum: Hazır")
        self.status_label.pack(pady=10)
        
        self.progress = ttk.Progressbar(self.root, length=500, mode='determinate')
        self.progress.pack(pady=10)
        
        self.result_frame = ttk.Frame(self.root)
        self.result_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        self.result_text = tk.Text(self.result_frame, height=15, width=100, bg='#ECF0F1', fg='#2C3E50', font=('Arial', 11))
        self.result_text.pack(pady=5, padx=5)
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
        self.fig.patch.set_facecolor('#1A2526')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(pady=10)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        self.toolbar.pack()

    def full_process(self):
        self.status_label.config(text="EggZayn: Veri hazırlama aşaması...")
        self.progress['value'] = 0
        self.root.update()
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "EggZayn: EEGMMIDB hazırlanıyor...\n")
        self.root.update()
        start_time = time.time()
        try:
            X_train, Y_train, X_val, Y_val, X_test, Y_test = self.model.prepare_eegmmidb_data()
            self.X_train, self.Y_train = X_train, Y_train
            self.X_val, self.Y_val = X_val, Y_val
            self.X_test, self.Y_test = X_test, Y_test
            self.progress['value'] = 33
            self.result_text.insert(tk.END, f"EggZayn: Veri hazır! Süre: {time.time() - start_time:.2f} saniye\n")
        except Exception as e:
            self.result_text.insert(tk.END, f"Hata: Veri hazırlama başarısız: {e}\n")
            messagebox.showerror("Hata", f"EggZayn: Veri hazırlama başarısız: {e}")
            return
        
        self.status_label.config(text="EggZayn: Model eğitim aşaması...")
        self.result_text.insert(tk.END, "EggZayn: Model eğitiliyor...\n")
        self.root.update()
        start_time = time.time()
        try:
            self.model.train(self.X_train, self.Y_train, self.X_val, self.Y_val)
            self.progress['value'] = 66
            self.result_text.insert(tk.END, f"EggZayn: Eğitim tamamlandı! Süre: {time.time() - start_time:.2f} saniye\n")
            self.update_training_plot()
        except Exception as e:
            self.result_text.insert(tk.END, f"Hata: Eğitim başarısız: {e}\n")
            messagebox.showerror("Hata", f"EggZayn: Eğitim başarısız: {e}")
            return
        
        self.status_label.config(text="EggZayn: Değerlendirme aşaması...")
        self.result_text.insert(tk.END, "EggZayn: Model değerlendiriliyor...\n")
        self.root.update()
        start_time = time.time()
        try:
            loss, accuracy, report, cm = self.model.evaluate(self.X_test, self.Y_test)
            self.progress['value'] = 100
            self.result_text.insert(tk.END, f"EggZayn: Değerlendirme tamamlandı! Süre: {time.time() - start_time:.2f} saniye\n")
            self.result_text.insert(tk.END, f"\nTest Loss: {loss:.4f}\nTest Accuracy: {accuracy:.4f}\n\nClassification Report:\n{report}\n")
            
            self.ax2.clear()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.model.class_names, yticklabels=self.model.class_names, ax=self.ax2)
            self.ax2.set_title('EggZayn Confusion Matrix', color='white')
            self.ax2.set_xlabel('Predicted', color='white')
            self.ax2.set_ylabel('True', color='white')
            self.ax2.tick_params(colors='white')
            self.canvas.draw()
            
            self.status_label.config(text="EggZayn: Model hazır!")
        except Exception as e:
            self.result_text.insert(tk.END, f"Hata: Değerlendirme başarısız: {e}\n")
            messagebox.showerror("Hata", f"EggZayn: Değerlendirme başarısız: {e}")

    def custom_train(self):
        file_path = filedialog.askopenfilename(title="EEG Veri Dosyasını Seç (EDF veya NumPy)", 
                                               filetypes=[("EDF files", "*.edf"), ("NumPy files", "*.npy")])
        if not file_path:
            return
        
        self.status_label.config(text="EggZayn: Kendi veri seti hazırlanıyor...")
        self.progress['value'] = 0
        self.root.update()
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "EggZayn: Kendi veri seti hazırlanıyor...\n")
        self.root.update()
        start_time = time.time()
        try:
            if file_path.endswith('.npy'):
                data = np.load(file_path, allow_pickle=True)
                if 'X' not in data or 'Y' not in data:
                    raise ValueError("EggZayn: .npy dosyasında 'X' ve 'Y' anahtarları olmalı.")
                X_temp, Y_temp = data['X'], data['Y']
                X_train, X_temp, Y_train, Y_temp = train_test_split(X_temp, Y_temp, test_size=0.3, random_state=42, stratify=Y_temp)
                X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp)
            else:
                X_train, Y_train, X_val, Y_val, X_test, Y_test = self.model.prepare_custom_data(file_path)
            self.X_train, self.Y_train = X_train, Y_train
            self.X_val, self.Y_val = X_val, Y_val
            self.X_test, self.Y_test = X_test, Y_test
            self.progress['value'] = 33
            self.result_text.insert(tk.END, f"EggZayn: Kendi veri hazır! Süre: {time.time() - start_time:.2f} saniye\n")
        except Exception as e:
            self.result_text.insert(tk.END, f"Hata: Kendi veri hazırlama başarısız: {e}\n")
            messagebox.showerror("Hata", f"EggZayn: Kendi veri hazırlama başarısız: {e}")
            return
        
        self.status_label.config(text="EggZayn: Model eğitim aşaması...")
        self.result_text.insert(tk.END, "EggZayn: Model eğitiliyor...\n")
        self.root.update()
        start_time = time.time()
        try:
            self.model.train(self.X_train, self.Y_train, self.X_val, self.Y_val)
            self.progress['value'] = 66
            self.result_text.insert(tk.END, f"EggZayn: Eğitim tamamlandı! Süre: {time.time() - start_time:.2f} saniye\n")
            self.update_training_plot()
        except Exception as e:
            self.result_text.insert(tk.END, f"Hata: Eğitim başarısız: {e}\n")
            messagebox.showerror("Hata", f"EggZayn: Eğitim başarısız: {e}")
            return
        
        self.status_label.config(text="EggZayn: Değerlendirme aşaması...")
        self.result_text.insert(tk.END, "EggZayn: Model değerlendiriliyor...\n")
        self.root.update()
        start_time = time.time()
        try:
            loss, accuracy, report, cm = self.model.evaluate(self.X_test, self.Y_test)
            self.progress['value'] = 100
            self.result_text.insert(tk.END, f"EggZayn: Değerlendirme tamamlandı! Süre: {time.time() - start_time:.2f} saniye\n")
            self.result_text.insert(tk.END, f"\nTest Loss: {loss:.4f}\nTest Accuracy: {accuracy:.4f}\n\nClassification Report:\n{report}\n")
            
            self.ax2.clear()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.model.class_names, yticklabels=self.model.class_names, ax=self.ax2)
            self.ax2.set_title('EggZayn Confusion Matrix', color='white')
            self.ax2.set_xlabel('Predicted', color='white')
            self.ax2.set_ylabel('True', color='white')
            self.ax2.tick_params(colors='white')
            self.canvas.draw()
            
            self.status_label.config(text="EggZayn: Model hazır")
        except Exception as e:
            self.result_text.insert(tk.END, f"Hata: Değerlendirme başarısız: {e}\n")
            messagebox.showerror("Hata", f"EggZayn: Değerlendirme başarısız: {e}")

    def update_training_plot(self):
        if self.model.history:
            self.ax1.clear()
            self.ax1.plot(self.model.history.history['accuracy'], label='Training Accuracy', color='cyan')
            self.ax1.plot(self.model.history.history['val_accuracy'], label='Validation Accuracy', color='orange')
            self.ax1.plot(self.model.history.history['loss'], label='Training Loss', color='red')
            self.ax1.plot(self.model.history.history['val_loss'], label='Validation Loss', color='purple')
            self.ax1.set_title('EggZayn Training Metrics', color='white')
            self.ax1.set_xlabel('Epoch', color='white')
            self.ax1.set_ylabel('Value', color='white')
            self.ax1.legend(facecolor='#1A2526', edgecolor='white', loc='best', labelcolor='white')
            self.ax1.tick_params(colors='white')
            self.ax1.set_facecolor('#ECF0F1')
            self.canvas.draw()

    def predict_new_data(self):
        file_path = filedialog.askopenfilename(title="EEG Sinyal Dosyasını Seç (EDF veya NumPy)", 
                                               filetypes=[("EDF files", "*.edf"), ("NumPy files", "*.npy")])
        if not file_path:
            return
        
        self.status_label.config(text="EggZayn: Sinyal analiz ediliyor...")
        self.progress['value'] = 0
        self.root.update()
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "EggZayn: Sinyal analiz ediliyor...\n")
        self.root.update()
        try:
            predictions, report = self.model.predict(file_path)
            self.progress['value'] = 100
            self.result_text.insert(tk.END, report)
            
            pred_probs = np.array([prob for _, prob in predictions])
            self.ax1.clear()
            self.ax1.bar(self.model.class_names, pred_probs.mean(axis=0), color='skyblue', edgecolor='black')
            self.ax1.set_title('EggZayn Ortalama Tahmin Olasılıkları', color='white')
            self.ax1.set_ylabel('Olasılık', color='white')
            self.ax1.set_ylim(0, 1)
            self.ax1.tick_params(colors='white')
            self.ax1.set_facecolor('#ECF0F1')
            
            self.ax2.clear()
            self.canvas.draw()
            
            self.status_label.config(text="EggZayn: Analiz tamamlandı!")
        except Exception as e:
            self.result_text.insert(tk.END, f"Hata: Sinyal analizi başarısız: {e}\n")
            messagebox.showerror("Hata", f"EggZayn: Sinyal analizi başarısız: {e}")

    def run_full_process_thread(self):
        threading.Thread(target=self.full_process, daemon=True).start()

    def run_custom_train_thread(self):
        threading.Thread(target=self.custom_train, daemon=True).start()

if __name__ == '__main__':
    root = tk.Tk()
    app = EggZaynGUI(root)
    root.mainloop()