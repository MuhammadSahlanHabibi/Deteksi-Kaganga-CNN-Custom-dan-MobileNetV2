# 🔡 Deteksi Tulisan Kaganga — Aksara Komering Ulu

> **Tugas Mata Kuliah Computer Vision**
> Perbandingan CNN Custom vs Transfer Learning (MobileNetV2) untuk klasifikasi aksara tradisional Kaganga dari Komering Ulu, Sumatera Selatan.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MuhammadSahlanHabibi/Deteksi-Kaganga-CNN-Custom-dan-MobileNetV2/blob/main/Kaganga_Detection_Komering_Ulu.ipynb)

---

## 📋 Daftar Isi

- [Tentang Proyek](#-tentang-proyek)
- [Aksara Kaganga](#-aksara-kaganga)
- [Arsitektur Sistem](#-arsitektur-sistem)
- [Dataset](#-dataset)
- [Struktur Notebook](#-struktur-notebook)
- [Instalasi & Persyaratan](#-instalasi--persyaratan)
- [Cara Penggunaan](#-cara-penggunaan)
- [Detail Model](#-detail-model)
- [Pipeline Preprocessing](#-pipeline-preprocessing)
- [Evaluasi & Metrik](#-evaluasi--metrik)
- [Hasil Keluaran](#-hasil-keluaran)
- [Konfigurasi Hyperparameter](#-konfigurasi-hyperparameter)
- [Struktur Direktori](#-struktur-direktori)

---

## 🧐 Tentang Proyek

Proyek ini membangun sistem **deteksi/klasifikasi otomatis aksara Kaganga** dari wilayah Komering Ulu, Sumatera Selatan, menggunakan pendekatan Deep Learning berbasis gambar. Dua pendekatan diimplementasikan dan dibandingkan:

| Pendekatan | Deskripsi |
|---|---|
| **CNN Custom** | Arsitektur Convolutional Neural Network yang dibangun dari awal sebagai baseline |
| **Transfer Learning (MobileNetV2)** | Model MobileNetV2 pre-trained ImageNet dengan fine-tuning untuk aksara Kaganga |

Perbandingan ini bertujuan mengevaluasi efektivitas transfer learning dibandingkan model CNN buatan sendiri dalam konteks pengenalan karakter aksara daerah yang berjumlah ratusan kelas.

---

## 🔤 Aksara Kaganga

**Kaganga** adalah sistem aksara tradisional yang digunakan oleh masyarakat rumpun bahasa Rejang-Komering di Sumatera Selatan, termasuk wilayah Komering Ulu. Aksara ini termasuk dalam keluarga aksara Brahmi yang berkembang secara lokal di Nusantara. Proyek ini berkontribusi pada upaya **pelestarian dan digitalisasi** warisan budaya aksara daerah melalui teknologi pengenalan citra.

---

## 🏗️ Arsitektur Sistem

```
Input Gambar Aksara
        │
        ▼
┌─────────────────────┐
│   Preprocessing     │
│  - Resize 128x128   │
│  - Normalisasi [0,1]│
│  - Data Augmentasi  │
└─────────┬───────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
┌────────┐  ┌──────────────────┐
│  CNN   │  │  MobileNetV2 +   │
│ Custom │  │  Transfer Learn. │
└───┬────┘  └────────┬─────────┘
    │                │
    └─────┬──────────┘
          ▼
   ┌─────────────┐
   │  Evaluasi   │
   │ Acc/Pre/Rec │
   │ F1/ConfMat  │
   └─────────────┘
```

---

## 📦 Dataset

| Atribut | Detail |
|---|---|
| **Nama** | Dataset Komering Ulu |
| **Format** | `.zip` (berisi subfolder per kelas) |
| **Sumber** | Google Drive (`COMVIS CNN/dataset komering ulu.zip`) |
| **Jumlah Kelas** | **336 kelas** (336 karakter aksara Kaganga) |
| **Format Gambar** | PNG / JPG / JPEG |
| **Ukuran Target** | 128×128 piksel (di-resize otomatis) |

### Pembagian Dataset

```
Total Dataset
├── 70% → Training Set   (dengan augmentasi)
├── 15% → Validation Set
└── 15% → Test Set
```

Pembagian dilakukan secara **stratified** agar distribusi kelas tetap proporsional di setiap split. Struktur folder dataset dideteksi otomatis.

---

## 📓 Struktur Notebook

Notebook terdiri dari **11 bagian utama**:

```
1️⃣  Setup & Mount Google Drive
2️⃣  Import Library & Konfigurasi
3️⃣  Eksplorasi Dataset (EDA)
4️⃣  Preprocessing & Data Augmentation
5️⃣  Model 1 — CNN Custom (Baseline)
6️⃣  Model 2 — Transfer Learning (MobileNetV2)
7️⃣  Visualisasi Kurva Training
8️⃣  Evaluasi Model pada Test Set
9️⃣  Perbandingan Model
🔟  Prediksi & Visualisasi Contoh Gambar
1️⃣1️⃣ Simpan Model & Hasil ke Google Drive
```

---

## 🛠️ Instalasi & Persyaratan

### Library yang Digunakan

```python
tensorflow        # Deep learning framework utama
numpy             # Komputasi numerik
pandas            # Manipulasi data & tabel perbandingan
matplotlib        # Visualisasi grafik & kurva training
seaborn           # Visualisasi confusion matrix
scikit-learn      # Metrik evaluasi & stratified split
Pillow (PIL)      # Pemrosesan gambar
```

### Instalasi

```bash
pip install tensorflow matplotlib seaborn scikit-learn pillow
```

> **Catatan:** Notebook ini dirancang untuk berjalan di **Google Colab** dengan GPU. Disarankan menggunakan Runtime → Change runtime type → **GPU (T4 atau lebih tinggi)**.

---

## 🚀 Cara Penggunaan

### 1. Persiapan Dataset
Upload file `dataset komering ulu.zip` ke Google Drive pada path:
```
MyDrive/COMVIS CNN/dataset komering ulu.zip
```

### 2. Buka Notebook di Colab
Klik tombol **Open in Colab** di bagian atas, atau buka secara manual di [Google Colab](https://colab.research.google.com/).

### 3. Aktifkan GPU
```
Runtime → Change runtime type → Hardware accelerator: GPU
```

### 4. Jalankan Sel Secara Berurutan
Jalankan semua sel dari atas ke bawah. Setiap bagian dirancang berjalan secara sekuensial.

### 5. Hasil Otomatis Tersimpan
Model dan grafik hasil evaluasi otomatis disimpan ke:
```
MyDrive/COMPUTER VISION/hasil_kaganga/
```

---

## 🧠 Detail Model

### Model 1 — CNN Custom

Arsitektur CNN yang dibangun dari awal dengan 4 blok konvolusi:

```
Input (128×128×3)
│
├── Block 1: Conv2D(32) → BN → Conv2D(32) → MaxPool → Dropout(0.25)
├── Block 2: Conv2D(64) → BN → Conv2D(64) → MaxPool → Dropout(0.25)
├── Block 3: Conv2D(128) → BN → Conv2D(128) → MaxPool → Dropout(0.25)
├── Block 4: Conv2D(256) → BN → GlobalAveragePooling2D
│
└── Classifier:
    Dense(512, relu) → BN → Dropout(0.5)
    → Dense(256, relu) → Dropout(0.3)
    → Dense(336, softmax)
```

**Konfigurasi Training:**
- Optimizer: Adam (lr = 1e-3)
- Loss: Sparse Categorical Crossentropy
- Epochs: 25
- Early Stopping: patience=7, monitor=val_accuracy
- ReduceLROnPlateau: factor=0.5, patience=3

---

### Model 2 — MobileNetV2 (Transfer Learning)

Strategi dua fase:

**Phase 1 — Train Classifier Head (10 epoch):**
```
MobileNetV2 (ImageNet, FROZEN)
│
└── GlobalAveragePooling2D
    → Dense(512, relu) → BN → Dropout(0.4)
    → Dense(256, relu) → Dropout(0.3)
    → Dense(336, softmax)
```

**Phase 2 — Fine-tuning (20 epoch):**
- 50 layer terakhir base model di-unfreeze
- Layer-layer awal tetap frozen (fitur umum dipertahankan)
- Learning rate dikurangi 10× (LR / 10) untuk stabilitas

---

## ⚙️ Pipeline Preprocessing

### Normalisasi
Seluruh gambar di-resize ke **128×128 piksel** dan dinormalisasi ke rentang `[0, 1]`:
```python
img = tf.image.resize(img, [128, 128])
img = tf.cast(img, tf.float32) / 255.0
```

### Data Augmentasi (Training Only)

| Augmentasi | Parameter |
|---|---|
| Random Horizontal Flip | — |
| Random Brightness | max_delta = 0.15 |
| Random Contrast | lower=0.8, upper=1.2 |
| Random Saturation | lower=0.8, upper=1.2 |
| Random Crop | Padding +10px lalu crop ke 128×128 |

### TF Data Pipeline
Dataset menggunakan `tf.data` dengan:
- `shuffle()` pada training set
- `map()` dengan `num_parallel_calls=AUTOTUNE`
- `batch(32)`
- `prefetch(AUTOTUNE)` untuk efisiensi I/O

---

## 📊 Evaluasi & Metrik

Evaluasi dilakukan pada **test set yang belum pernah dilihat model** menggunakan 4 metrik:

| Metrik | Formula | Keterangan |
|---|---|---|
| **Accuracy** | TP+TN / Total | Proporsi prediksi benar secara keseluruhan |
| **Precision** | TP / (TP+FP) | Ketepatan prediksi positif (weighted avg) |
| **Recall** | TP / (TP+FN) | Kelengkapan deteksi kelas benar (weighted avg) |
| **F1-Score** | 2×(P×R)/(P+R) | Harmonic mean precision & recall |

Selain itu, **Confusion Matrix** (336×336) divisualisasikan sebagai heatmap warna untuk mendeteksi pola kesalahan antar kelas aksara.

**Classification Report** lengkap per kelas juga ditampilkan untuk analisis per karakter.

---

## 📁 Hasil Keluaran

Setelah training selesai, file berikut disimpan ke Google Drive:

```
MyDrive/COMPUTER VISION/hasil_kaganga/
├── model_cnn_custom.h5          # Model CNN Custom tersimpan
├── model_mobilenetv2_tl.h5      # Model MobileNetV2 TL tersimpan
├── hasil_evaluasi.json          # Ringkasan metrik (JSON)
├── distribusi_dataset.png       # Bar chart & pie chart distribusi kelas
├── sampel_dataset.png           # Contoh gambar per kelas aksara
├── augmentasi.png               # Visualisasi efek augmentasi
├── training_cnn.png             # Kurva accuracy & loss CNN Custom
├── training_tl.png              # Kurva accuracy & loss MobileNetV2 TL
├── cm_cnn.png                   # Confusion matrix CNN Custom
├── cm_tl.png                    # Confusion matrix MobileNetV2 TL
└── perbandingan_model.png       # Bar chart + radar chart perbandingan
```

### Contoh Isi `hasil_evaluasi.json`

```json
{
  "timestamp": "2024-xx-xxTxx:xx:xx",
  "dataset": {
    "total_images": <N>,
    "num_classes": 336,
    "train_size": <N>,
    "val_size": <N>,
    "test_size": <N>
  },
  "results": {
    "cnn_custom": {
      "accuracy": 0.xxxx,
      "precision": 0.xxxx,
      "recall": 0.xxxx,
      "f1_score": 0.xxxx
    },
    "mobilenetv2_tl": {
      "accuracy": 0.xxxx,
      "precision": 0.xxxx,
      "recall": 0.xxxx,
      "f1_score": 0.xxxx
    }
  },
  "best_model": "MobileNetV2 TL"
}
```

---

## ⚙️ Konfigurasi Hyperparameter

Semua hyperparameter utama terpusat di bagian **Konfigurasi** pada sel awal:

```python
IMG_SIZE    = 128    # Ukuran input gambar (128×128 piksel)
BATCH_SIZE  = 32     # Jumlah gambar per batch
EPOCHS_CNN  = 25     # Maksimum epoch CNN Custom
EPOCHS_TL   = 20     # Maksimum epoch fine-tuning MobileNetV2
LR          = 1e-3   # Learning rate awal
VAL_SPLIT   = 0.15   # 15% untuk validasi
TEST_SPLIT  = 0.15   # 15% untuk pengujian
SEED        = 42     # Seed untuk reproducibility
```

---

## 📂 Struktur Direktori (Lokal Colab)

```
/content/
├── dataset_kaganga/             # Dataset yang diekstrak
│   ├── <kelas_1>/               # Subfolder tiap aksara Kaganga
│   ├── <kelas_2>/
│   └── ...                      # (336 kelas total)
├── cnn_best.h5                  # Checkpoint CNN terbaik (sementara)
├── tl_best.h5                   # Checkpoint TL terbaik (sementara)
├── distribusi_dataset.png
├── sampel_dataset.png
├── augmentasi.png
├── training_cnn.png
├── training_tl.png
├── cm_cnn.png
├── cm_tl.png
└── perbandingan_model.png
```

---

## 🏆 Kesimpulan

Notebook ini mengimplementasikan perbandingan **dua pendekatan deep learning** untuk klasifikasi 336 kelas aksara Kaganga Komering Ulu. Model terbaik dipilih otomatis berdasarkan akurasi pada test set dan divisualisasikan dalam ringkasan akhir berbentuk tabel ASCII. Secara umum, Transfer Learning (MobileNetV2) diharapkan menghasilkan akurasi lebih tinggi karena memanfaatkan fitur visual yang telah dipelajari dari jutaan gambar ImageNet, terutama pada dataset dengan jumlah kelas yang sangat banyak (336 kelas).

---

## 👤 Penulis

**Ghazi Al-Ghifari G1A023053, Muhammad Sahlan Habibi G1A023058, Ricardo Gellael G1A023061**
Tugas Mata Kuliah Computer Vision

---

## 📜 Lisensi

Proyek ini dibuat untuk keperluan akademik. Dataset aksara Kaganga adalah kekayaan budaya masyarakat Komering Ulu, Sumatera Selatan.
