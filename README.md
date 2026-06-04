# Deteksi-Kaganga-CNN-Custom-dan-MobileNetV2
Ghazi Al-Ghifari G1A023053, Muhammad Sahlan Habibi G1A023058, Ricardo Gellael G1A023061

# 🔡 Deteksi Tulisan Kaganga — Aksara Komering Ulu Menggunakan Deep Learning

> Tugas Mata Kuliah **Computer Vision** — Semester 6 Teknik Informatika Universitas Bengkulu  
> Membandingkan arsitektur Convolutional Neural Network (CNN) Custom dan Transfer Learning (MobileNetV2)

---

## 📋 Daftar Isi

- [Deskripsi Proyek](#-deskripsi-proyek)
- [Dataset](#-dataset)
- [Struktur Proyek](#-struktur-proyek)
- [Instalasi & Setup](#-instalasi--setup)
- [Pipeline](#-pipeline)
- [Konfigurasi & Arsitektur](#-konfigurasi--arsitektur)
- [Model yang Digunakan](#-model-yang-digunakan)
- [Metrik Evaluasi](#-metrik-evaluasi)
- [Cara Menjalankan](#-cara-menjalankan)
- [Teknologi yang Digunakan](#-teknologi-yang-digunakan)

---

## 📌 Deskripsi Proyek

Proyek ini mengimplementasikan sistem klasifikasi citra menggunakan teknik **Deep Learning** berbasis **Computer Vision** untuk mendeteksi dan mengklasifikasikan aksara tulisan Kaganga, khususnya varian Aksara Komering Ulu. 

Mengingat dataset yang sangat masif dan jumlah kelas yang banyak, proyek ini menggunakan dan membandingkan dua buah pendekatan utama untuk menemukan performa pengenalan aksara yang paling optimal.

---

## 📦 Dataset

| Properti | Detail |
|----------|--------|
| **Sumber** | Google Drive (`dataset komering ulu.zip`) |
| **Jumlah Kelas** | 336 kategori kelas aksara |
| **Total Gambar** | 21.150 gambar |
| **Distribusi** | Bervariasi / *Imbalanced* (min: kelas 'a' 51 gambar, max: kelas 'nggar' 163 gambar) |
| **Split** | Training 70% : Validation 15% : Testing 15% |

---

## 🗂 Struktur Proyek

```text
kaganga-classification/
│
├── dataset_kaganga/               # Hasil ekstraksi dataset
│   ├── a/
│   ├── ba/
│   ├── ...
│   └── nggar/
│
├── models/                        # Folder penyimpanan bobot model
│   ├── cnn_custom_model.h5
│   └── mobilenetv2_model.h5
│
├── output/                        # Visualisasi hasil evaluasi
│   ├── class_distribution.png
│   ├── accuracy_loss_curve.png
│   └── confusion_matrix.png
│
└── README.md

# ⚙️ Instalasi & Setup
1. Buka Google Colab dan Mount Google Drive
Pastikan file dataset komering ulu.zip sudah berada di Google Drive Anda.
