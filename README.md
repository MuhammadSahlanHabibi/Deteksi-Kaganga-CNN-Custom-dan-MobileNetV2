# Deteksi-Kaganga-CNN-Custom-dan-MobileNetV2
Ghazi Al-Ghifari G1A023053, Muhammad Sahlan Habibi G1A023058, Ricardo Gellael G1A023061

🔡 Deteksi Tulisan Kaganga — Aksara Komering Ulu
📌 Deskripsi Proyek
Proyek ini bertujuan untuk mendeteksi dan mengklasifikasikan aksara tulisan Kaganga, khususnya varian Aksara Komering Ulu. Pembuatan model dalam proyek ini menggunakan dan membandingkan dua buah pendekatan, yaitu teknik Transfer Learning menggunakan arsitektur MobileNetV2 dan membangun model Convolutional Neural Network (CNN) Custom.  
IPYNB
+ 1

📁 Informasi Dataset
Dataset diekstrak di Google Colab dari file arsip bernama dataset komering ulu.zip yang terhubung melalui Google Drive.  
IPYNB

Total Gambar: 21.150 gambar data aksara.  
IPYNB

Total Kelas: 336 kategori kelas aksara yang berbeda.  
IPYNB

Distribusi Kelas: Jumlah data cukup bervariasi pada tiap kelasnya (contoh: kelas 'a' memiliki 51 gambar, kelas 'ar' memiliki 141 gambar, hingga kelas 'nggar' memiliki 163 gambar).  
IPYNB

⚙️ Konfigurasi & Hyperparameter Model
Proyek ini diatur menggunakan sejumlah nilai hyperparameter berikut:

Ukuran Gambar (IMG_SIZE): 128x128 pixel.  
IPYNB

Batch Size: 32.  
IPYNB

Epoch (Model CNN Custom): 25 Epoch.  
IPYNB

Epoch (Transfer Learning): 20 Epoch.  
IPYNB

Learning Rate (LR): 1e-3.  
IPYNB

Validation Split: 15% dari total dataset.  
IPYNB

Test Split: 15% dari total dataset.  
IPYNB

SEED: 42 (Digunakan untuk memastikan reproduktibilitas hasil model).  
IPYNB

📊 Metrik Evaluasi
Untuk menilai dan mengukur performa pengenalan aksara, model ini dievaluasi menggunakan metrik standar klasifikasi berikut:

Accuracy.  
IPYNB

Precision.  
IPYNB

Recall.  
IPYNB

F1-Score.  
IPYNB

Confusion Matrix.  
IPYNB

🛠️ Teknologi & Library yang Digunakan
Proyek ini berjalan pada environment Python 3.10 dengan dukungan GPU (Tesla T4) dan memanfaatkan beberapa library utama:

TensorFlow (v2.20.0) / Keras: Digunakan sebagai framework utama dalam membangun arsitektur model deep learning dan prapemrosesan gambar (ImageDataGenerator).  
IPYNB

Scikit-learn: Digunakan untuk memisahkan dataset dan mengkalkulasi metrik klasifikasi.  
IPYNB

Matplotlib & Seaborn: Digunakan untuk melakukan plotting dan visualisasi data.  
IPYNB

Pandas & Numpy: Digunakan untuk struktur data dan operasi matematika berbasis matriks/array.  
IPYNB

Pillow (PIL): Digunakan untuk membaca dan memanipulasi format gambar.  
IPYNB

🚀 Tahapan Pelaksanaan (Pipeline Proyek)
Setup & Mount Google Drive: Melakukan instalasi library eksternal tambahan (jika dibutuhkan), melakukan perizinan akses Google Drive, serta melakukan ekstraksi file dataset aksara berformat ZIP ke dalam direktori lokal Colab (/content/dataset_kaganga).  
IPYNB

Import Library & Konfigurasi: Mengimpor seluruh dependensi program, mengecek ketersediaan modul eksekusi GPU pada perangkat, dan melakukan deklarasi hyperparameter pelatihan.  
IPYNB

Eksplorasi Dataset (EDA): Menggunakan skrip otomatis untuk mendeteksi letak root folder gambar aksara, serta memuat kalkulasi untuk menghitung ringkasan jumlah kelas, total gambar keseluruhan, beserta grafik distribusi jumlah dataset yang tersedia untuk tiap kelasnya.  
IPYNB
