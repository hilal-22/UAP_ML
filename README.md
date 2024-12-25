Meat Consumption Analysis and Prediction
Deskripsi Proyek
Meat Consumption Analysis and Prediction adalah sebuah proyek yang bertujuan untuk menganalisis pola konsumsi daging berdasarkan data historis dan membuat prediksi konsumsi di masa depan. Proyek ini mencakup eksplorasi data, identifikasi tren, dan faktor yang memengaruhi konsumsi daging, seperti perubahan harga, preferensi konsumen, dan kebijakan pemerintah. Selain itu, model prediksi menggunakan teknik pembelajaran mesin dirancang untuk memperkirakan konsumsi di berbagai wilayah atau kelompok demografis. Hasil proyek ini dapat digunakan untuk mendukung keputusan dalam industri pangan, perencanaan kebijakan, dan keberlanjutan lingkungan.

Tujuan Pengembangan
Menyediakan alat bantu diagnosis penyakit hewan ternak yang cepat dan akurat.
Meningkatkan efisiensi dalam pengambilan keputusan terkait kesehatan hewan ternak.
Menggunakan machine learning untuk mempelajari pola gejala penyakit dan prediksi penyakit berdasarkan data yang dikumpulkan.
Langkah Instalasi
Prasyarat
Sebelum memulai, pastikan bahwa Anda telah menginstal Python 3.7 atau lebih baru. Anda juga memerlukan pip untuk mengelola dependensi.

1. Instalasi Dependensi
Clone repositori ini ke dalam direktori lokal Anda:

git clone https://github.com/username/repository.git
cd repository
Instalasi dependensi Python yang diperlukan:

pip install -r requirements.txt
File requirements.txt harus mencakup, di antaranya:

pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
keras
tensorflow
streamlit
joblib
2. Menjalankan Aplikasi Web
Untuk menjalankan aplikasi web menggunakan Streamlit, jalankan perintah berikut:

streamlit run app.py
Aplikasi web akan terbuka di browser Anda. Anda dapat mengaksesnya melalui URL yang tertera di terminal.

Deskripsi Model
Model yang Digunakan
Random Forest Classifier
Model ini digunakan untuk klasifikasi penyakit hewan ternak berdasarkan gejala yang terlihat. Random Forest merupakan algoritma ensemble yang memanfaatkan banyak pohon keputusan untuk meningkatkan akurasi prediksi.

XGBoost Classifier
XGBoost adalah algoritma gradient boosting yang terkenal dengan performanya dalam berbagai kompetisi machine learning. Model ini digunakan untuk meningkatkan akurasi dan mengatasi masalah overfitting yang mungkin terjadi pada model lain.

Feedforward Neural Network (FFNN)
Neural Network digunakan untuk membangun model klasifikasi yang lebih kompleks dengan memanfaatkan banyak lapisan neuron. Model ini dilatih menggunakan dataset gejala penyakit pada hewan ternak.

Analisis Performa Model
Model yang digunakan diuji menggunakan dataset yang mencakup gejala penyakit dan klasifikasi penyakit hewan ternak. Akurasi dari setiap model dievaluasi dengan menggunakan metrik accuracy, precision, recall, dan f1-score.

Berikut adalah hasil evaluasi model yang digunakan:

Random Forest Classifier
Akurasi: 97.60%
Classification Report:
Precision:  0.97 
Recall:  0.98  
F1-Score: 0.98
XGBoost Classifier
Akurasi:  98.26%
Classification Report:
Precision: 0.98  
Recall: 0.98  
F1-Score: 0.98  
Feedforward Neural Network (FFNN)
Akurasi: 89.32%
Classification Report:
Precision:  0.89
Recall:  0.89
F1-Score:  0.89
