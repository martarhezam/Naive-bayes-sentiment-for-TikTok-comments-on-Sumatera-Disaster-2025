# Naive-bayes-sentiment-for-TikTok-comments-on-Sumatera-Disaster-2025

Projek ini menganalisis sentimen dari komentar TikTok terkait bencana banjir di Sumatera. Alur sistemnya menggunakan pre-processing kata secara keseluruhan, **Indonesian RoBERTa Base Sentiment Classifier** untuk labeling otomatis, **TF-IDF** untuk nilai kata yang muncul dan **Multinomial Naive Bayes** sebagai model klasifikasi. Dari total sekitar 1100 data, 218 sampel digunakan sebagai data testing.

#### **Analisis Hasil & Kemungkinan Masalah yg terjadi**
Secara keseluruhan, model dapet **Akurasi ~69.7%**. Tapi kalau saya bedah lewat Confusion Matrix di folder HASIL_CM/confusion_matrix.png , ada beberapa hal menarik:
*   **Efek Topik Bencana:** Karena topiknya adalah banjir, wajar kalau teksnya didominasi kata-kata keluhan atau duka. Hasilnya, **IndoBERT** lebih banyak melabeli data ke arah `negative` (115 sampel di data uji).
*   **Karakteristik Naive Bayes:** Algoritma Naive Bayes itu sangat bergantung pada frekuensi kata. Karena data negatif melimpah, model jadi bias dan sering main aman dengan nebak "negatif". Makanya, Recall kelas negatif tinggi banget (**92.1%** / 106 dari 115 data sukses ditebak).
*   **Sentimen Netral yang Terabaikan:** Dari 51 data asli netral, **34 di antaranya malah ditebak negatif**. Komentar TikTok itu penuh bahasa gaul, singkatan, atau sarkasme walaupun sudah saya coba ubah ke bahasa formal menggunakan bag of words yg ada kemungkinan masih ada bahasa atau singkatan yg tidak berubah. Kombinasi TF-IDF (yang berbasis kemunculan kata kunci) dan Naive Bayes kesulitan membedakan mana kalimat netral dan mana yang negatif.

#### **Solusi Berikutnya yang Bisa Dicoba**
1.  **Mengimbangkan Fitur TF-IDF:** Gunakan teknik oversampling seperti SMOTE pada hasil fitur TF-IDF khusus untuk kelas `neutral` dan `positive` sebelum masuk ke tahap training Naive Bayes.
2.  **Atur Parameter Naive Bayes:** Coba utak-atik parameter smoothing (`alpha`) atau matikan prior probability otomatis dengan set parameter `fit_prior=False` di Scikit-Learn biar model gak terlalu condong ke kelas mayoritas.

Thanks for reading.
