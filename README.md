# K-Nearest Neighbor (KNN)
Salah satu proyek dari course Advanced ML di Pacmann Academy
## Tentang KNN
KNN merupakan salah satu algoritma unsupervised learning yang memiliki cara kerja yang cukup sederhana. Termasuk ke dalam model nonparametrik, KNN tidak menghasilkan nilai parameter sebagaimana model parametrik seperti regresi linear. Dalam regresi linear, tujuan kita adalah mencari parameter (koefisien dan konstanta) yang digunakan untuk mengaproksimasi true function.

$$ 
y = wx + b 
$$

Ketika membuat model regresi linear, tugas kita adalah mencari nilai w dan b dengan galat sekecil mungkin dari data y dan x yang diketahui dari data training. Tetapi dalam KNN tidak demikian. Kita hanya perlu mencari sejumlah k titik terdekat terhadap titik data yang ingin kita prediksi (target) lalu membandingkan label/output tetangga target tersebut.
init commit yang salah

![image](https://github.com/rahmanazizf/mlfratchman/assets/100136072/6a8d92a6-c772-4beb-aa2a-d0edb367e178)

$$
d_{(i, j)} = ({{\sum_{k = 1}^{N}|x_k^{(i)} - x_k^{(i)}|^p}})^{\frac{1}{p}}
$$

Persamaan di atas adalah persamaan umum untuk mencari jarak antara dua titik dengan persamaan Minkowski, Manhattan, dan Euclidean, di mana d merupakan jarak (distance), N merupakan jumlah fitur dan p merupakan pangkat (power). Persamaan Manhattan dapat diperoleh dengan mensubstitusi p dengan 1, Euclidean p = 2 dan selain itu menjadi persamaan Minkowski.

## Struktur Modul
![image](https://github.com/rahmanazizf/mlfratchman/assets/100136072/85658ac9-a1d5-4593-b94f-f997a6742c80)
## Algoritma
![Untitled](https://github.com/rahmanazizf/mlfratchman/assets/100136072/6f492e40-11af-4999-a351-70222256a646)
Diberikan suatu data dengan ukuran tertentu. Algoritma KNN pertama akan menghitung jarak data yang ingin kita prediksi ke setiap titik data yang telah diketahui kelas/labelnya (data train). Setelah jarak antara titik data target ke setiap titik data train didapatkan, kita perlu menentukan sebanyak k titik data terdekat terhadap titik data target. Dari sebanyak k tetangga yang kita sudah tentukan sebelumnya, kita dapat mengekstrak indeks baris dari data tetangga tersebut dan menggunakannya untuk mendapatkan nilai kelas/label dari baris data yang bersesuaian. Terakhir lalukan prediksi dan sesuaikan metode prediksi dengan kasus yang ingin kita selesaikan. Jika kita ingin melakukan klasifikasi, lakukan prediksi dengan majority vote (probabilitas). Namun jika ingin melakukan regresi, gunakan perhitungan rata-rata. Perhitungan KNN dilakukan secara iteratif sampai baris data terakhir.
### KNN Classifier
Metode prediksi KNN classifier menggunakan perhitungan probabilitas. Argumen kelas yang menghasilkan nilai probabilitas tertinggi akan menjadi nilai prediksi kelas titik data target.
![image](https://github.com/rahmanazizf/mlfratchman/assets/100136072/22d801fd-572f-4576-b52b-fe5f1178e92e)
### KNN Regressor
Berbeda dengan KNN Classifer, perhitungan untuk prediksi kelas pada KNN regressor lebih sederhana. Nilai kelas/label prediksi dihitung dengan cara merata-ratakan nilai kelas/label dari setiap tetangga terdekat.
![image](https://github.com/rahmanazizf/mlfratchman/assets/100136072/d1112391-f553-4497-85b9-0978a2f8aa61)
