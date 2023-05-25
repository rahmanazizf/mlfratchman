# K-Nearest Neighbor (KNN)
Salah satu proyek dari course Advanced ML di Pacmann Academy
## Tentang KNN
KNN merupakan salah satu algoritma unsupervised learning yang memiliki cara kerja yang cukup sederhana. Termasuk ke dalam model nonparametrik, KNN tidak menghasilkan nilai parameter sebagaimana model parametrik seperti regresi linear. Dalam regresi linear, tujuan kita adalah mencari parameter (koefisien dan konstanta) yang digunakan untuk mengaproksimasi true function.

$$ 
y = wx + b 
$$

Ketika membuat model regresi linear, tugas kita adalah mencari nilai w dan b dengan galat sekecil mungkin dari data y dan x yang diketahui dari data training. Tetapi dalam KNN tidak demikian. Kita hanya perlu mencari sejumlah k titik terdekat terhadap titik data yang ingin kita prediksi (target) lalu membandingkan label/output tetangga target tersebut.

![image](https://github.com/rahmanazizf/mlfratchman/assets/100136072/ba64029e-f951-47c4-af19-37d33e346c8e)

$$
d_{(i, j)} = ({{\sum_{k = 1}^{N}|x_k^{(i)} - x_k^{(i)}|^p}})^{\frac{1}{p}}
$$

Persamaan di atas adalah persamaan umum untuk mencari jarak antara dua titik dengan persamaan Minkowski, Manhattan, dan Euclidean, di mana d merupakan jarak (distance), N merupakan jumlah fitur dan p merupakan pangkat (power). Persamaan Manhattan dapat diperoleh dengan mensubstitusi p dengan 1, Euclidean p = 2 dan selain itu menjadi persamaan Minkowski.
## Struktur Modul
![image](https://github.com/rahmanazizf/mlfratchman/assets/100136072/85658ac9-a1d5-4593-b94f-f997a6742c80)
## Algoritma
![Untitled](https://github.com/rahmanazizf/mlfratchman/assets/100136072/6f492e40-11af-4999-a351-70222256a646)
