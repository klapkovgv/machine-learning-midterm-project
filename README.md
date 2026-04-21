# machine-learning-midterm-project

Bu ara sınav ödevi kapsamında, scikit-learn kütüphanesinde yer alan 'Wine Classification' veri seti üzerinde uçtan uca bir makine öğrenmesi süreci gerçekleştirilmiştir. Çalışma boyunca aşağıdaki temel adımlar takip edilmiştir:
1. Veri Ön İşleme: Ham veriler üzerinde eksik değer ve aykırı değer (outlier) analizleri yapılmış.
2. Boyut İndirgeme: Veri setinin karmaşıklığını azaltmak ve sınıf ayrımını optimize etmek amacıyla PCA (Temel Bileşen Analizi) ve LDA (Doğrusal Ayrıştırma Analizi) teknikleri uygulanmıştır.
3. Model Eğitimi: Klasik makine öğrenmesi algoritmaları (Lojistik Regresyon, Karar Ağaçları, Random Forest, XGBoost ve Naive Bayes) kullanılarak 3 farklı veri temsili üzerinde toplamda 15 farklı sınıflandırma modeli kurulmuş ve performansları karşılaştırılmıştır.
4. XAI (Açıklanabilir Yapay Zeka): En iyi performansı gösteren modeller, SHAP analizleri kullanılarak yorumlanmış; modelin karar verme süreçleri ve hangi özelliklerin tahminlerde daha etkili olduğu şeffaf bir şekilde analiz edilmiştir.

## 1. Veri Setinin Yüklenmesi 

Bu bölümde, Scikit-learn kütüphanesinde hazır olarak sunulan "Wine Classification" veri seti yüklenmiş, bağımsız değişkenler (X) ve hedef değişken (y) ayrıştırılarak bir pandas DataFrame yapısı oluşturulmuştur.

Veri setinin yapısını ve içerdiği değerleri anlamak amacıyla ilk 5 satır aşağıda tablo olarak sunulmuştur:
| alcohol | malic_acid | ash | alcalinity_of_ash | magnesium | total_phenols | flavanoids | nonflavanoid_phenols | proanthocyanins | color_intensity | hue | od280/od315 | proline | target |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 14.23 | 1.71 | 2.43 | 15.6 | 127.0 | 2.80 | 3.06 | 0.28 | 2.29 | 5.64 | 1.04 | 3.92 | 1065.0 | 0 |
| 13.20 | 1.78 | 2.14 | 11.2 | 100.0 | 2.65 | 2.76 | 0.26 | 1.28 | 4.38 | 1.05 | 3.40 | 1050.0 | 0 |
| 13.16 | 2.36 | 2.67 | 18.6 | 101.0 | 2.80 | 3.24 | 0.30 | 2.81 | 5.68 | 1.03 | 3.17 | 1185.0 | 0 |
| 14.37 | 1.95 | 2.50 | 16.8 | 113.0 | 3.85 | 3.49 | 0.24 | 2.18 | 7.80 | 0.86 | 3.45 | 1480.0 | 0 |
| 13.24 | 2.59 | 2.87 | 21.0 | 118.0 | 2.80 | 2.69 | 0.39 | 1.82 | 4.32 | 1.04 | 2.93 | 735.0 | 0 |

Veri setindeki her bir sütunun içerdiği özgün (unique) değer sayıları incelenmiştir. Bu inceleme, hangi değişkenlerin sürekli hangilerinin kategorik olduğu hakkında bilgi vermektedir:
| Özellik (Feature) | Özgün Değer Sayısı (Unique Counts) |
| :--- | :--- |
| alcohol | 126 |
| malic_acid | 133 |
| ash | 79 |
| alcalinity_of_ash | 63 |
| magnesium | 53 |
| total_phenols | 97 |
| flavanoids | 132 |
| nonflavanoid_phenols | 39 |
| proanthocyanins | 101 |
| color_intensity | 132 |
| hue | 78 |
| od280/od315_of_diluted_wines | 122 |
| proline | 121 |
| **target (Hedef)** | **3** |


Veri seti 13 adet sürekli sayısal değişkenden ve 3 farklı sınıfa sahip bir hedef değişkenden oluşmaktadır. Özgün değer sayılarının yüksek olması, verilerin hassas ölçümler içerdiğini göstermektedir. Hedef değişkendeki 3 özgün değer, çalışmanın bir "çok sınıflı sınıflandırma" (multiclass classification) problemi olduğunu teyit etmektedir.

## 2. Veri Seti Kalite Kontrolleri

Bu aşamada veri setinin güvenilirliğini sağlamak amacıyla eksik değer, aykırı değer ve veri tipleri üzerinde detaylı kontroller yapılmıştır.

Veri setindeki her bir sütun için doluluk oranları incelenmiştir. Yapılan df.info() sorgusu sonucunda elde edilen bilgiler aşağıdadır:
| # | Column | Non-Null Count | Dtype |
|---|-------------------|--------------------------------------|------------|
| 0 | alcohol | 178 non-null | float64 |
| 1 | malic_acid | 178 non-null | float64 |
| 2 | ash | 178 non-null | float64 |
| 3 | alcalinity_of_ash | 178 non-null | float64 |
| 4 | magnesium | 178 non-null | float64 |
| 5 | total_phenols | 178 non-null | float64 |
| 6 | flavanoids | 178 non-null | float64 |
| 7 | nonflavanoid_phenols | 178 non-null | float64 |
| 8 | proanthocyanins | 178 non-null | float64 |
| 9 | color_intensity | 178 non-null | float64 |
| 10 | hue | 178 non-null | float64 |
| 11 | od280/od315_of_diluted_wines | 178 non-null | float64 |
| 12 | proline | 178 non-null | float64 |
| 13 | target | 178 non-null | int64 |

Sonuç: Veri setinde hiçbir sütunda eksik değer (missing value) bulunmamaktadır. Bu nedenle veri tamamlama veya silme işlemi yapılmasına gerek duyulmamıştır.

IQR (Interquartile Range) yöntemi kullanılarak yapılan analiz sonucunda, verilerin normal dağılım sınırları dışında kalan gözlem sayıları belirlenmiştir:
| Özellik (Feature) | Aykırı Değer Sayısı |
| :--- | :--- |
| malic_acid | 3 |
| ash | 3 |
| alcalinity_of_ash | 4 |
| magnesium | 4 |
| proanthocyanins | 2 |
| color_intensity | 4 |
| hue | 1 |

Aykırı değerler, özellikle Lojistik Regresyon ve PCA gibi varyans tabanlı algoritmaların katsayılarını saptırabilir ve modelin hatalı öğrenmesine yol açabilir. Bu değerlerin yarattığı olumsuz etkiyi minimize etmek amacıyla, çalışmanın ilerleyen safhalarında medyan tabanlı ölçeklendirme yapan RobustScaler yöntemi tercih edilmiştir.

Veri setindeki değişkenlerin programlama tarafındaki veri tipleri kontrol edilmiştir:
* Sayısal Değişkenler (13 adet): alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins, color_intensity, hue, od280/od315, proline — Tümü float64 tipindedir.
* Hedef Değişken (1 adet): target — int64 tipindedir.

Veri setinin tamlığı ve veri tiplerinin homojenliği (tüm özelliklerin sayısal olması), modellerin kurulması için uygun bir zemin hazırlamaktadır. Aykırı değerlerin varlığı tespit edilmiş olup, bu durumun model performansını düşürmemesi için uygun ölçeklendirme stratejisi belirlenmiştir.

## 3. Keşifsel Veri Analizi (EDA)

Bu aşamada verilerin istatistiksel dağılımları, değişkenler arası ilişkiler ve aykırı değer eğilimleri incelenmiştir.

Veri setindeki değişkenlerin merkezi eğilim ve dağılım ölçüleri df.describe() metodu ile analiz edilmiştir:
|index|alcohol|malic\_acid|ash|alcalinity\_of\_ash|magnesium|total\_phenols|flavanoids|nonflavanoid\_phenols|proanthocyanins|color\_intensity|hue|od280/od315\_of\_diluted\_wines|proline|target|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|count|178\.0|178\.0|178\.0|178\.0|178\.0|178\.0|178\.0|178\.0|178\.0|178\.0|178\.0|178\.0|178\.0|178\.0|
|mean|13\.00061797752809|2\.3363483146067416|2\.3665168539325845|19\.49494382022472|99\.74157303370787|2\.295112359550562|2\.0292696629213487|0\.3618539325842696|1\.5908988764044945|5\.058089882022472|0\.9574494382022471|2\.6116853932584267|746\.8932584269663|0\.9382022471910112|
|std|0\.8118265380058577|1\.1171460976144627|0\.2743440090608148|3\.3395637671735052|14\.282483515295668|0\.6258510488339891|0\.9988586850169465|0\.12445334029667939|0\.5723588626747611|2\.318285871822413|0\.22857156582982338|0\.7099904287650505|314\.9074742768489|0\.7750349899850565|
|min|11\.03|0\.74|1\.36|10\.6|70\.0|0\.98|0\.34|0\.13|0\.41|1\.28|0\.48|1\.27|278\.0|0\.0|
|25%|12\.362499999999999|1\.6025|2\.21|17\.2|88\.0|1\.7425|1\.205|0\.27|1\.25|3\.2199999999999998|0\.7825|1\.9375|500\.5|0\.0|
|50%|13\.05|1\.8650000000000002|2\.36|19\.5|98\.0|2\.355|2\.135|0\.34|1\.5550000000000002|4\.6899999999999995|0\.965|2\.78|673\.5|1\.0|
|75%|13\.6775|3\.0825|2\.5575|21\.5|107\.0|2\.8|2\.875|0\.4375|1\.95|6\.2|1\.12|3\.17|985\.0|2\.0|
|max|14\.83|5\.8|3\.23|30\.0|162\.0|3\.88|5\.08|0\.66|3\.58|13\.0|1\.71|4\.0|1680\.0|2\.0|

Veri setinin tanımlayıcı istatistikleri incelendiğinde, özelliklerin çok farklı ölçeklerde olduğu görülmektedir. Örneğin, proline değişkeni 278 ile 1680 arasında değerler alırken, nonflavanoid_phenols 0.13 ile 0.66 arasında değişmektedir. Bu durum, mesafe tabanlı algoritmaların (PCA, LDA, Lojistik Regresyon) hatalı ağırlıklandırma yapmaması için standartlaştırma işleminin zorunlu olduğunu kanıtlamaktadır.

### Korelasyon Matrisi Analizi

Değişkenler arasındaki doğrusal ilişkileri anlamak için Pearson korelasyon katsayıları hesaplanmış ve ısı haritası (heatmap) ile görselleştirilmiştir.

<img width="720" height="552" alt="image" src="https://github.com/user-attachments/assets/434def53-32bf-45bb-8f37-3405cdfd2951" />

En Yüksek Korelasyona Sahip 3 Özellik Çifti:
* Flavanoids & Total Phenols (0.86): Bu iki özellik arasında çok güçlü bir doğrusal ilişki vardır.
* Flavanoids & OD280/OD315 (0.79): Şarabın seyreltilme oranı ile flavanoid içeriği yüksek ilişkilidir.
* Total Phenols & OD280/OD315 (0.70): Toplam fenoller ile seyreltilme oranı arasında güçlü bir bağ vardır.

### Boxplot Analizi

<img width="700" height="400" alt="image" src="https://github.com/user-attachments/assets/10e588ca-a457-4d7d-9d49-55dedb2bee25" />

Her özellik için çizilen boxplot grafikleri, veri setindeki dağılımı ve aykırı değer eğilimlerini net bir şekilde göstermektedir. Özellikle malic_acid, magnesium, proanthocyanins ve color_intensity değişkenlerinde üst sınırın dışında kalan (outlier) değerler mevcuttur. Bu aykırı değerlerin varlığı, veriyi medyan ve çeyreklikler üzerinden ölçeklendiren RobustScaler yönteminin kullanılmasını desteklemektedir.

## 4, 5 ve 6. Veri Setinin Bölünmesi, Ölçeklendirme ve Boyut İndirgeme

Bu aşamada veriler model eğitimine hazır hale getirilmiş ve özellik seçimi ile boyut indirgeme işlemleri uygulanmıştır.

### Veri Bölme ve Ölçeklendirme (Split & Scaling)

Veri sızıntısını (Data Leakage) önlemek amacıyla, tüm işlemlerden önce veri seti bölünmüştür. Ölçeklendirme parametreleri yalnızca eğitim seti üzerinden hesaplanmıştır.

* Bölünme Oranları: %70 Eğitim, %10 Validasyon, %20 Test.
* Ölçeklendirme Yöntemi: RobustScaler.
* Tercih Nedeni: EDA aşamasında tespit edilen aykırı değerlerin (outliers) etkisi medyan ve çeyreklikler baz alınarak minimize edilmiştir.

```python
# Splitting the Dataset
X_train_full, X_test, y_train_full, y_test = train_test_split(feature_names, target, test_size=0.20, random_state=42, stratify=target)

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.125, random_state=42, stratify=y_train_full)

# Data Scaling
scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(feature_names)

# Feature Selection and Dimensionality Reduction

# Principal Component Analysis
pca_temp = PCA().fit(X_train_scaled)
avg_var = 1.0 / len(pca_temp.explained_variance_ratio_)
# components whose variance ratio is greater than the mean
n_comp_pca = np.sum(pca_temp.explained_variance_ratio_ > avg_var)

pca = PCA(n_components=n_comp_pca)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Linear Discriminant Analysis
n_lda = min(len(np.unique(y_train)) - 1, 3)
lda = LDA(n_components=n_lda)
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_val_lda = lda.transform(X_val_scaled)
X_test_lda = lda.transform(X_test_scaled)
```

### Boyut İndirgeme (Dimensionality Reduction)

Model performansını artırmak ve veriyi görselleştirmek amacıyla iki farklı boyut indirgeme yöntemi uygulanmıştır: PCA (Principal Component Analysis) ve LDA (Linear Discriminant Analysis)

<img width="500" height="271" alt="image" src="https://github.com/user-attachments/assets/9f275382-fd9d-4abc-9969-bac445697afd" />

<img width="1990" height="575" alt="image" src="https://github.com/user-attachments/assets/06030f80-4ffd-42ea-8c85-e39c812488a2" />

PCA Explained Variance grafiği, seçilen bileşenlerin verideki bilgiyi ne kadar temsil ettiğini açıkça göstermektedir. Yeşil kesikli çizgi, ödevde istenen 'ortalama varyans oranı' (1/13 ≈ 0.077) eşiğini temsil etmektedir. Grafikte görüldüğü üzere, ilk 4 bileşen bu eşiğin üzerinde kalarak verideki en önemli bilgiyi taşımaktadır. Sadece 4 bileşen seçilerek, veri boyutu %70 oranında azaltılmış ancak toplam varyansın yaklaşık %75-80'i korunmuştur. Bu, modelin karmaşıklığını azaltırken bilgi kaybını minimize eden mükemmel bir sonuçtur.

Boyut indirgeme çalışmaları sonucunda hem PCA hem de LDA yöntemlerinden üstün başarı elde edilmiştir.
* PCA tarafında, 13 boyuttan 4 boyuta inilmesine rağmen verinin karakteristik yapısı korunmuş ve sınıflar arasındaki temel farklılıklar belirginleşmiştir.
* LDA tarafında, 2 boyuta inilmesine rağmen sınıflar arasındaki ayrım (separation) neredeyse kusursuz hale gelmiştir.
Bu iki yöntemin sağladığı yüksek kaliteli veri temsilleri, modellerin eğitim aşamasında %100'e yakın doğruluk payı ile çalışmasına olanak sağlamıştır. Elde edilen görselleştirmeler, şarap sınıflarının bu yeni boyutlarda ne kadar net bir şekilde kümelendiğini kanıtlamaktadır.
* Raw Scaled Data: İki özellik (alcohol vs flavanoids) üzerinden bakıldığında sınıfların iç içe geçtiği ve ayrımın zor olduğu görülmektedir.

## 7 & 8. Makine Öğrenmesi Modellerinin Kurulması ve Performans Ölçümü

Bu bölümde, belirlenen 5 farklı algoritma, hazırlanan 3 farklı veri temsili üzerinde eğitilmiş ve performansları valide edilmiştir.

### Modelleme Yaklaşımı

Kod karmaşıklığını minimize etmek ve süreci daha modüler bir yapıya kavuşturmak amacıyla "Dictionary" tabanlı bir mimari tercih edilmiştir.
* Algoritmalar: Lojistik Regresyon, Karar Ağaçları, Random Forest, XGBoost ve Naive Bayes.
* Yöntem: Modeller ve veri setleri sözlük (dictionary) yapısında birleştirilerek, iç içe döngülerle (Grid Search Approach / Brute Force) tüm kombinasyonlar otomatik olarak test edilmiştir. Bu sayede kod daha temiz, okunabilir ve yönetilebilir hale getirilmiştir.

### Validation Performans Sonuçları

Toplamda 15 farklı model eğitilmiş ve validasyon seti üzerinde elde edilen metrik sonuçları aşağıdaki tabloda özetlenmiştir:

|index|Data Representation|Algorithm|Accuracy|Precision|Recall|F1-Score|ROC-AUC|
|---|---|---|---|---|---|---|---|
|0|Raw Data|Logistic Regression|1\.0|1\.0|1\.0|1\.0|1\.0|
|1|Raw Data|Decision Tree|0\.94|0\.95|0\.94|0\.94|0\.95|
|2|Raw Data|Rondom Forest|1\.0|1\.0|1\.0|1\.0|1\.0|
|3|Raw Data|XGBoost|1\.0|1\.0|1\.0|1\.0|1\.0|
|4|Raw Data|Naive Bayes|1\.0|1\.0|1\.0|1\.0|1\.0|
|5|PCA Data|Logistic Regression|1\.0|1\.0|1\.0|1\.0|1\.0|
|6|PCA Data|Decision Tree|0\.94|0\.95|0\.95|0\.94|0\.96|
|7|PCA Data|Rondom Forest|0\.94|0\.95|0\.95|0\.94|1\.0|
|8|PCA Data|XGBoost|0\.94|0\.95|0\.93|0\.93|1\.0|
|9|PCA Data|Naive Bayes|1\.0|1\.0|1\.0|1\.0|1\.0|
|10|LDA Data|Logistic Regression|1\.0|1\.0|1\.0|1\.0|1\.0|
|11|LDA Data|Decision Tree|1\.0|1\.0|1\.0|1\.0|1\.0|
|12|LDA Data|Rondom Forest|1\.0|1\.0|1\.0|1\.0|1\.0|
|13|LDA Data|XGBoost|1\.0|1\.0|1\.0|1\.0|1\.0|
|14|LDA Data|Naive Bayes|1\.0|1\.0|1\.0|1\.0|1\.0|

Elde edilen metrik sonuçlarına bakıldığında, modellerin büyük çoğunluğunun %100'e yakın doğruluk oranlarına ulaştığı söylenebilir. Özellikle LDA tabanlı verilerle eğitilen modellerin tamamı mükemmel performans sergilemiştir. Bu durum, veri ön işleme, ölçeklendirme ve boyut indirgeme adımlarının başarıyla gerçekleştirildiğini göstermektedir.

## 9. En İyi Modelin Test Üzerinde Değerlendirilmesi

Validasyon aşamasında en kararlı ve yüksek performansı sergileyen Raw Data üzerindeki Random Forest modeli, nihai test süreci için seçilmiştir. Model, daha önce eğitim sürecine dahil edilmeyen test seti (%20) üzerinde değerlendirilmiştir.

### Performans Metrikleri (Classification Report)

Modelin test verileri üzerindeki detaylı başarı sonuçları aşağıdadır:
| Sınıf (Class) | Precision | Recall | F1-Score | Örnek Sayısı (Support) |
| :--- | :--- | :--- | :--- | :--- |
| **Class 0** | 1.00 | 1.00 | 1.00 | 12 |
| **Class 1** | 1.00 | 1.00 | 1.00 | 14 |
| **Class 2** | 1.00 | 1.00 | 1.00 | 10 |
| **Genel Ortalama (Accuracy)** | **1.00** | **1.00** | **1.00** | **36** |

Model, test setindeki tüm örnekleri hatasız bir şekilde sınıflandırarak %100 doğruluk (Accuracy) oranına ulaşmıştır. Bu sonuç, modelin veriyi ezberlemediğini (overfitting olmadığını) ve yeni verilere mükemmel uyum sağladığını göstermektedir.

###  Karmaşıklık Matrisi (Confusion Matrix)

<img width="500" height="450" alt="image" src="https://github.com/user-attachments/assets/814f2914-92ca-4332-afb7-a4fcbca5cf82" />

Karmaşıklık matrisi incelendiğinde, test setindeki tüm örneklerin (12 Class 0, 14 Class 1, 10 Class 2) hatasız bir şekilde kendi sınıflarına atandığı görülmektedir. Matrisin ana diyagonali dışındaki tüm değerlerin 0 olması, modelin sınıfları birbirine karıştırmadığını kanıtlamaktadır.

### ROC Eğrisi ve Eşik (Threshold) Analizi

<img width="500" height="450" alt="image" src="https://github.com/user-attachments/assets/d92c5134-c76a-4f76-98f7-c54ca58629b1" />

ROC eğrisi analizi sonucunda, her üç sınıf için de AUC (Eğri Altındaki Alan) değeri 1.00 olarak hesaplanmıştır. Eğrilerin grafiğin sol üst köşesine tam yapışık olması, modelin sınıfları birbirinden ayırma kabiliyetinin kusursuz olduğunu gösterir. Eşik (Threshold) etkisi bakımından; modelin tahmin olasılıkları sınıflar arasında o kadar net bir ayrım yapmaktadır ki, karar eşiği değiştirilse dahi modelin mükemmel ayrıştırma gücü korunmaktadır. Bu sonuçlar, modelin Wine veri seti üzerinde son derece güvenilir ve başarılı bir şekilde çalıştığını teyit etmektedir.

## 10. XAI – SHAP Açıklanabilirlik Analizi

Bu bölümde, %100 doğrulukla çalışan modellerin bu kararları hangi kriterlere göre verdiği, "Açıklanabilir Yapay Zeka" (XAI) tekniklerinden biri olan SHAP (SHapley Additive exPlanations) yöntemiyle analiz edilmiştir.

### En İyi Model (Random Forest) için SHAP Analizi

Modelin kararlarını hangi kimyasal özelliklerin yönlendirdiğini anlamak için Beeswarm ve Bar plot grafikleri kullanılmıştır.

<img width="500" height="420" alt="image" src="https://github.com/user-attachments/assets/9c603ab9-f50d-411f-ba8c-71e7af859e69" />

Özellik Önem Sıralaması (Bar Plot): Modelin tahminlerinde en etkili olan ilk üç özellik sırasıyla Flavanoids (Feature 6), Proline (Feature 12) ve Color Intensity (Feature 9) olarak belirlenmiştir. Bu sonuçlar, şarap türlerinin ayırt edilmesinde flavanoid içeriği ve proline miktarının birincil derecede önemli olduğunu göstermektedir.

Model Performansı İlişkisi

SHAP önem sıralaması ile modelin %100 doğruluk payı arasında doğrudan bir ilişki vardır; model, gürültülü veriler yerine şarabın kimyasal karakterini yansıtan anlamlı özelliklere odaklandığı için test setinde hatasız sonuç vermiştir.

<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/a9390a09-d9f6-4e75-b768-d548ad37acb6" />
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/db6f6698-2bdb-4b56-970b-567d1317c584" />
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/f1099a03-e2a5-4e4a-bfe0-8326ad4459b0" />

Modelin karar mekanizmasını tam olarak kavramak adına her üç şarap sınıfı için ayrı ayrı SHAP Beeswarm grafikleri oluşturulmuştur. Bu analiz, Random Forest modelinin sınıfları birbirinden ayırırken her sınıf için farklı kimyasal özelliklere odaklandığını kanıtlamaktadır:
* Class 0 Analizi: Bu sınıfın en belirleyici özellikleri proline, flavanoids ve alcohol'dür. Grafikte görüldüğü üzere, bu özelliklerin yüksek değerleri (pembe noktalar) pozitif SHAP değerleri üreterek modelin 'Class 0' kararı vermesini
sağlamaktadır.
* Class 1 Analizi: Bu sınıf, diğerlerinden tamamen farklı bir mantığa sahiptir. Class 1 tahmini için düşük color_intensity, düşük proline ve düşük alcohol değerleri (mavi noktalar) en büyük pozitif etkiye sahiptir. Bu, modelin Class 1
şaraplarını kimyasal olarak 'daha hafif' profiller üzerinden tanımladığını kanıtlar.
* Class 2 Analizi: Bu sınıf için en kritik ayırt edici düşük flavanoids ve yüksek color_intensity değerleridir. Flavanoidlerin düşük olması tahmini pozitif yönde etkilerken, renk yoğunluğunun yüksek olması Class 2 seçimini
desteklemektedir.

Genel Değerlendirme:
Üç sınıfın karşılaştırmalı analizi, modelin veriyi ezberlemek yerine her şarap türü için tutarlı ve mantıklı birer 'kimyasal imza' oluşturduğunu göstermektedir. Sınıflar arasındaki bu net mantıksal ayrım, test setinde elde edilen %100 doğruluk başarısının tesadüf olmadığını, modelin verideki ayırt edici desenleri tamamen çözdüğünü bilimsel olarak ispatlamaktadır."

### PCA ve LDA Temsilleri için SHAP Karşılaştırması

Boyut indirgeme yapılmış veriler üzerindeki model kararları SHAP üzerinden kıyaslanmıştır.

PCA SHAP Analizi

<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/5b5ce165-b296-4238-819d-9f1a72574203" />
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/379c8693-6406-4227-96f5-4c9aaac0afc4" />
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/4b066849-5a11-442d-bf2a-6ccbe0c84000" />

Class 0 Analizi:
* Baskın Bileşenler: Bu sınıf için en kritik bileşenler PC2 ve PC1'dir. Her iki bileşenin de yüksek değerleri (pembe noktalar), SHAP değerini pozitife çekerek modelin "Class 0" kararı vermesini sağlar.
* PC3 ve PC4: Bu bileşenlerin etkisi daha sınırlıdır. Ancak PC4'ün bazı yüksek değerlerinin tahmini hafifçe desteklediği, PC3'ün ise karara çok düşük bir katkı sağladığı görülmektedir.

Class 1 Analizi:
* Baskın Bileşenler: PC2 bu sınıfta merkezi bir rol oynar ancak Class 0'ın aksine ters bir etkiye sahiptir. Düşük PC2 değerleri (mavi noktalar) tahmini pozitif yönde etkilerken, yüksek değerler negatif etki yaratır.
* PC1: Yüksek PC1 değerleri bu sınıf için genellikle negatif bir ağırlığa sahiptir.
* PC3 ve PC4: Bu bileşenler 0 noktasına çok yakın kümelenmiştir. Bu, modelin Class 1 kararını verirken PC3 ve PC4'ü ana ayırıcı olarak değil, sadece ikincil bir destekleyici olarak kullandığını gösterir.

Class 2 Analizi:
* Baskın Bileşenler: Bu sınıfın en güçlü belirleyicisi PC1 bileşenidir. Düşük PC1 değerleri (mavi noktalar) çok yüksek pozitif SHAP değerleri üreterek modelin "Class 2" demesini sağlayan en büyük itici güçtür.
* PC2: PC2'nin yüksek değerleri tahmini pozitif yönde desteklemektedir.
* PC3 ve PC4: Diğer sınıflarda olduğu gibi, bu bileşenlerin etki aralığı PC1 ve PC2'ye kıyasla çok daha dardır. Model, Class 2 kararını verirken bu bileşenlerden gelen düşük seviyeli varyans bilgilerinden faydalanmaktadır.

Analiz sonucunda, PCA modelinin tüm sınıflar için öncelikle PC1 ve PC2 bileşenlerindeki büyük değişimlere odaklandığı, PC3 ve PC4 bileşenlerini ise kararlarını hassaslaştırmak (fine-tuning) için yardımcı veri olarak kullandığı
ispatlanmıştır. Bilginin bileşenler arasındaki bu dağılımı, PCA'nın verideki varyansı koruma vizyonuyla tam uyumludur


LDA SHAP Analizi

<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/d64d88cf-c2a2-47c0-920f-f30ca3ad69bf" />
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/934f5fe7-d121-46d4-9dd8-f0301f63c1de" />
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/97e1ec35-d811-4ce7-89dd-0f4929620838" />

LDA tarafında LD1 bileşeninin "karar verici" rolü, sınıflar bazında şu şekilde netleşmektedir:
* Class 0 Karar Eşiği: LD1 değeri yaklaşık -2 eşiğinin altında olduğunda SHAP değeri aniden yükselerek modelin Class 0 kararı vermesini sağlar.
* Class 1 Karar Eşiği: Bu sınıf bir "aralık" içerisinde tanımlanmıştır. LD1 değeri -2 ile +2 arasında olduğunda SHAP değerleri pozitife dönmekte, bu aralığın dışına çıkıldığında ise model Class 1 olasılığını reddetmektedir.
* Class 2 Karar Eşiği: LD1 değeri +2 eşiğini geçtiği anda SHAP değerinde radikal bir yükseliş (yaklaşık +4 puan) görülür. Bu, LD1'in yüksek değerlerinin doğrudan Class 2'yi temsil ettiğini kanıtlar
