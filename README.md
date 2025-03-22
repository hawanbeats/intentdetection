# Intent Classification Example with Naive Bayes

## Proje Özeti
Bu Python kodu, scikit-learn'in ``CountVectorizer`` ve ``MultinomialNB`` sınıflarını kullanarak metin verilerinin sınıflandırılmasını sağlar. Kullanıcı tarafından girilen metin, daha önce etiketlenmiş örneklerle eğitilmiş bir modelle sınıflandırılır. Kod, belirli bir niyeti (toplantı sorgusu, hava durumu, hatırlatıcı ekleme vb.) tahmin eder.

## Çalıştırma Adımları
1. Python ve Gerekli Kütüphaneleri Yükleme
- Python'un yüklü olduğundan emin olun. Python 3.x sürümü önerilir.
- ``scikit-learn`` kütüphanesini yüklemek için terminal veya komut istemcisinde aşağıdaki komutu çalıştırın:
``
pip install scikit-learn
``
2. Kodun Çalıştırılması
- ``sentences`` ve ``labels`` üzerinde model eğitimi yapılır. Model, her cümle için bir etiket tahmin eder.
- Kullanıcı bir soru sorar ve sistem, verilen cümleyi sınıflandırarak hangi niyete ait olduğunu tahmin eder.

## Kod Açıklaması
```python
from sklearn.feature_extraction.text import CountVectorizer # type: ignore
from sklearn.naive_bayes import MultinomialNB # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore

# Örnek niyetler ve cümleler
sentences = ["Yarın toplantım var mı?", 
             "Bana bir hatırlatıcı ekle", 
             "Saat kaçta dersim var?",
             "Bugün hava nasıl?",
             "Yağmur yağacak mı?",
             "Alarmı sabah 7’ye kur",
             "Öğle yemeği için rezervasyon yap"]

labels = ["toplantı_sorgu", "hatırlatıcı", "ders_sorgu", "hava_durumu",
          "hava_durumu", "hatırlatıcı", "hatırlatıcı"]

# Etiketleri sayısal hale getir
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Metinleri vektörleştir
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

# Modeli eğit
model = MultinomialNB()
model.fit(X, y)

# Kullanıcıdan giriş alarak sürekli çalıştır
while True:
    user_input = input("Bir soru sor (Çıkmak için 'q' yazın): ")
    
    if user_input.lower() == "q":
        print("Sistem kapatılıyor...")
        break

    test_vector = vectorizer.transform([user_input])
    predicted_label = model.predict(test_vector)[0]

    print("Tahmini Niyet:", label_encoder.inverse_transform([predicted_label])[0])
```
- ``CountVectorizer``: Cümleleri sayısal verilere dönüştürmek için kullanılır.
- ``MultinomialNB``: Naive Bayes sınıflandırıcı modelidir, metin sınıflandırma için yaygın olarak kullanılır.
- ``LabelEncoder``: Etiketleri sayısal değerlere dönüştürmek için kullanılır.

## Örnek Çıktı
- Girdi:
```
Bir soru sor (Çıkmak için 'q' yazın): Bugün hava nasıl?
```
- Çıktı:
```
Tahmini Niyet: hava_durumu
```
- Girdi:
```
Bir soru sor (Çıkmak için 'q' yazın): Alarmı sabah 7’ye kur
```
- Çıktı:
```
Tahmini Niyet: hatırlatıcı
```
## Katkıda Bulunma
Katkıda bulunmak isterseniz, önerilerinizi veya hataları GitHub issues bölümünde paylaşabilir veya pull request gönderebilirsiniz.
