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
