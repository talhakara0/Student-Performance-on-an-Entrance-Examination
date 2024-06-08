import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import zipfile
import os

# ZIP dosyasını indirip açalım
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
zip_path = "student.zip"

# Eğer zip dosyası yoksa indir
if not os.path.exists(zip_path):
    import urllib.request
    urllib.request.urlretrieve(url, zip_path)

# ZIP dosyasını açalım
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall()

# CSV dosyasını okuyalım
student_data = pd.read_csv('student-mat.csv', sep=';')

# Veri setinin ilk birkaç satırına göz atalım
print(student_data.head())

# Veri setinin boyutunu ve sütun adlarını görelim
print(student_data.shape)
print(student_data.columns)

# Veri setinin özet istatistikleri
print(student_data.describe())

# Veri setindeki eksik değerleri kontrol edelim
print(student_data.isnull().sum())

# Kategorik değişkenleri one-hot encoding ile dönüştürelim
student_data_encoded = pd.get_dummies(student_data)

# Dönüştürülmüş veri setinin ilk birkaç satırına bakalım
print(student_data_encoded.head())

# Normalizasyon için MinMaxScaler kullanarak veriyi 0-1 aralığına getirelim
scaler = MinMaxScaler()
student_data_normalized = scaler.fit_transform(student_data_encoded)

# Normalizasyon sonrası veri setinin ilk birkaç satırına bakalım
print(student_data_normalized[:5])

# Bağımlı ve bağımsız değişkenleri ayıralım
X = student_data_encoded.drop("G1", axis=1)  # Örneğin, G1 değişkenini tahmin etmeye çalışalım
y = student_data_encoded["G1"]

# Veriyi eğitim ve test setlerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree modelini oluşturalım ve eğitelim
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Test seti ile tahmin yapalım ve doğruluk skorunu hesaplayalım
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree Accuracy: {:.2f}%".format(accuracy * 100))

# Random Forest modelini oluşturalım ve eğitelim
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Test seti ile tahmin yapalım ve doğruluk skorunu hesaplayalım
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy: {:.2f}%".format(accuracy_rf * 100))

# Örnek olarak, modelin tahmin ettiği ve gerçek değerleri karşılaştıran bir grafik çizelim ve kaydedelim
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.7)
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Edilen Değerler")
plt.title("Rastgele Orman Modeli: Gerçek vs Tahmin")
plt.savefig('rastgele_orman_modeli.png')
plt.show()
