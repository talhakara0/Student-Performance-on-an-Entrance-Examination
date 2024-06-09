import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Veri setini yükleme ve işleme
student_data = pd.read_csv('student-mat.csv', sep=';')

# Temel istatistikler
print(student_data.describe())

# Sadece sayısal sütunları seçelim
numeric_student_data = student_data.select_dtypes(include=[np.number])

# Korelasyon matrisi
corr_matrix = numeric_student_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Korelasyon Matrisi')
plt.savefig('correlation_matrix.png')
plt.show()

# Cinsiyete göre not ortalamaları
plt.figure(figsize=(10, 6))
sns.boxplot(x='sex', y='G3', data=student_data)
plt.title('Cinsiyete Göre Not Ortalamaları')
plt.xlabel('Cinsiyet')
plt.ylabel('G3 Notu')
plt.savefig('gender_vs_grades.png')
plt.show()

# Yaşa göre başarı dağılımı
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='G3', data=student_data)
plt.title('Yaşa Göre Başarı Dağılımı')
plt.xlabel('Yaş')
plt.ylabel('G3 Notu')
plt.savefig('age_vs_grades.png')
plt.show()

# Anne eğitim seviyesine göre not ortalamaları
plt.figure(figsize=(10, 6))
sns.boxplot(x='Medu', y='G3', data=student_data)
plt.title('Anne Eğitim Seviyesine Göre Not Ortalamaları')
plt.xlabel('Anne Eğitim Seviyesi')
plt.ylabel('G3 Notu')
plt.savefig('medu_vs_grades.png')
plt.show()

# Baba eğitim seviyesine göre not ortalamaları
plt.figure(figsize=(10, 6))
sns.boxplot(x='Fedu', y='G3', data=student_data)
plt.title('Baba Eğitim Seviyesine Göre Not Ortalamaları')
plt.xlabel('Baba Eğitim Seviyesi')
plt.ylabel('G3 Notu')
plt.savefig('fedu_vs_grades.png')
plt.show()

# Haftalık çalışma süresine göre not ortalamaları
plt.figure(figsize=(10, 6))
sns.boxplot(x='studytime', y='G3', data=student_data)
plt.title('Haftalık Çalışma Süresine Göre Not Ortalamaları')
plt.xlabel('Haftalık Çalışma Süresi')
plt.ylabel('G3 Notu')
plt.savefig('studytime_vs_grades.png')
plt.show()

# Model performans görselleştirmesi
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Kategorik değişkenleri one-hot encoding ile dönüştürelim
student_data_encoded = pd.get_dummies(student_data)

# Normalizasyon için MinMaxScaler kullanarak veriyi 0-1 aralığına getirelim
scaler = MinMaxScaler()
student_data_normalized = scaler.fit_transform(student_data_encoded)

# Bağımlı ve bağımsız değişkenleri ayıralım
X = student_data_encoded.drop("G1", axis=1)  # Örneğin, G1 değişkenini tahmin etmeye çalışalım
y = student_data_encoded["G1"]

# Veriyi eğitim ve test setlerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest modelini oluşturalım ve eğitelim
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Test seti ile tahmin yapalım ve doğruluk skorunu hesaplayalım
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy: {:.2f}%".format(accuracy_rf * 100))

# Modelin tahmin ettiği ve gerçek değerleri karşılaştıran bir grafik çizelim ve kaydedelim
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.7)
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Edilen Değerler")
plt.title("Rastgele Orman Modeli: Gerçek vs Tahmin")
plt.savefig('rastgele_orman_modeli.png')
plt.show()
