import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm

#Lineer Regresyonumuzu Tanımladık 

class Linear_Regression(): 
  
    # Lineer Sınıf Parametreler başlatılıyor.
    def __init__(self, learning_rate, no_of_itr): 
        self.learning_rate = learning_rate 
        self.no_of_itr = no_of_itr 
  
    def fit(self, X, Y): 
  
        # Eğitim örneği sayısı ve sayısı özellikleri. 
        self.m, self.n = X.shape     # Satır ve sütun sayısı
        # Ağırlığı ve biansı başlatmak
        self.w = np.zeros((self.n, 1)) 
        self.b = 0
        self.X = X 
        self.Y = Y 
  
        # Gradyan inişinin uygulanması.
        for i in range(self.no_of_itr): 
            self.update_weigths() 
    #Ağırlığı güncelleme 
    def update_weigths(self): 
        Y_prediction = self.predict(self.X) 
  
      # Gradyanların hesaplanması 
        dw = -(self.X.T).dot(self.Y - Y_prediction)/self.m 
  
        db = -np.sum(self.Y - Y_prediction)/self.m 
  
        # Ağırlıklar ve bias değerlerini güncelleniyor
        self.w = self.w - self.learning_rate * dw 
        self.b = self.b - self.learning_rate * db 
  
  #Regresyon modelinin hesaplanmasını içerir. X giriş verilerini, ağırlıklar 'self.w' ve bias' self.b' ile çarparak ve toplayarak modelin tahminlerini elde ediyoruz.
    def predict(self, X): 
        return X.dot(self.w) + self.b 
    
    # Gradyan inişini gerçekleştirelim
    def gradient_descent(self):
        Y_prediction = self.predict(self.X)

        # Gradyanları hesaplama
        dw = -(self.X.T).dot(self.Y - Y_prediction) / self.m
        db = -np.sum(self.Y - Y_prediction) / self.m

    #Ağırlığı yazdırıyor
    def print_weights(self): 
        print('İlgili özellikler için ağırlıklar :') 
        print(self.w) 
        print() 
        print('Regresyon için bias değeri ', self.b) 

#veri setimizi yüklüyoruz 

veriseti = pd.read_csv('dataset/dataset_Facebook.csv',delimiter=";")
 # verisetindeki null değerlerini öne cıkarma ve doldurma 
veriseti.dropna()
veriseti['Paid'].fillna(1, inplace=True)
veriseti['like'].fillna(1, inplace=True)
veriseti['share'].fillna(1, inplace=True)
 
#Datasetini x ve y olarak sütunlara bölme işlemi 

xSutun =['Category', 'Page total likes', 'Post Month', 'Post Hour', 'Post Weekday', 'Paid']
ySutun = ['Total Interactions']

# Ödevimize uygun hale gelmesi için bazı sütünları seçelim
x = veriseti.iloc[:, [0, 2, 3, 4, 5, 6,13,15,16,17]]
y = veriseti.iloc[:, 18:]

# 'Type' sütununu numerik değerlere dönüştürüp doğruluğu arttıralım
ohe = OneHotEncoder()
c = veriseti.iloc[:, 1:2]
c = ohe.fit_transform(c).toarray().astype(int)
tip = pd.DataFrame(data=c, index=range(500), columns=['Link', 'Photo', 'Status', 'Video'])
x = pd.concat([tip, x.iloc[:, :]], axis=1)

# Verileri eğitim ve test olarak bölelim
x_train ,x_test ,y_train ,y_test =train_test_split(x,y,test_size=0.2,random_state=10)


model = Linear_Regression(learning_rate=0.001, 
                          no_of_itr=20000) 
scaler=StandardScaler() #veri önişleme aşamasında kullanılır


#Burada veri ölçeklendirme yapmaktayız(fit transform) xtrain ,xtest,ytrain,ytest için
#bu özelliklerin farklı olması durumunda model performansını artırmak ve eğitimi stabilize etmek için kullanılır.
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
y_train_scaled=scaler.fit_transform(y_train)
y_test_scaled=scaler.transform(y_test)

model.fit(x_train_scaled, y_train_scaled) #bir makine öğrenimi modelini eğitmek için kullanılır
model.print_weights()#eğitilen lineer regresyon modelinin öğrenilen ağırlıklarını ve bias değerini ekrana yazdırmaktır
tahmin=model.predict(x_test_scaled) #ölçeklendirilmiş test verileri (x_test_scaled) üzerinde tahminler yapmaktadır.
hata_test=y_test_scaled-tahmin #modelin tahminleri ile gerçek test etiketleri arasındaki hataları hesaplar.
tahmin_egitim=model.predict(x_train_scaled) #eğitilmiş bir modeli kullanarak, ölçeklendirilmiş eğitim verisi (x_train_scaled) üzerinde tahminler yapmaktadır.
hata_egitim=y_train_scaled-tahmin_egitim  #modelin eğitim verisi üzerindeki tahminler ile gerçek eğitim etiketleri arasındaki hataları hesaplar.

plt.scatter(np.arange(len(hata_egitim)),hata_egitim,label="Eğitim Hataları")#eğitim verisi üzerindeki hataların dağılımını görselleştirmek amacıyla bir saçılım (scatter) grafiği oluşturur.
plt.scatter(np.arange(len(hata_test)),hata_test,label="Test Hataları") #test verisi üzerindeki hataların dağılımını görselleştirmek amacıyla bir saçılım (scatter) grafiği oluşturur
#grafiğini daha anlaşılır hale getirmek ve grafiği görsel olarak düzenlemek için kullanılır.
plt.xlabel('Veri noktası') # X ekseni etiketi
plt.ylabel('Hata') # Y ekseni etiketi
plt.legend() # Etiketleri göster
plt.show() # Grafiği göster

#modelin test verileri üzerindeki tahmin hatalarının normal dağılıma ne kadar uyduğunu görmek için bir QQ (Quantile-Quantile) çizelgesi oluşturur.
#QQ çizelgesi, bir veri setinin normal dağılıma ne kadar uyduğunu gösteren bir grafiktir.
residuals = y_test_scaled - tahmin
sm.qqplot(residuals, line='s')
plt.title('QQ Çizgisi')
plt.show()

#Ortalama kare hatası 
mse = np.mean((hata_test)**2)

print('Ortalama Kare Hatası:', mse)

# toplam kare hatası
sse = np.sum((hata_test)**2)

print('Toplam Kare Hata Oranı:', sse)




