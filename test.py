import numpy as np
from tensorflow import keras

# Modeli yükle
model = keras.models.load_model('EggZayn_final.h9_4')

# Test verilerini yükle
X_test = np.load('X_test.npy')
Y_test = np.load('Y_test.npy')

# İlk 20 örnek için tahmin yap
predictions = model.predict(X_test[:20])

# "0" sınıfını penalize et
adjusted_predictions = predictions.copy()
adjusted_predictions[:, 0] *= 0.7  # "0" sınıfını %30 azalt

# Eşik ayarı (%20, daha düşük eşik)
threshold = 0.2
predicted_classes = np.where(np.max(adjusted_predictions, axis=1) > threshold, np.argmax(adjusted_predictions, axis=1), -1)
print(f"Düzeltilmiş Tahminler (Eşik: {threshold}): {predicted_classes}")
print(f"Gerçek Etiketler: {Y_test[:20]}")