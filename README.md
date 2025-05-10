NOTE: THIS IS PROBLEM ACCUARY %30

Ne Yapar?: Bu model, özel olarak işlenmiş EEG beyin dalgası verilerini girdi olarak alır. Bu veriler, bir kişinin belirli hareketleri (sol yumruk, sağ yumruk, iki yumruk, iki ayak) yaptığı veya hayal ettiği sıradaki beyin aktivitesini temsil eder. Model, bu sinyal desenlerini analiz ederek, kişinin o anda hangi hareketi düşündüğünü veya yaptığını % olarak bir güven skoruyla birlikte tahmin eder.
Girdi: Modelin çalışması için, belirli elektrotlardan (özellikle motor korteksle ilgili olanlar: Fc3, Fc4, C3, C4, Cz, Cp3, Cp4) alınmış, filtrelenmiş ve belirli bir formata getirilmiş kısa süreli EEG sinyal parçalarına ihtiyaç duyar.
Çıktı: Model, analiz ettiği EEG sinyaline dayanarak dört olası durumdan hangisinin en olası olduğunu söyler:
Sol Yumruk Hareketi/Hayali
Sağ Yumruk Hareketi/Hayali
Her İki Yumruğun Hareketi/Hayali
Her İki Ayağın Hareketi/Hayali
Amacı: Bu tür bir modelin temel amacı, beyin aktivitesini doğrudan bir komuta dönüştürmektir. Bu, özellikle Beyin-Bilgisayar Arayüzleri (Brain-Computer Interface - BCI) için kritik bir adımdır. Örneğin, felçli bir bireyin sadece düşünerek bir protez kolu hareket ettirmesi gibi uygulamaların temelini oluşturabilir.
Kodun Kendisi Ne Sağlıyor?

Kod, yukarıda açıklanan bu modeli oluşturmak, eğitmek, değerlendirmek ve kullanmak için gerekli tüm adımları ve araçları içerir:

Veri İşleme: Büyük bir EEG veri setini (EEGMMIDB) veya kullanıcının kendi EEG verisini alır, beyin sinyallerindeki gürültüyü temizler, ilgili frekans aralıklarına odaklar ve modeli eğitmek için uygun hale getirir.
Model Mimarisi: Tahmin işini yapacak olan yapay zeka modelini (derin öğrenme, özellikle Transformer blokları ve contrastive learning gibi teknikler kullanarak) tanımlar.
Eğitim Süreci: Modeli, hazırlanan EEG verileriyle besleyerek hangi sinyal deseninin hangi harekete karşılık geldiğini öğrenmesini sağlar. Eğitim sonucunda "EggZayn_final.h9_4" ve ".tflite" uzantılı model dosyaları kaydedilir.
Değerlendirme: Eğitilen modelin ne kadar başarılı olduğunu (doğruluk oranı, hangi hareketleri ne kadar iyi ayırt edebildiği vb.) test verileri üzerinde ölçer ve raporlar.
Kullanıcı Arayüzü (GUI): Tüm bu işlemleri (veri hazırlama, eğitim, değerlendirme, yeni sinyal tahmini) kullanıcının kolayca yapabilmesi için "EggZayn v9.4" başlıklı bir grafiksel arayüz (pencere) sunar. Bu arayüz üzerinden model eğitilebilir, performansı görülebilir ve yeni EEG sinyalleri analiz edilebilir. Eğitim ve değerlendirme sonuçları grafiksel olarak da gösterilir.
Özetle: Bu kod, belirli motor hareketlerini düşünmeye veya yapmaya karşılık gelen EEG beyin dalgalarını tanıyabilen bir yapay zeka modeli yaratır ve bu modelin oluşturulup kullanılabileceği bir masaüstü uygulaması (GUI) sağlar. Sonuç, beyin sinyallerinden niyet okuyabilen bir BCI sisteminin temel taşıdır.

English Explanation:

What is the Result of This Code and What Does the Resulting Model Do?

This code essentially creates a system that analyzes brain signals (EEG) to predict which specific movement a person is thinking about or performing, and provides an interface to use this system.

The Result (The Trained Model - "EggZayn v9.4"):

What it Does: This model takes specifically processed EEG brainwave data as input. This data represents the brain activity recorded while a person is performing or imagining specific movements (left fist, right fist, both fists, both feet). The model analyzes these signal patterns and predicts which movement the person was likely thinking about or performing at that moment, along with a percentage confidence score.
Input: To work, the model needs short segments of EEG signals acquired from specific electrodes (particularly those related to the motor cortex: Fc3, Fc4, C3, C4, Cz, Cp3, Cp4), which have been filtered and formatted correctly.
Output: Based on the EEG signal it analyzes, the model indicates which of the four possible states is most likely:
Left Fist Movement/Imagery
Right Fist Movement/Imagery
Both Fists Movement/Imagery
Both Feet Movement/Imagery
Purpose: The fundamental goal of such a model is to translate brain activity directly into a command. This is a critical step for Brain-Computer Interfaces (BCIs). For example, it could form the basis for applications like allowing a paralyzed individual to control a prosthetic arm just by thinking about the movement.
What the Code Itself Provides:

The code contains all the necessary steps and tools to create, train, evaluate, and use the model described above:

Data Processing: It takes a large EEG dataset (EEGMMIDB) or the user's custom EEG data, cleans the noise from the brain signals, focuses on relevant frequency bands, and prepares it suitably for training the model.
Model Architecture: It defines the artificial intelligence model (using deep learning, specifically techniques like Transformer blocks and contrastive learning) that will perform the prediction task.
Training Process: It feeds the prepared EEG data to the model, allowing it to learn which signal patterns correspond to which movement. The training results in saved model files ("EggZayn_final.h9_4" and ".tflite").
Evaluation: It measures how successful the trained model is (accuracy, how well it distinguishes between movements, etc.) on unseen test data and reports the performance.
User Interface (GUI): It provides a graphical interface (a window titled "EggZayn v9.4") so that a user can easily manage all these operations: data preparation, training, evaluation, and predicting from new signals. This interface allows training the model, viewing its performance, and analyzing new EEG signals. Training and evaluation results are also visualized graphically.
In summary: This code creates an artificial intelligence model capable of recognizing EEG brainwaves corresponding to specific motor movements or imagery, and provides a desktop application (GUI) to build and use this model. The outcome is a foundational piece for a BCI system that can interpret intentions from brain signals.
