import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
import os

class Classifier:
    def __init__(self, name):
        self.name = name
        self.classes = {}

    def add_class(self, class_name):
        self.classes[class_name] = []

    def add_data(self, class_name, data):
        self.classes[class_name].append(data)

    def train(self, model_type='SVM'):
        if model_type == '1':
            self.model = SVC(kernel='linear')
        elif model_type == '2':
            self.model = MultinomialNB()
        elif model_type == '3':
            self.model = DecisionTreeClassifier()
        elif model_type == '4':
            self.model = KNeighborsClassifier()
        elif model_type == '5':
            self.model = LogisticRegression()
        else:
            print("Geçersiz seçim. Varsayılan olarak Destek Vektör Makineleri (SVM) kullanılacak.")
            self.model = SVC(kernel='linear')

        # Verileri vektörlere dönüştür
        self.vectorizer = TfidfVectorizer(max_features=500)  # max_features parametresini ayarlayarak vektör boyutunu kontrol edin
        X_train_vectors = self.vectorizer.fit_transform([data for class_name, data_list in self.classes.items() for data in data_list])
        y_train = [class_name for class_name, data_list in self.classes.items() for data in data_list]

        # Modeli eğit
        self.model.fit(X_train_vectors, y_train)

    def predict(self, text):
        text_vector = self.vectorizer.transform([text])
        predicted_class = self.model.predict(text_vector)
        return predicted_class[0]

# Eğitilmiş modelin kaydedilmesi ve yüklenmesi için sınıf
class ModelManager:
    def save_model(self, classifier, model_file):
        with open(model_file, 'wb') as file:
            pickle.dump(classifier, file, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model başarıyla kaydedildi:", model_file)

    def load_model(self, model_file):
        with open(model_file, 'rb') as file:
            classifier = pickle.load(file)
        return classifier

# Mevcut modellerin yüklenmesi
def load_existing_models():
    models = {}
    for file in os.listdir('.'):
        if file.endswith("_trained_model.pkl"):
            model_name = file.split('_')[0]
            models[model_name] = file
    return models

# Kullanıcıya mevcut modelleri kullanma veya yeni bir model oluşturma seçeneği sunma
def ask_user_choice(models):
    print("Mevcut Modeller:")
    for idx, model_name in enumerate(models.keys(), start=1):
        print(f"{idx}. {model_name}")

    choice = input("Önceden eğitilmiş bir model seçin (1 - {}) veya yeni bir model oluşturmak için 'y' girin: ".format(len(models)))
    if choice.lower() == 'y':
        return None
    else:
        selected_model_name = list(models.keys())[int(choice) - 1]
        model_file = models[selected_model_name]
        return model_file

# Mevcut modelleri yükleme
models = load_existing_models()

# Kullanıcıya mevcut modelleri kullanma veya yeni bir model oluşturma seçeneği sunma
selected_model_file = ask_user_choice(models)

# Eğer mevcut bir model seçilmediyse, yeni bir model oluşturma adımlarına geçme
if selected_model_file is None:
    # Yeni bir sınıflandırıcı oluşturulması
    classifier_name = input("Yeni sınıflandırıcı adı girin: ")
    classifier = Classifier(classifier_name)

    # Kullanıcıdan class isimlerini ve verilerini al
    while True:
        class_name = input("Bir class adı girin (Çıkmak için 'q' ya basın): ")
        if class_name.lower() == 'q':
            break
        classifier.add_class(class_name)

        # Kullanıcıdan veri giriş yöntemini seçmesini iste
        data_input_choice = input("Veri giriş yöntemini seçin:\n"
                                  "1. Elle veri girişi\n"
                                  "2. Dosya yükleme\n"
                                  "Seçiminiz (1 veya 2): ")

        if data_input_choice == '1':  # Elle veri girişi
            while True:
                data = input("Veri girin (Class'ı değiştirmek için 'q' ya basın): ")
                if data.lower() == 'q':
                    break
                classifier.add_data(class_name, data)
        elif data_input_choice == '2':  # Dosya yükleme
            while True:
                file_path = input("Yüklemek istediğiniz dosyanın yolunu girin (Devam etmek için 'q' ya basın): ")
                if file_path.lower() == 'q':
                    break
                try:
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        for index, row in df.iterrows():
                            classifier.add_data(class_name, row.iloc[0])  # Assuming the text is in the first column
                    elif file_path.endswith('.txt'):
                        with open(file_path, 'r', encoding='ISO-8859-1') as file:
                            text = file.read()
                        classifier.add_data(class_name, text)
                    else:
                        print("Geçersiz dosya türü.")
                        continue
                except FileNotFoundError:
                    print("Dosya bulunamadı veya okunamadı.")

        else:
            print("Geçersiz seçim.")

    # Kullanıcıdan hangi modeli seçmek istediğini al
    model_choice = input("Kullanmak istediğiniz sınıflandırma algoritmasını seçin:\n"
                         "1. Destek Vektör Makineleri (SVM)\n"
                         "2. Naive Bayes\n"
                         "3. Karar Ağacı\n"
                         "4. K-En Yakın Komşu (KNN)\n"
                         "5. Lojistik Regresyon\n"
                         "Seçiminiz (1-5): ")

    # Sınıflandırıcıyı eğit
    classifier.train(model_type=model_choice)

    # Modelin kaydedilmesi
    model_file = f'{classifier_name}_trained_model.pkl'
    ModelManager().save_model(classifier, model_file)

    selected_model_file = model_file

# Seçilen modelin yüklenmesi
classifier = ModelManager().load_model(selected_model_file)
print(f"{selected_model_file} modeli başarıyla yüklendi.")

# Kullanıcıdan bir metin al ve tahmin yap
while True:
    user_input = input("Bir metin girin (Çıkmak için 'q' ya basın): ")
    if user_input.lower() == 'q':
        break
    predicted_class = classifier.predict(user_input)
    print("Tahmin edilen class:", predicted_class)
