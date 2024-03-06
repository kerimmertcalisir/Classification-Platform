# py -m pip install
# py -m uvicorn main:app --reload
from fastapi import FastAPI, Query, File, UploadFile, Form
from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
from typing import List, Optional
import io

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
        self.vectorizer = TfidfVectorizer()
        X_train_vectors = self.vectorizer.fit_transform([data for class_name, data_list in self.classes.items() for data in data_list])
        y_train = [class_name for class_name, data_list in self.classes.items() for data in data_list]

        # Modeli eğit
        self.model.fit(X_train_vectors, y_train)

    def predict(self, text):
        text_vector = self.vectorizer.transform([text])
        predicted_class = self.model.predict(text_vector)
        return predicted_class[0]

app = FastAPI()
classifier = None
@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    uploaded_files = []
    for file in files:
        file_contents = await file.read()
        file_name = file.filename
        # Veriyi işleyebilir veya kaydedebilirsiniz
        uploaded_files.append({"filename": file_name, "contents": file_contents.decode()})
    return JSONResponse(status_code=200, content={"message": "Dosyalar başarıyla yüklendi.", "uploaded_files": uploaded_files})

@app.post("/train/")
async def train_classifier(classifier_name: str = Form(...),
                           class_names: List[str] = Form(...),
                           data_input_choices: List[str] = Form(...),
                           data: Optional[List[UploadFile]] = File(...),
                           model_choice: str = Form(...)):
    global classifier
    classifier = Classifier(classifier_name)

    for class_name in class_names:
        classifier.add_class(class_name)

    for i, class_name in enumerate(class_names):
        data_input_choice = data_input_choices[i]
        for item in data[i].file.read().decode('utf-8').splitlines():
            if data_input_choice == '1':
                classifier.add_data(class_name, item)
            elif data_input_choice == '2':
                if item.endswith('.csv'):
                    df = pd.read_csv(io.StringIO(item))
                    for index, row in df.iterrows():
                        classifier.add_data(class_name, row.iloc[0])  # Assuming the text is in the first column
                elif item.endswith('.txt'):
                    text = item
                    classifier.add_data(class_name, text)
                else:
                    return JSONResponse(status_code=400, content={"message": "Geçersiz dosya türü."})
            else:
                return JSONResponse(status_code=400, content={"message": "Geçersiz seçim."})

    classifier.train(model_type=model_choice)


    # Modeli kaydetme işlemi
    model_file = 'trained_model.pkl'
    with open(model_file, 'wb') as file:
        pickle.dump(classifier, file)

    return JSONResponse(status_code=200, content={"message": "Model başarıyla eğitildi ve kaydedildi."})

@app.post("/predict/")
async def predict_text(text: str):
    if classifier:
        predicted_class = classifier.predict(text)
        return JSONResponse(status_code=200, content={"predicted_class": predicted_class})
    else:
        return JSONResponse(status_code=400, content={"message": "Model henüz eğitilmedi."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
