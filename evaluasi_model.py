import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(test_dir, batch_size=16):
    # Preprocessing data pengujian
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),  # Sesuaikan dengan ukuran input model
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # Penting agar urutan data konsisten dengan prediksi
    )

    # Memuat model yang telah dilatih
    model = tf.keras.models.load_model('model_hama_plant.keras')

    # Evaluasi model (accuracy dan loss)
    test_loss, test_acc = model.evaluate(
        test_generator, 
        steps=test_generator.samples // test_generator.batch_size
    )
    print(f"Test accuracy: {test_acc}")
    print(f"Test loss: {test_loss}")

    # Prediksi menggunakan model
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)  # Mengambil indeks label prediksi
    y_true = test_generator.classes           # Label asli dari data pengujian

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)

    # Visualisasi Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=test_generator.class_indices.keys(), 
                yticklabels=test_generator.class_indices.keys())
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys()))
    print("\nClassification Report:\n", report)

if __name__ == "__main__":
    # Path dataset pengujian
    test_dir = 'dataset/test'
    
    # Jalankan evaluasi model
    evaluate_model(test_dir)
