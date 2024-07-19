import numpy as np
import cv2
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import argparse
import pickle
from pyimagesearch.faces import detect_faces, load_face_dataset

def preprocess_image(faceROI_grey):
    # Ensure the image is resized to 160x160 and convert to 3-channel RGB
    faceROI_rgb = cv2.cvtColor(faceROI_grey, cv2.COLOR_GRAY2RGB)
    faceROI_rgb = cv2.resize(faceROI_rgb, (160, 160))
    
    # Convert to NumPy array and ensure the dtype is float32
    faceROI_rgb = faceROI_rgb.astype(np.float32)
    
    # Add batch dimension to make shape (1, 160, 160, 3)
    faceROI_rgb = np.expand_dims(faceROI_rgb, axis=0)
    
    return faceROI_rgb

def predict_single_image(image_path, embedder, svm_model, label_encoder):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale and preprocess the image
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_image = preprocess_image(image_gray)
    
    # Generate embeddings using FaceNet model
    embeddings = embedder.embeddings(processed_image)
    
    # Predict using SVM model
    predictions = svm_model.predict(embeddings)
    
    # Decode predictions using label encoder
    predicted_label = label_encoder.inverse_transform(predictions)[0]
    
    return predicted_label

def main():
    # Define the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", help="path to input dataset", default='Faces')
    args = vars(ap.parse_args())

    print("[INFO] loading face detector model...")
    net = cv2.dnn.readNet('face_detector/deploy.prototxt', 'face_detector/res10_300x300_ssd_iter_140000.caffemodel')

    print("[INFO] loading dataset...")
    (faces, labels) = load_face_dataset(args["input"], net, minConfidence=0.5, minSamples=20)
    print("[INFO] {} images in dataset".format(len(faces)))

    # Initialize the FaceNet model
    embedder = FaceNet()

    # Preprocess the images
    faces = [preprocess_image(face) for face in faces]

    # Check the shape of the first preprocessed image
    print(f"Shape of first preprocessed image: {faces[0].shape}")

    # Generate embeddings
    print("[INFO] generating embeddings...")
    embeddings = np.array([embedder.embeddings(face) for face in faces])
    embeddings = embeddings.reshape(len(faces), -1)

    # Encode the string labels as integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # Construct our training and testing split
    split = train_test_split(embeddings, labels, test_size=0.25, stratify=labels, random_state=42)
    (trainX, testX, trainY, testY) = split

    # Train an SVM model on the embeddings
    print("[INFO] training SVM model...")
    model = SVC(kernel="linear", C=1.0, probability=True)
    model.fit(trainX, trainY)

    # Save the trained model and label encoder
    with open('svm_model.pkl', 'wb') as f:
        pickle.dump((model, le), f)

    # Evaluate the model
    print("[INFO] evaluating SVM model...")
    predictions = model.predict(testX)
    print(classification_report(testY, predictions, target_names=le.classes_))

    # Testing on a single image
    predicted_name = predict_single_image('2024-07-19-090128.jpg', embedder, model, le)
    print(f"Predicted name: {predicted_name}")

if __name__ == "__main__":
    main()
