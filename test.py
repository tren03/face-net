import cv2
import numpy as np
import pickle
from keras_facenet import FaceNet
from pyimagesearch.faces import detect_faces

def preprocess_image(faceROI_grey):
    # Convert to 3-channel RGB and resize to 160x160
    faceROI_rgb = cv2.cvtColor(faceROI_grey, cv2.COLOR_GRAY2RGB)
    faceROI_rgb = cv2.resize(faceROI_rgb, (160, 160))
    
    # Convert to float32 and add batch dimension
    faceROI_rgb = faceROI_rgb.astype(np.float32)
    faceROI_rgb = np.expand_dims(faceROI_rgb, axis=0)
    
    return faceROI_rgb

def load_model_and_label_encoder(model_path):
    # Load the pickled model and label encoder
    with open(model_path, 'rb') as f:
        model, le = pickle.load(f)
    return model, le

def predict_single_image(faceROI, embedder, svm_model, label_encoder, threshold=0.6):
    # Convert to grayscale and preprocess the image
    faceROI_gray = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)
    processed_image = preprocess_image(faceROI_gray)
    
    # Generate embeddings using FaceNet model
    embeddings = embedder.embeddings(processed_image)
    
    # Predict using SVM model and get confidence scores
    predictions_proba = svm_model.decision_function(embeddings)
    predictions = svm_model.predict(embeddings)
    
    # Decode predictions using label encoder
    predicted_label = label_encoder.inverse_transform(predictions)[0]
    confidence = np.max(predictions_proba)  # Highest confidence score
    
    # Check if confidence meets the threshold
    if confidence < threshold:
        return "Unknown", confidence
    
    return predicted_label, confidence  

def main():
    # Load the model and label encoder
    model, le = load_model_and_label_encoder('svm_model.pkl')
    
    # Initialize the FaceNet model
    embedder = FaceNet()
    
    # Load the face detector model
    net = cv2.dnn.readNet('face_detector/deploy.prototxt', 'face_detector/res10_300x300_ssd_iter_140000.caffemodel')
    
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame
        boxes = detect_faces(net, frame)  # Get bounding boxes from the detect_faces function
        
        # Loop over detected faces
        for (startX, startY, endX, endY) in boxes:
            faceROI = frame[startY:endY, startX:endX]
            
            # Ensure the face ROI is valid
            if faceROI.shape[0] > 0 and faceROI.shape[1] > 0:
                # Predict the face and get confidence
                predicted_name, confidence = predict_single_image(faceROI, embedder, model, le)
                
                # Draw bounding box and label with confidence on the frame
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label_text = f"{predicted_name} ({confidence:.2f})"
                cv2.putText(frame, label_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Face Recognition', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
