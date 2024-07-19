import cv2
import numpy as np
import os
from imutils import paths
import albumentations as A

def detect_faces(net, image, minConfidence=0.5):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > minConfidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            boxes.append((startX, startY, endX, endY))
    return boxes

def load_face_dataset(inputPath, net, minConfidence=0.5, minSamples=15):
    def augment_image(image):
        transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
        ])
        augmented = transform(image=image)
        return augmented['image']
    
    def apply_directional_lighting(image, intensity=0.3, direction=(1, 0)):
        h, w = image.shape[:2]
        mask = np.zeros_like(image, dtype=np.float32)
        cv2.line(mask, (w//2, h//2), (w//2 + int(direction[0]*w), h//2 + int(direction[1]*h)), (1, 1, 1), thickness=20)
        mask = cv2.GaussianBlur(mask, (101, 101), 0)
        lighting_effect = cv2.addWeighted(image.astype(np.float32), 1.0, mask * 255 * intensity, intensity, 0)
        return lighting_effect.astype(np.uint8)

    imagePaths = list(paths.list_images(inputPath))
    print(len(imagePaths))
    names = [p.split(os.path.sep)[-2] for p in imagePaths]
    (names, counts) = np.unique(names, return_counts=True)
    names = names.tolist()

    faces = []
    labels = []

    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        name = imagePath.split(os.path.sep)[-2]

        if counts[names.index(name)] < minSamples:
            continue

        boxes = detect_faces(net, image, minConfidence)
        for (startX, startY, endX, endY) in boxes:
            faceROI = image[startY:endY, startX:endX]
            faceROI = cv2.resize(faceROI, (160, 160))
            # faceROI = cv2.GaussianBlur(faceROI, (5, 5), 0)
            faceROI_grey = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)
            faceROI = np.stack([faceROI_grey] * 3, axis=-1)

            # Original face
            faces.append(faceROI_grey)
            labels.append(name)

            # Augmented face
            augmented_face = augment_image(faceROI)
            augmented_face_grey = cv2.cvtColor(augmented_face, cv2.COLOR_BGR2GRAY)
            faces.append(augmented_face_grey)
            labels.append(name)

            # Directional lighting
            for direction in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                lighting_face = apply_directional_lighting(faceROI, intensity=0.3, direction=direction)
                lighting_face_grey = cv2.cvtColor(lighting_face, cv2.COLOR_BGR2GRAY)
                faces.append(lighting_face_grey)
                labels.append(name)

    faces = np.array(faces)
    labels = np.array(labels)

    print(len(faces))

    return faces, labels











# TESTING ##

# inputPath = "../../Faces/sriram"  # Provide the path to the directory containing the face images
# net = cv2.dnn.readNet('../../Detector_model/deploy.prototxt', '../../Detector_model/res10_300x300_ssd_iter_140000.caffemodel')  # Initialize the face detection neural network
# minConfidence = 0.5  # Minimum confidence threshold for face detection (optional, default is 0.5)
# minSamples = 15  # Minimum number of samples required per face class (optional, default is 15)

# (faces,labels) = load_face_dataset(inputPath,net,minConfidence,minSamples)
# # to display image

# print(labels.size)

# for face, label in zip(faces, labels):
#     img = cv2.imshow(label, face)
#     cv2.waitKey(0)

# cv2.destroyAllWindows() 


