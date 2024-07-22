import cv2
import torch
from deepface import DeepFace

model = torch.hub.load('ultralytics/yolov8', 'yolov8n', pretrained=True)
def load_deepface_model():
    try:
        # Attempt to load a pre-trained model from DeepFace
        return DeepFace.build_model('VGG-Face')
    except:
        # If loading fails, provide instructions for custom model creation
        print("DeepFace model loading failed. Please consider these options:")
        print("- Create a custom face recognition model using DeepFace utilities.")
        print("- Download a pre-trained DeepFace model and place it in a suitable location.")
        return None

deepface_model = load_deepface_model()

def detect_and_recognize(image):
    if deepface_model is None:
        return None

    # Preprocess image for YOLOv8
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform face detection using YOLOv8
    results = model(image)

    # Extract face recognition results (if any faces are detected)
    faces = []
    for detection in results.pandas().xyxy[0]:
        if detection['name'] == '0':  # Assuming face class index is 0 (modify if different)
            x_min, y_min, x_max, y_max, conf, cls = detection

            # Crop the detected face region
            face_image = image[y_min:y_max, x_min:x_max]

            # Perform face recognition using DeepFace
            try:
                recognition_result = deepface_model.predict(face_image)
                recognition_result['x_min'] = x_min
                recognition_result['y_min'] = y_min
                recognition_result['x_max'] = x_max
                recognition_result['y_max'] = y_max
                faces.append(recognition_result)
            except:
                print("Error during face recognition. Skipping this face.")

    return faces

# Example usage (assuming you have an image loaded as 'image')
recognized_faces = detect_and_recognize(image)

if recognized_faces:
    # Process recognized faces (display results, store in database, etc.)
    for face in recognized_faces:
        print(f"Face recognized: {face['Optional.Name']}")
        # Draw bounding box and label (optional)
        cv2.rectangle(image, (face['x_min'], face['y_min']), (face['x_max'], face['y_max']), (0, 255, 0), 2)
        cv2.putText(image, face['Optional.Name'], (face['x_min'] + 5, face['y_min'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
else:
    print("No faces detected or DeepFace model loading failed.")