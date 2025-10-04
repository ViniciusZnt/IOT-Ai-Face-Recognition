import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import face_recognition
import numpy as np
import os

# Baixar modelo YOLOv8 de detecção de rostos
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
model = YOLO(model_path)

# Carregar rostos conhecidos da pasta ./Faces
known_face_encodings = []
known_face_names = []
known_faces_dir = './Faces'
frame_count = 0
process_evey_frame = 2

for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpeg"):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            known_face_encodings.append(face_encodings[0])
            name = os.path.splitext(filename)[0]  # nome = nome do arquivo
            known_face_names.append(name)

# Abrir webcam
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
# cap.set(cv2.CAP_PROP_FPS, 15)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Reduzir resolução do frame
    frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    # YOLO detecta rostos
    results = model(frame_small, verbose=False)

    for result in results:
        for box in result.boxes:
            # Extrair coordenadas corretamente
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Recorte do rosto detectado
            face_frame = frame_small[y1:y2, x1:x2]

            # Ignorar rostos muito pequenos
            if face_frame.shape[0] < 30 or face_frame.shape[1] < 30:
                continue

            # Converter para RGB (face_recognition usa RGB)
            rgb_face = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)

            # Extrair encoding do rosto recortado
            face_encodings = face_recognition.face_encodings(rgb_face)

            name = "Unknown"
            if face_encodings:
                face_encoding = face_encodings[0]

                # Comparar com encodings conhecidos
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                if True in matches:
                    match_index = matches.index(True)
                    name = known_face_names[match_index]

            # Desenhar box + nome no frame pequeno
            cv2.rectangle(frame_small, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_small, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

    frame_count += 1
    if frame_count % process_evey_frame != 0:
        cv2.imshow("Reconhecimento Facial YOLO + FaceRec", cv2.resize(frame_small, (640, 480)))
        continue

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
