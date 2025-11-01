import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import face_recognition
import numpy as np
import os
from collections import defaultdict
import time

# ===== CONFIGURAÇÕES =====
FRAME_WIDTH = 640  # Resolução padrão (melhor para Raspberry Pi)
FRAME_HEIGHT = 480
PROCESS_EVERY_N_FRAMES = 2
RECOGNITION_INTERVAL = 5  # Reconhecer 1 a cada 5 detecções
MIN_FACE_SIZE = 70  # Ignorar rostos muito pequenos/distantes
TOLERANCE = 0.45  # MUITO permissivo (teste começar aqui)
CONFIDENCE_THRESHOLD = 0.30  # YOLO mais permissivo
IOU_THRESHOLD = 0.25  # Tracking mais permissivo
MIN_TRACK_HITS = 3  # Mínimo de hits para considerar track válido
MAX_TRACK_AGE = 45  # Frames máximos sem detecção (1.5s a 30fps)

# ===== CARREGAR MODELO YOLO =====
print("Carregando modelo YOLO...")
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
model = YOLO(model_path)
model.fuse()

# ===== CARREGAR ROSTOS CONHECIDOS =====
print("\nCarregando rostos conhecidos...")
known_face_encodings = []
known_face_names = []
known_faces_dir = './Faces'

for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith((".jpeg", ".jpg", ".png")):
        image_path = os.path.join(known_faces_dir, filename)
        print(f"   Processando {filename}...", end=" ")
        
        image = face_recognition.load_image_file(image_path)
        
        # Mudar Canal da Imagem para BGR
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # CLAHE(Contrast Limited Adaptive Histogram Equalization) algorithm
        # Usado para melhorar contraste e melhorar a qualidade da imagem
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        image_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Gerar encodings com LARGE model e mais jitters
        face_encodings = face_recognition.face_encodings(
            image, 
            num_jitters=2,  # Mais precisão (mas mais lento no carregamento)
            model='large'
        )
        
        if face_encodings:
            known_face_encodings.append(face_encodings[0])
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)
            print("✓", end=" ")
        else:
            print("SEM ROSTO!")

if len(known_face_names) == 0:
    print("\nERRO: Nenhum rosto foi carregado!")
    print("Execute 'diagnostic_test.py' primeiro!")
    exit()

print(f"\n✓ {len(known_face_names)} rostos: {', '.join(known_face_names)}\n")

# ===== KALMAN FILTER SIMPLIFICADO =====
class KalmanBoxTracker:
    def __init__(self, bbox):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                              [0, 1, 0, 1],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.kf.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        
        self.bbox = bbox
        self.age = 0
        self.hits = 0
        
    def predict(self):
        self.age += 1
        predicted = self.kf.predict()
        cx, cy = predicted[0, 0], predicted[1, 0]
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
    
    def update(self, bbox):
        self.age = 0
        self.hits += 1
        self.bbox = bbox
        
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.kf.correct(np.array([[cx], [cy]], np.float32))

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

# ===== TRACKER ROBUSTO =====
class RobustFaceTracker:
    def __init__(self):
        self.trackers = {}
        self.next_id = 0
        
    def update(self, detections):
        """
        detections: [(x1, y1, x2, y2, name, conf), ...]
        retorna: [(x1, y1, x2, y2, track_id, confirmed_name, stability), ...]
        """
        # Predizer posições dos trackers existentes
        predicted_boxes = {}
        to_delete = []
        
        for track_id, data in self.trackers.items():
            predicted = data['kalman'].predict()
            predicted_boxes[track_id] = predicted
            
            # Remover tracks muito antigos
            if data['kalman'].age > MAX_TRACK_AGE:
                to_delete.append(track_id)
        
        for tid in to_delete:
            del self.trackers[tid]
        
        # Associar detecções com tracks
        matched_tracks = set()
        matched_detections = set()
        
        # Matriz de custos (IOU)
        cost_matrix = np.zeros((len(detections), len(predicted_boxes)))
        track_ids = list(predicted_boxes.keys())
        
        for i, (x1, y1, x2, y2, name, conf) in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                pred_box = predicted_boxes[track_id]
                iou = calculate_iou([x1, y1, x2, y2], pred_box)
                cost_matrix[i, j] = iou
        
        # Matching guloso (Hungarian seria melhor, mas mais complexo)
        results = []
        
        for i in range(len(detections)):
            if i in matched_detections:
                continue
            
            best_j = -1
            best_iou = IOU_THRESHOLD
            
            for j in range(len(track_ids)):
                if j in matched_tracks:
                    continue
                if cost_matrix[i, j] > best_iou:
                    best_iou = cost_matrix[i, j]
                    best_j = j
            
            x1, y1, x2, y2, name, conf = detections[i]
            
            if best_j >= 0:
                # Match encontrado - atualizar tracker
                track_id = track_ids[best_j]
                tracker_data = self.trackers[track_id]
                tracker_data['kalman'].update([x1, y1, x2, y2])
                
                # Atualizar nome com votação
                if name and name != "Desconhecido":
                    tracker_data['name_votes'][name] += 1
                    total_votes = sum(tracker_data['name_votes'].values())
                    
                    # Confirmar se tem maioria clara
                    if tracker_data['name_votes'][name] >= 2:
                        tracker_data['confirmed_name'] = name
                
                matched_tracks.add(best_j)
                matched_detections.add(i)
                
                # Calcular estabilidade
                stability = min(tracker_data['kalman'].hits / 10.0, 1.0)
                
                results.append((
                    x1, y1, x2, y2,
                    track_id,
                    tracker_data['confirmed_name'],
                    stability
                ))
            else:
                # Criar novo tracker
                track_id = self.next_id
                self.next_id += 1
                
                kalman = KalmanBoxTracker([x1, y1, x2, y2])
                
                self.trackers[track_id] = {
                    'kalman': kalman,
                    'name_votes': defaultdict(int),
                    'confirmed_name': None
                }
                
                if name and name != "Desconhecido":
                    self.trackers[track_id]['name_votes'][name] = 1
                
                matched_detections.add(i)
                
                results.append((x1, y1, x2, y2, track_id, None, 0.0))
        
        # Adicionar trackers não-matcheados mas ainda válidos (predição)
        for j, track_id in enumerate(track_ids):
            if j not in matched_tracks:
                tracker_data = self.trackers[track_id]
                if tracker_data['kalman'].hits >= MIN_TRACK_HITS:
                    pred_box = predicted_boxes[track_id]
                    x1, y1, x2, y2 = map(int, pred_box)
                    
                    stability = min(tracker_data['kalman'].hits / 10.0, 1.0)
                    
                    results.append((
                        x1, y1, x2, y2,
                        track_id,
                        tracker_data['confirmed_name'],
                        stability * 0.5  # Reduzir estabilidade de predições
                    ))
        
        return results

tracker = RobustFaceTracker()

# ===== CÂMERA =====
print("\nIniciando câmera...")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

# Warm-up
for _ in range(30):
    cap.read()

print("\n✓ Sistema pronto!")
print(f"Tolerância: {TOLERANCE} | IOU: {IOU_THRESHOLD}")
print("Teclas: q= sair | r= reset | += tolerância+ | -= tolerância-\n")

# ===== LOOP =====
frame_count = 0
detection_count = 0
recognized_people = set()
last_log = {}

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame_count += 1
    process = (frame_count % PROCESS_EVERY_N_FRAMES == 0)
    
    detections = []
    
    if process:
        # CLAHE
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        frame_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # YOLO
        results = model(frame_enhanced, verbose=False, conf=CONFIDENCE_THRESHOLD)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w, h = x2 - x1, y2 - y1
                
                if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                    continue
                
                name = ""
                should_recognize = (detection_count % RECOGNITION_INTERVAL == 0)
                
                if should_recognize:
                    # Crop com margem
                    margin = 20
                    y1_c = max(0, y1 - margin)
                    y2_c = min(frame.shape[0], y2 + margin)
                    x1_c = max(0, x1 - margin)
                    x2_c = min(frame.shape[1], x2 + margin)
                    
                    face = frame_enhanced[y1_c:y2_c, x1_c:x2_c]
                    rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    
                    # Encoding SEM jitter (velocidade)
                    encs = face_recognition.face_encodings(rgb_face, num_jitters=0, model='large')
                    
                    if encs:
                        distances = face_recognition.face_distance(known_face_encodings, encs[0])
                        
                        if len(distances) > 0:
                            best_idx = np.argmin(distances)
                            best_dist = distances[best_idx]
                            
                            if best_dist < TOLERANCE:
                                name = known_face_names[best_idx]
                                conf_pct = (1 - best_dist) * 100
                                
                                # Log sem spam
                                now = time.time()
                                if name not in last_log or now - last_log[name] > 3:
                                    print(f"✓ {name} | {conf_pct:.0f}% | dist={best_dist:.3f}")
                                    last_log[name] = now
                
                detection_count += 1
                detections.append((x1, y1, x2, y2, name, 1.0))
    
    # Tracking
    tracked = tracker.update(detections)
    
    # Desenhar
    for x1, y1, x2, y2, tid, confirmed, stability in tracked:
        if confirmed:
            label = confirmed
            color = (0, 255, 0)  # Verde sólido
            thickness = 3
            
            # Registrar
            if confirmed not in recognized_people:
                recognized_people.add(confirmed)
                print(f"\nPRESENÇA: {confirmed}\n")
        elif stability > 0.3:
            label = f"#{tid}"
            color = (0, 255, 255)  # Amarelo
            thickness = 2
        else:
            label = f"#{tid}"
            color = (128, 128, 128)  # Cinza
            thickness = 1
        
        # Box SEMPRE visível (não pisca!)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Label com fundo
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # HUD
    cv2.putText(frame, f"Presentes: {len(recognized_people)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Tol: {TOLERANCE:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("Reconhecimento", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        recognized_people.clear()
        print("\nReset\n")
    elif key == ord('+'):
        TOLERANCE += 0.05
        print(f"Tolerância: {TOLERANCE:.2f}")
    elif key == ord('-'):
        TOLERANCE = max(0.3, TOLERANCE - 0.05)
        print(f"Tolerância: {TOLERANCE:.2f}")

cap.release()
cv2.destroyAllWindows()
print(f"\n✓ Fim | Presenças: {', '.join(recognized_people) if recognized_people else 'nenhuma'}")