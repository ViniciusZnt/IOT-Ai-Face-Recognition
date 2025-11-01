import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import face_recognition
import numpy as np
import os
from collections import deque, defaultdict
import time

# ===== CONFIGURAÇÕES DE PERFORMANCE =====
FRAME_WIDTH = 416  # Aumentado para melhor qualidade
FRAME_HEIGHT = 416
PROCESS_EVERY_N_FRAMES = 2  # Reduzido para mais frequência
RECOGNITION_EVERY_N_DETECTIONS = 3  # Reconhecer menos vezes, mas com mais qualidade
MIN_FACE_SIZE = 50  # Aumentado para ignorar rostos muito distantes
RECOGNITION_CACHE_TIME = 5.0  # Mais tempo de cache
TOLERANCE = 0.55  # Mais tolerante (0.6 = padrão, 0.5-0.55 = mais permissivo)
CONFIDENCE_THRESHOLD = 0.4  # YOLO: aceitar detecções mais fracas
STABILIZATION_FRAMES = 3  # Quantos frames para confirmar um reconhecimento

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
        image = face_recognition.load_image_file(image_path)
        
        # Garantir que a imagem tem tamanho adequado
        if image.shape[0] < 100 or image.shape[1] < 100:
            print(f" {filename} - Imagem muito pequena, considere usar foto maior")
        
        face_encodings = face_recognition.face_encodings(image, num_jitters=2)  # Mais precisão
        if face_encodings:
            known_face_encodings.append(face_encodings[0])
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {filename} - Nenhum rosto detectado! Use foto frontal com boa iluminação")

if len(known_face_names) == 0:
    print("\nERRO: Nenhum rosto foi carregado! Verifique as imagens em ./Faces")
    exit()

print(f"\n✓ Total de rostos cadastrados: {len(known_face_names)}")

# ===== SISTEMA DE TRACKING E ESTABILIZAÇÃO =====
class FaceTracker:
    def __init__(self, cache_time=5.0, stabilization_frames=3):
        self.tracks = {}  # {track_id: {'name': str, 'last_seen': float, 'votes': Counter}}
        self.cache_time = cache_time
        self.stabilization_frames = stabilization_frames
        self.next_track_id = 0
        
    def find_track(self, x, y, tolerance=80):
        """Encontra track existente próximo a essa posição"""
        current_time = time.time()
        best_track = None
        best_distance = tolerance
        
        for track_id, data in list(self.tracks.items()):
            # Limpar tracks antigos
            if current_time - data['last_seen'] > self.cache_time:
                del self.tracks[track_id]
                continue
            
            # Calcular distância
            tx, ty = data['position']
            distance = np.sqrt((tx - x)**2 + (ty - y)**2)
            
            if distance < best_distance:
                best_distance = distance
                best_track = track_id
        
        return best_track
    
    def update_track(self, track_id, x, y, name):
        """Atualiza ou cria track"""
        current_time = time.time()
        
        if track_id is None:
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = {
                'position': (x, y),
                'votes': defaultdict(int),
                'last_seen': current_time,
                'confirmed_name': None
            }
        
        # Atualizar posição e tempo
        self.tracks[track_id]['position'] = (x, y)
        self.tracks[track_id]['last_seen'] = current_time
        
        # Adicionar voto
        if name != "Desconhecido":
            self.tracks[track_id]['votes'][name] += 1
            
            # Confirmar nome se tiver votos suficientes
            if self.tracks[track_id]['votes'][name] >= self.stabilization_frames:
                self.tracks[track_id]['confirmed_name'] = name
        
        return track_id
    
    def get_name(self, track_id):
        """Retorna nome confirmado do track"""
        if track_id in self.tracks:
            confirmed = self.tracks[track_id]['confirmed_name']
            if confirmed:
                return confirmed
            
            # Se não confirmado, retornar o mais votado
            votes = self.tracks[track_id]['votes']
            if votes:
                return max(votes, key=votes.get)
        
        return "Desconhecido"

tracker = FaceTracker(RECOGNITION_CACHE_TIME, STABILIZATION_FRAMES)

# ===== CONFIGURAR CÂMERA =====
print("\nIniciando câmera...")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 15)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Auto-ajuste de exposição e foco (se disponível)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

# ===== VARIÁVEIS DE CONTROLE =====
frame_count = 0
detection_count = 0
fps_counter = deque(maxlen=30)
last_time = time.time()
recognized_people = set()  # Para controle de presenças únicas

print("\nSistema iniciado!")
print(f"Configurações: Tolerância={TOLERANCE}, Estabilização={STABILIZATION_FRAMES} frames")
print("Pressione 'q' para sair, 'r' para resetar reconhecimentos\n")

# ===== LOOP PRINCIPAL =====
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar frame")
        break
    
    frame_count += 1
    
    # Calcular FPS
    current_time = time.time()
    fps = 1.0 / (current_time - last_time)
    fps_counter.append(fps)
    last_time = current_time
    avg_fps = np.mean(fps_counter)
    
    # Processar apenas frames selecionados
    process_this_frame = (frame_count % PROCESS_EVERY_N_FRAMES == 0)
    
    if process_this_frame:
        # YOLO: Detectar rostos
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
        
        for result in results:
            for box in result.boxes:
                # Extrair coordenadas e confiança
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                w, h = x2 - x1, y2 - y1
                
                # Ignorar rostos muito pequenos
                if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                    continue
                
                # Centro do rosto
                cx, cy = x1 + w // 2, y1 + h // 2
                
                # Buscar track existente
                track_id = tracker.find_track(cx, cy)
                
                # Fazer reconhecimento periodicamente
                should_recognize = (detection_count % RECOGNITION_EVERY_N_DETECTIONS == 0)
                
                if should_recognize:
                    # Recortar rosto com margem generosa
                    margin = 20
                    y1_crop = max(0, y1 - margin)
                    y2_crop = min(frame.shape[0], y2 + margin)
                    x1_crop = max(0, x1 - margin)
                    x2_crop = min(frame.shape[1], x2 + margin)
                    
                    face_frame = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                    
                    # Converter para RGB
                    rgb_face = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
                    
                    # Extrair encoding (sem jitter para velocidade)
                    face_encodings = face_recognition.face_encodings(rgb_face, num_jitters=0)
                    
                    name = "Desconhecido"
                    best_distance = 1.0
                    
                    if face_encodings:
                        face_encoding = face_encodings[0]
                        
                        # Calcular distâncias
                        face_distances = face_recognition.face_distance(
                            known_face_encodings, face_encoding
                        )
                        
                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)
                            best_distance = face_distances[best_match_index]
                            
                            if best_distance < TOLERANCE:
                                name = known_face_names[best_match_index]
                                confidence_pct = (1 - best_distance) * 100
                                print(f"✓ {name} detectado (confiança: {confidence_pct:.1f}%, distância: {best_distance:.3f})")
                    else:
                        print(f"⚠️  Face encoding falhou (rosto parcial ou borrado?)")
                    
                    # Atualizar track
                    track_id = tracker.update_track(track_id, cx, cy, name)
                    
                    # Registrar pessoa reconhecida
                    if name != "Desconhecido" and name not in recognized_people:
                        recognized_people.add(name)
                        print(f"\nPRESENÇA REGISTRADA: {name}")
                        print(f"   Total de pessoas presentes: {len(recognized_people)}\n")
                        # TODO: Aqui chamar função SSH/IoT
                        # registrar_presenca(name)
                else:
                    # Não reconhecer, apenas atualizar posição
                    if track_id is not None:
                        track_id = tracker.update_track(track_id, cx, cy, "")
                
                detection_count += 1
                
                # Obter nome confirmado do track
                display_name = tracker.get_name(track_id) if track_id is not None else "Detectando..."
                
                # Cores: verde = reconhecido, amarelo = detectando, vermelho = desconhecido
                if display_name != "Desconhecido" and display_name != "Detectando...":
                    color = (0, 255, 0)  # Verde
                elif display_name == "Detectando...":
                    color = (0, 255, 255)  # Amarelo
                else:
                    color = (0, 0, 255)  # Vermelho
                
                # Desenhar retângulo
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Desenhar nome e confiança do YOLO
                label = f"{display_name} ({confidence:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Informações na tela
    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Presentes: {len(recognized_people)}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Reconhecimento Facial", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        recognized_people.clear()
        print("\nReconhecimentos resetados\n")

# ===== LIMPEZA =====
cap.release()
cv2.destroyAllWindows()
print(f"\n✓ Sistema encerrado. Total de presenças registradas: {len(recognized_people)}")
if recognized_people:
    print(f"   Pessoas: {', '.join(recognized_people)}")