from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import cv2

# 1️⃣ Baixar modelo
model_path = hf_hub_download(
    repo_id="arnabdhar/YOLOv8-Face-Detection",
    filename="model.pt"
)

# 2️⃣ Carregar modelo
model = YOLO(model_path)

# 3️⃣ Abrir webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: não foi possível acessar a câmera")
    exit()

# 4️⃣ Nome fixo para a janela
window_name = "Detecção de Rostos - Pressione 'q' para sair"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Cria a janela apenas uma vez

while True:
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível ler o frame da câmera.")
        break

    # Rodar detecção
    results = model(frame, verbose=False)

    # Desenhar boxes
    annotated_frame = results[0].plot()

    # Ultralytics retorna RGB, OpenCV precisa BGR
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Mostrar na mesma janela
    cv2.imshow(window_name, annotated_frame)

    # Sair quando pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
