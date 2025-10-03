# load libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
import cv2

# 1️⃣ Baixar o modelo do Hugging Face (se já baixado, ele só reutiliza o cache)
model_path = hf_hub_download(
    repo_id="arnabdhar/YOLOv8-Face-Detection",
    filename="model.pt"
)

# 2️⃣ Carregar o modelo YOLO
model = YOLO(model_path)

# 3️⃣ Caminho de uma imagem específica
image_path = (
    "/home/manja_zana/YOLOv8-Face-Detection/WIDER FACE.v1i.yolov8/test/images/"
    "0_Parade_marchingband_1_5_jpg.rf.fe01b5faf6f1786fad361d0daadf3e0c.jpg"
)

# 4️⃣ Carregar a imagem
img = Image.open(image_path)

# 5️⃣ Fazer inferência (detecção)
output = model(img)

# 6️⃣ Converter resultados para Supervision (opcional, se quiser manipular as boxes)
results = Detections.from_ultralytics(output[0])
print("Detecções:", results)  # apenas para ver as coordenadas no terminal

# 7️⃣ Gerar imagem com caixas já desenhadas
annotated_frame = output[0].plot()

# 8️⃣ Mostrar em uma janela OpenCV
cv2.imshow("Rostos Detectados", annotated_frame)
cv2.waitKey(0)  # espera até qualquer tecla
cv2.destroyAllWindows()

# 9️⃣ (Opcional) Salvar a imagem com caixas
cv2.imwrite("saida_com_boxes.jpg", annotated_frame)
print("Imagem anotada salva como: saida_com_boxes.jpg")
