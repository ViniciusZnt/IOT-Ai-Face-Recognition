from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import cv2

1️⃣ Importações

    hf_hub_download: baixa arquivos de modelos hospedados no Hugging Face.

    YOLO (Ultralytics): carrega e executa modelos YOLOv8 para detecção de objetos.

    cv2 (OpenCV): biblioteca para manipulação de imagens, vídeos e interfaces de vídeo (mostrar janelas, capturar webcam etc).

# 1️⃣ Baixar modelo
model_path = hf_hub_download(
    repo_id="arnabdhar/YOLOv8-Face-Detection",
    filename="model.pt"
)

    repo_id → nome do repositório do Hugging Face.

    filename → arquivo específico dentro do repo.

    hf_hub_download baixa o arquivo .pt para cache local e retorna o caminho local, que depois será usado pelo YOLO.

# 2️⃣ Carregar modelo
model = YOLO(model_path)

    YOLO(model_path) carrega os pesos do modelo .pt no Python.

    O objeto model permite fazer inferência (model(frame)) em imagens ou vídeos.

# 3️⃣ Abrir webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: não foi possível acessar a câmera")
    exit()

    cv2.VideoCapture(0) abre a webcam padrão.

        0 → primeira câmera conectada.

        Se houver outra câmera, pode usar 1, 2 etc.

    cap.isOpened() verifica se a câmera abriu corretamente.

    exit() encerra o programa se não conseguiu acessar a câmera.

# 4️⃣ Nome fixo para a janela
window_name = "Detecção de Rostos - Pressione 'q' para sair"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Cria a janela apenas uma vez

    cv2.namedWindow() cria uma janela de exibição única antes do loop.

    cv2.WINDOW_NORMAL permite redimensionar a janela.

    Importante: não criar a janela dentro do loop, isso evita múltiplas janelas.

while True:
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível ler o frame da câmera.")
        break

    cap.read() captura um frame da câmera.

        ret → booleano indicando sucesso da captura.

        frame → array NumPy representando a imagem capturada.

    Se não conseguir capturar (ret=False), o loop termina com break.

# Rodar detecção
results = model(frame, verbose=False)

    model(frame) executa a detecção YOLOv8 no frame atual.

    verbose=False evita prints detalhados no terminal.

    results → lista de resultados, geralmente com apenas um item (results[0]) por imagem.

        Cada item contém caixas (boxes), classes (cls) e scores (conf).

# Desenhar boxes
annotated_frame = results[0].plot()

    results[0].plot() retorna uma cópia da imagem com as caixas desenhadas.

    Cada rosto detectado terá uma bounding box verde (ou padrão do modelo) e possivelmente a confiança (score) exibida.

# Ultralytics retorna RGB, OpenCV precisa BGR
annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    OpenCV usa BGR em vez de RGB.

    cv2.cvtColor() converte de RGB → BGR.

    Se não fizer isso, o OpenCV pode abrir novas janelas ou exibir cores erradas.

# Mostrar na mesma janela
cv2.imshow(window_name, annotated_frame)

    Atualiza a janela já criada (window_name) com o frame anotado.

    Importante: mesma janela a cada frame → evita múltiplas janelas.

# Sair quando pressionar 'q'
if cv2.waitKey(1) & 0xFF == ord('q'):
    break

    cv2.waitKey(1) espera 1 ms por uma tecla.

    & 0xFF == ord('q') verifica se a tecla q foi pressionada.

    Se sim, sai do loop (break) e encerra o programa.

    Isso permite interromper a execução em tempo real.

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

    cap.release() libera a webcam.

    cv2.destroyAllWindows() fecha todas as janelas OpenCV abertas.

    Essencial para evitar que a câmera fique “presa” ou janelas abertas após o programa terminar.
