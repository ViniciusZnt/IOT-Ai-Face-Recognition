# IOT-Ai-Face-Recognition

Este projeto tem como objetivo o **monitoramento de frequência utilizando reconhecimento facial com IoT**.  
Ele combina visão computacional com inteligência artificial para identificar rostos e registrar presença de forma automática.

---

## 🚀 Funcionalidades
- Reconhecimento facial em tempo real
- Registro automático de presença
- Integração com dispositivos IoT
- Interface de fácil uso para monitoramento

---

## 📦 Instalação

Clone este repositório:
```bash
git clone https://github.com/ViniciusZnt/IOT-Ai-Face-Recognition.git
cd IOT-Ai-Face-Recognition
```

Crie e ative um ambiente virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

Instale as dependências:
```bash
pip install -r requirements.txt
```

---

## ⚙️ Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:

```env
# Porta em que a aplicação vai rodar
PORT=5000

# URL de conexão com o banco de dados (exemplo: PostgreSQL, MySQL, etc.)
DATABASE_URL=mysql://user:password@localhost:3306/iot_face_recognition

# Chave secreta para autenticação / JWT
SECRET_KEY=sua_chave_secreta_aqui

# Caminho para o modelo de IA treinado
MODEL_PATH=./model.pt
```

---

## ▶️ Execução

Depois de instalar as dependências e configurar o `.env`, basta rodar:

```bash
python app.py
```

Ou, caso o projeto use Flask/FastAPI:
```bash
uvicorn app:app --reload
```

---

## 🛠️ Dependências principais

As bibliotecas usadas (exemplo, ajuste conforme seu código):

- opencv-python → Processamento de imagem e captura da câmera
- torch ou tensorflow → Modelo de IA para reconhecimento facial
- ultralytics → YOLOv8 para detecção facial
- python-dotenv → Carregar variáveis de ambiente
- sqlalchemy ou pymysql → Integração com banco de dados
- fastapi ou flask → API da aplicação

Instale com:
```bash
pip install opencv-python torch ultralytics python-dotenv fastapi uvicorn sqlalchemy
```

---

## 📌 Próximos Passos
- Treinar modelo de reconhecimento facial com dataset próprio
- Melhorar integração com IoT (ESP32, Raspberry Pi, etc.)
- Criar interface web para monitoramento

---

## 👨‍💻 Autor
Projeto desenvolvido por **Vinicius Zanatta**.
