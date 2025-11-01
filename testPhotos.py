import face_recognition
import cv2
import os
import numpy as np

print("=" * 60)
print("🔍 DIAGNÓSTICO DE FOTOS DE REFERÊNCIA")
print("=" * 60)

known_faces_dir = './Faces'

# Testar cada foto
for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith((".jpeg", ".jpg", ".png")):
        print(f"\n📸 Testando: {filename}")
        image_path = os.path.join(known_faces_dir, filename)
        
        # Carregar imagem
        image = face_recognition.load_image_file(image_path)
        print(f"   Resolução: {image.shape[1]}x{image.shape[0]} pixels")
        
        # Detectar rostos
        face_locations = face_recognition.face_locations(image, model='hog')
        print(f"   Rostos detectados (HOG): {len(face_locations)}")
        
        face_locations_cnn = face_recognition.face_locations(image, model='cnn')
        print(f"   Rostos detectados (CNN): {len(face_locations_cnn)}")
        
        # Gerar encodings
        encodings_small = face_recognition.face_encodings(image, num_jitters=1, model='small')
        encodings_large = face_recognition.face_encodings(image, num_jitters=1, model='large')
        
        print(f"   Encodings (small model): {len(encodings_small)}")
        print(f"   Encodings (large model): {len(encodings_large)}")
        
        # Teste de consistência (mesma foto deve dar distância ~0)
        if len(encodings_large) > 0:
            self_dist = face_recognition.face_distance([encodings_large[0]], encodings_large[0])[0]
            print(f"   Auto-distância: {self_dist:.6f} (ideal: ~0.000)")
            
            if self_dist > 0.01:
                print("   ⚠️  AVISO: Auto-distância alta! Foto pode ter problemas.")
        
        # Mostrar foto com detecção
        if len(face_locations) > 0:
            img_display = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(img_display, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Redimensionar se muito grande
            max_width = 800
            if img_display.shape[1] > max_width:
                scale = max_width / img_display.shape[1]
                new_width = int(img_display.shape[1] * scale)
                new_height = int(img_display.shape[0] * scale)
                img_display = cv2.resize(img_display, (new_width, new_height))
            
            cv2.imshow(f"Detecção: {filename}", img_display)
            print(f"   ✓ Mostrando detecção (pressione qualquer tecla)")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("   ❌ NENHUM ROSTO DETECTADO! Esta foto NÃO funcionará!")
            print("      Sugestão: Tire uma nova foto frontal com boa iluminação")

print("\n" + "=" * 60)
print("📊 RESUMO E RECOMENDAÇÕES")
print("=" * 60)
print("""
✅ FOTO BOA:
   - Resolução >= 300x300
   - 1 rosto detectado (HOG e CNN)
   - Auto-distância < 0.01
   - Rosto frontal, bem iluminado

❌ FOTO RUIM:
   - Resolução < 200x200
   - 0 rostos detectados
   - Rosto de perfil/lateral
   - Sombras fortes, muito escuro

💡 DICA: Use o celular para tirar fotos novas:
   - Fundo claro e liso
   - Luz natural ou ambiente bem iluminado
   - Rosto ocupando boa parte da foto
   - Olhar diretamente para câmera
   - Salvar como JPEG ou PNG
""")