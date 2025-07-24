# ==============================================================================
# Esse Programa foi criado em dedicação ao meu filho Victor Katchor Cruz
# Diego Fernando Cruz – Motor VideoBit Brazil Beta (2025)
# Paypal diegocodigobits@gmail.com Pix Brazil santander27@gmail.com
# ------------------------------------------------------------------------------
# This program was created in dedication to my son Victor Katchor Cruz
# Diego Fernando Cruz – Motor VideoBit Brazil Beta (2025)
# Paypal diegocodigobits@gmail.com Pix Brazil santander27@gmail.com
# ==============================================================================

import cv2
import numpy as np
from tqdm import tqdm
import os

# CONFIGURAÇÃO
ARQUIVO_VIDEO = "video_teste.mp4"         # Caminho do vídeo de entrada
FRAMES_POR_BLOCO = 5                      # Quantos frames analisar por bloco
TAMANHO_MINIATURA = (32, 32)              # Reduz frame para acelerar

def extrair_bits_frame(frame, frame_anterior=None, index=0):
    small = cv2.resize(frame, TAMANHO_MINIATURA)
    mean_color = np.mean(small, axis=(0, 1))
    bit_b = int(mean_color[0] > 100)
    bit_g = int(mean_color[1] > 100)
    bit_r = int(mean_color[2] > 100)
    bit_mov = 0
    if frame_anterior is not None:
        diff = np.mean(np.abs(small.astype(float) - frame_anterior.astype(float)))
        bit_mov = int(diff > 10)

    print(f"[Bloco {index}] Média de cor: B={mean_color[0]:.1f}, G={mean_color[1]:.1f}, R={mean_color[2]:.1f} | Movimento: {bit_mov}")
    return [bit_b, bit_g, bit_r, bit_mov], small

def processar_video(arquivo_video):
    cap = cv2.VideoCapture(arquivo_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bits_video = []
    frame_anterior = None
    bloco_frames = []
    bloco_index = 0

    pbar = tqdm(total=total_frames, desc="Analisando vídeo")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bloco_frames.append(frame)
        pbar.update(1)
        if len(bloco_frames) == FRAMES_POR_BLOCO:
            frame_central = bloco_frames[len(bloco_frames)//2]
            bits, preview = extrair_bits_frame(frame_central, frame_anterior=frame_anterior, index=bloco_index)
            bits_video.append(bits)
            frame_anterior = cv2.resize(frame_central, TAMANHO_MINIATURA)

            cv2.imshow("Preview do Bloco", preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            bloco_frames = []
            bloco_index += 1

    cap.release()
    cv2.destroyAllWindows()
    pbar.close()
    return bits_video

def salvar_bits(bits_video, arquivo_saida="assinatura_videobit.txt"):
    with open(arquivo_saida, "w") as f:
        for bits in bits_video:
            linha = ''.join(str(b) for b in bits)
            f.write(linha + "\n")

if __name__ == "__main__":
    print("Iniciando análise do vídeo...")
    if not os.path.exists(ARQUIVO_VIDEO):
        print(f"Erro: Arquivo '{ARQUIVO_VIDEO}' não encontrado.")
    else:
        bits = processar_video(ARQUIVO_VIDEO)
        salvar_bits(bits)
        print("Processamento finalizado! Assinatura binária salva em 'assinatura_videobit.txt'")
