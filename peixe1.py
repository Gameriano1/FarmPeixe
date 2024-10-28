import cv2
import numpy as np

# Carregar a imagem principal
foto = cv2.imread('image.png')

# Define lower and upper bounds for color range
lowcor = np.array([10, 10, 10])
highcor = np.array([100, 100, 200])

# Convert the main image to grayscale and create a binary mask
foto_cinza = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY)
_, foto_mask = cv2.threshold(foto_cinza, 127, 255, cv2.THRESH_BINARY)




# im2, contours, hierarchy = cv2.findContours(foto_cinza, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


# Função para realizar a correspondência de modelo
def find_template(foto_mask, template_path, threshold=0.8):
    # Carregar a imagem do template
    template = cv2.imread(template_path)
    h, w = template.shape[:2]

    # Converter o template para escala de cinza
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Realizar a correspondência de template
    result = cv2.matchTemplate(foto_mask, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # Verificar se a correspondência excede o limiar
    if max_val >= threshold:
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        return (top_left, bottom_right), max_val
    return None, max_val


# Primeiro, tentar encontrar `seta.png`
location, max_val = find_template(foto_mask, 'seta.png', threshold=0.6)

# Se não encontrado, tentar `seta2.png`
if location is None:
    location, max_val = find_template(foto_mask, 'seta2.png', threshold=0.6)

# Variáveis para armazenar os centros dos retângulos
center_seta = None
center_peixe = None

# Se a "seta" for encontrada, procurar "peixe.png"
if location:
    top_left, bottom_right = location
    cv2.rectangle(foto, top_left, bottom_right, (255, 0, 0), 2)

    # Calcular o centro do retângulo da "seta"
    center_seta = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)

    # Carregar e converter o template do peixe para escala de cinza
    peixe_template = cv2.imread('peixe.png', cv2.IMREAD_GRAYSCALE)
    h_peixe, w_peixe = peixe_template.shape[:2]

    # Realizar correspondência de template para "peixe.png" na imagem completa
    result_peixe = cv2.matchTemplate(foto_cinza, peixe_template, cv2.TM_CCOEFF_NORMED)
    _, max_val_peixe, _, max_loc_peixe = cv2.minMaxLoc(result_peixe)

    # Verificar se a correspondência de "peixe.png" excede o limiar
    if max_val_peixe >= 0.8:
        peixe_top_left = max_loc_peixe
        peixe_bottom_right = (peixe_top_left[0] + w_peixe, peixe_top_left[1] + h_peixe)
        cv2.rectangle(foto, peixe_top_left, peixe_bottom_right, (0, 255, 0), 2)
        print("Peixe encontrado!")

        # Calcular o centro do retângulo do "peixe"
        center_peixe = ((peixe_top_left[0] + peixe_bottom_right[0]) // 2,
                        (peixe_top_left[1] + peixe_bottom_right[1]) // 2)
    else:
        print("Peixe não encontrado!")
else:
    print("Seta não encontrada!")

# Desenhar linha e círculo vermelho se ambos os centros forem encontrados
if center_seta and center_peixe:
    # Desenhar uma linha reta entre os centros
    cv2.line(foto, center_seta, center_peixe, (255, 255, 0), 2)

    # Calcular o ponto médio entre os dois centros
    mid_point = ((center_seta[0] + center_peixe[0]) // 2, (center_seta[1] + center_peixe[1]) // 2)

    # Desenhar um círculo vermelho no ponto médio
    cv2.circle(foto, mid_point, 5, (0, 0, 255), -1)

    # Calcular a direção da linha entre a seta e o peixe
    dx = center_peixe[0] - center_seta[0]
    dy = center_peixe[1] - center_seta[1]

    # Determinar a direção com base em dx e dy
    if abs(dx) > abs(dy):  # Movimento predominante é horizontal
        if dx > 0:
            direction = "direita"
        else:
            direction = "esquerda"
    else:  # Movimento predominante é vertical
        direction = None

    print(f"A direção da linha da seta até o peixe é para: {direction}")

# Exibir o resultado
cv2.imshow("Resultado da Correspondência", foto)
cv2.waitKey(0)
cv2.destroyAllWindows()
