import time
import cv2
import numpy as np
import pygetwindow as gw
import winsound
from mousekey import MouseKey
import pyautogui

class AgitarButtonDetector:
    def __init__(self, button_template_path, seta_template_paths, peixe_template_path, threshold=0.8):
        # Carregar as imagens dos templates (botão, setas e peixe) em escala de cinza
        self.button_template = cv2.imread(button_template_path, cv2.IMREAD_GRAYSCALE)
        self.seta_templates = [cv2.imread(seta_path, cv2.IMREAD_GRAYSCALE) for seta_path in seta_template_paths]
        self.peixe_template = cv2.imread(peixe_template_path, cv2.IMREAD_GRAYSCALE)
        self.threshold = threshold  # Limite de confiança para detecção
        self.seta_location = None  # Variável para armazenar a localização da "seta"

    def get_roblox_window(self):
        # Listar todos os títulos de janelas para verificar o título do Roblox
        for win in gw.getWindowsWithTitle("Roblox"):
            if win.isActive or win.isMaximized or win.isMinimized:
                return win
        print("Roblox window not found.")
        return None

    def take_screenshot(self):
        # Buscar a janela do Roblox
        window = self.get_roblox_window()

        if window is None:
            return None

        # Fazer captura de tela da janela ativa do Roblox
        screenshot = pyautogui.screenshot(region=(window.left, window.top, window.width, window.height))
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)  # Converter diretamente para escala de cinza
        return screenshot, window

    def find_template(self, image, template):
        # Realizar correspondência de template na imagem em escala de cinza
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

        # Obter a melhor posição de correspondência e confiança
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # Verificar se a confiança da correspondência está acima do limite
        if max_val >= self.threshold:
            # Calcular o centro da região correspondente
            h, w = template.shape
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            return (center_x, center_y), max_val
        return None, max_val

    def detect_and_click(self):
        while True:
            # Tirar uma captura de tela da janela do Roblox
            screenshot_data = self.take_screenshot()

            if screenshot_data is None:
                return

            screenshot, window = screenshot_data

            # Verificar a presença de "seta" ou "seta2"
            for seta_template in self.seta_templates:
                _, foto_mask = cv2.threshold(screenshot, 127, 255, cv2.THRESH_BINARY)
                self.seta_location, max_val = self.find_template(foto_mask, seta_template)
                # cv2.destroyAllWindows()
                if self.seta_location:
                    winsound.Beep(1000, 500)
                    print("Seta encontrada!")
                    break  # Sair do loop de verificação das setas

            if self.seta_location:
                break  # Sair do loop principal ao encontrar qualquer uma das setas

            # Procurar o botão "Agitar" se nenhuma seta foi encontrada
            button_center = self.find_template(screenshot, self.button_template)[0]

            if button_center:
                # Clicar no botão detectado na janela do Roblox
                self.click_button(button_center, window)
            else:
                print("Nenhum botão 'Agitar' detectado.")

            # Adicionar um delay de 0,5 segundo entre cada detecção do botão
            time.sleep(0.5)

        # Após encontrar a "seta", continuar com o processamento do "peixe"
        self.find_and_process_peixe(screenshot, window)

    def find_and_process_peixe(self, screenshot, window):
        # Realizar correspondência de template para "peixe.png" na captura de tela
        peixe_location, max_val_peixe = self.find_template(screenshot, self.peixe_template)

        if peixe_location:
            # Marcar o peixe e seta com uma linha e desenhar um círculo vermelho no meio
            print("Peixe encontrado!")
            # Desenhar no screenshot para testes visuais
            h, w = self.peixe_template.shape
            top_left = (peixe_location[0] - w // 2, peixe_location[1] - h // 2)
            bottom_right = (peixe_location[0] + w // 2, peixe_location[1] + h // 2)
            cv2.rectangle(screenshot, top_left, bottom_right, (0, 255, 0), 2)

            # Cálculo da direção e desenho da linha e círculo
            self.draw_line_and_circle(screenshot, self.seta_location, peixe_location)
        else:
            print("Peixe não encontrado!")

        # Exibir resultado final para visualização
        cv2.imshow("Resultado da Correspondência", screenshot)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_line_and_circle(self, image, center_seta, center_peixe):
        # Desenhar uma linha reta entre os centros
        cv2.line(image, center_seta, center_peixe, (255, 255, 0), 2)

        # Calcular o ponto médio entre os dois centros
        mid_point = ((center_seta[0] + center_peixe[0]) // 2, (center_seta[1] + center_peixe[1]) // 2)

        # Desenhar um círculo vermelho no ponto médio
        cv2.circle(image, mid_point, 5, (0, 0, 255), -1)

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

    def click_button(self, coordinates, window):
        if coordinates:
            x, y = coordinates
            screen_x = window.left + x
            screen_y = window.top + y

            mkey = MouseKey()
            mkey.left_click_xy_natural(
                screen_x, screen_y, delay=.05, min_variation=-3, max_variation=3, use_every=4,
                sleeptime=(0, 0), print_coords=True, percent=90
            )
            print(f"Clicou em: ({screen_x}, {screen_y})")

# Uso
detector = AgitarButtonDetector(
    button_template_path='bola.png',
    seta_template_paths=['seta.png', 'seta2.png'],  # Lista com os caminhos das duas setas
    peixe_template_path='peixe.png',
    threshold=0.8
)
detector.detect_and_click()
