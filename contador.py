import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imagen en escala de grises
img = cv2.imread("colonias.jpg", cv2.IMREAD_GRAYSCALE)

# Binarizar (blanco/negro)
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Encontrar componentes conectados
num_labels, labels = cv2.connectedComponents(thresh)

# Restamos 1 porque el fondo cuenta como etiqueta
cantidad_puntos = num_labels - 1

def nothing(x):
    pass

# Crear ventana
cv2.namedWindow("Control")

# Crear slider (0–255)
cv2.createTrackbar("Threshold", "Control", 193, 255, nothing)

while True:
    # Leer valor del slider
    thresh_val = cv2.getTrackbarPos("Threshold", "Control")

    # Aplicar threshold dinámico
    _, thresh = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)

    # Mostrar resultado
    cv2.imshow("Imagen original", img)
    cv2.imshow("Threshold", thresh)

    # Salir con ESC
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()

print("Cantidad de puntos:", cantidad_puntos)
