import cv2
import numpy as np

def nothing(x):
    pass

# Cargar imagen
img = cv2.imread("colonias.jpg")

# Escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Ventana y slider
cv2.namedWindow("Control")
cv2.createTrackbar("Threshold", "Control", 150, 255, nothing)

# Parámetros del detector de blobs
params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 20
params.maxArea = 50000

params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False

# Detectar blobs claros (colonias blancas)
params.filterByColor = True
params.blobColor = 255

detector = cv2.SimpleBlobDetector_create(params)

while True:
    thresh_val = cv2.getTrackbarPos("Threshold", "Control")

    # Threshold
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    # Limpiar ruido
    kernel = np.ones((1,1), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Detectar blobs
    keypoints = detector.detect(clean)

    # Dibujar blobs
    output = cv2.drawKeypoints(
        img,
        keypoints,
        None,
        (0, 0, 255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Mostrar conteo en pantalla
    cv2.putText(output, f"Colonias: {len(keypoints)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # Mostrar ventanas

    cv2.imshow("Threshold", clean)
    cv2.imshow("Blobs", output)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
