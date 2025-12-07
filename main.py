import cv2
import numpy as np
from pydantic import BaseModel

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(1)

WINDOW_WIDHT = 1366
WINDOW_HEIGHT = 768

scene = np.zeros((WINDOW_HEIGHT, WINDOW_WIDHT,3), dtype=np.uint8)

SCALE_FACTOR = 2
VELOCITY_FACTOR = 0.6
WINDOW_CENTER = np.array([scene.shape[1] // 2, scene.shape[0] // 2])

class Rectangle():
    def __init__(self, pt1, pt2, color, thickness):
        self.pt1 = pt1
        self.pt2 = pt2
        self.color = color
        self.thickness = thickness

    def get_center(self):
        x1, y1 = self.pt1
        x2, y2 = self.pt2
        return (x1 + x2) // 2, (y1 + y2) // 2
    
    def move_to(self, new_center:np.array) -> np.array:
        x, y = new_center.tolist()
        x1, y1 = self.pt1
        x2, y2 = self.pt2
        self.pt1 = (x - (x2 - x1) // 2, y - (y2 - y1) // 2)
        self.pt2 = (x + (x2 - x1) // 2, y + (y2 - y1) // 2)

        return np.array([x, y])
    
    import numpy as np
import cv2

class Rectangle():
    def __init__(self, pt1, pt2, color, thickness):
        # pt1: Esquina superior izquierda (x1, y1)
        self.pt1 = pt1
        # pt2: Esquina inferior derecha (x2, y2)
        self.pt2 = pt2
        self.color = color
        self.thickness = thickness

    def get_corners(self):
        """
        Calcula y devuelve la ubicación de las 4 esquinas del rectángulo.
        Asume que pt1 es la esquina superior izquierda (x1, y1)
        y pt2 es la esquina inferior derecha (x2, y2).
        """
        x1, y1 = self.pt1
        x2, y2 = self.pt2
        
        top_left = (x1, y1)
        
        top_right = (x2, y1)
        
        bottom_right = (x2, y2)
        
        bottom_left = (x1, y2)
        
        return [top_left, top_right, bottom_right, bottom_left]
    
    def get_center(self):
        x1, y1 = self.pt1
        x2, y2 = self.pt2
        # El cálculo del centro es correcto
        center = np.array([(x1 + x2) // 2, (y1 + y2) // 2])
        
        # Se eliminaron las asignaciones a self.pt3 y self.pt4
        # ya que no representan las esquinas ni se usan correctamente aquí.
        
        return center

    def draw(self, img):
        cv2.rectangle(img, self.pt1, self.pt2, self.color, self.thickness)

    def move_to(self, new_center:np.array) -> np.array:
        x, y = new_center.tolist()
        x1, y1 = self.pt1
        x2, y2 = self.pt2
        
        # Calcular el ancho y alto
        width = x2 - x1
        height = y2 - y1
        
        # Calcular los nuevos pt1 y pt2 basados en el nuevo centro
        self.pt1 = (x - width // 2, y - height // 2)
        self.pt2 = (x + width // 2, y + height // 2)

        return np.array([x, y])

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    
    faces = face_classifier.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return faces

rectangle_0 = Rectangle(pt1=[10,10], pt2=[200,200], color=[255,255,255], thickness=2)
rectangle_1 = Rectangle(pt1=[10 * SCALE_FACTOR,10*SCALE_FACTOR], pt2=[200*SCALE_FACTOR,200*SCALE_FACTOR], color=[255,255,255], thickness=2)

face_center_from_window_center = WINDOW_CENTER - rectangle_0.get_center()

rectangle_0.move_to(WINDOW_CENTER)
rectangle_1.move_to(WINDOW_CENTER)
i = 0

while True:
    result, video_frame = video_capture.read()
    if result is False:
        break

    video_frame = cv2.resize(video_frame, (WINDOW_WIDHT, WINDOW_HEIGHT), interpolation=cv2.INTER_LINEAR)

    faces = detect_bounding_box(video_frame)

    scene = np.zeros((WINDOW_HEIGHT, WINDOW_WIDHT, 3), dtype=np.uint8)
    
    try:
        face_center = np.array([faces[0][0] + faces[0][2] // 2, faces[0][1] + faces[0][3]])

        face_center_from_window_center = WINDOW_CENTER - face_center
        
        rectangle_0.move_to(WINDOW_CENTER + face_center_from_window_center)

        delta_mov = face_center_from_window_center*VELOCITY_FACTOR

        rectangle_1.move_to(WINDOW_CENTER + delta_mov.astype(int))
    
    except IndexError:
        pass

    rectangle_1_corners = rectangle_1.get_corners()
    rectangle_0_corners = rectangle_0.get_corners()

    cv2.line(scene, rectangle_0_corners[0], rectangle_1_corners[0],(255,255,255),2)
    cv2.line(scene, rectangle_0_corners[1], rectangle_1_corners[1],(255,255,255),2)
    cv2.line(scene, rectangle_0_corners[2], rectangle_1_corners[2],(255,255,255),2)
    cv2.line(scene, rectangle_0_corners[3], rectangle_1_corners[3],(255,255,255),2)
    
    rectangle_0.draw(scene)

    rectangle_1.draw(scene)

    scene = cv2.flip(scene, 0)
    # frame = cv2.hconcat([video_frame, scene])

    cv2.imshow('Video', scene)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

