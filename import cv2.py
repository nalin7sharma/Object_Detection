import cv2
import numpy as np


yolo_ka_blueprint = "d:/Downloads/YOLO-Real-Time-Object-Detection-master/cfg/yolov3.cfg"
yolo_ka_weight = "d:/Downloads/YOLO-Real-Time-Object-Detection-master/weight/yolov3.weights"
chehre_ke_naam = "d:/Downloads/YOLO-Real-Time-Object-Detection-master/weight/coco.names"

dimaag = cv2.dnn.readNetFromDarknet(yolo_ka_blueprint, yolo_ka_weight)
dimaag.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
dimaag.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


with open(chehre_ke_naam, 'r') as f:
    naam_ke_labels = [line.strip() for line in f.readlines()]


secret_layers = dimaag.getLayerNames()
output_layers = [secret_layers[i - 1] for i in dimaag.getUnconnectedOutLayers()]


camra = cv2.VideoCapture(0)


while camra.isOpened():
    ret, photo = camra.read()
    if not ret:
        break

    
    oonchai, chorai, _ = photo.shape

    
    blob = cv2.dnn.blobFromImage(photo, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    dimaag.setInput(blob)
    detections = dimaag.forward(output_layers)

    
    items = []
    bharosa_levels = []
    dibbe = []

    
    for detection in detections:
        for item in detection:
            scores = item[5:]
            item_id = np.argmax(scores)
            bharosa = scores[item_id]
            if bharosa > 0.5:  
                center_x = int(item[0] * chorai)
                center_y = int(item[1] * oonchai)
                width = int(item[2] * chorai)
                height = int(item[3] * oonchai)

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                dibbe.append([x, y, width, height])
                bharosa_levels.append(float(bharosa))
                items.append(item_id)

    
    bachhe_hue = cv2.dnn.NMSBoxes(dibbe, bharosa_levels, 0.5, 0.4)

    
    if bachhe_hue is not None:
        for i in bachhe_hue.flatten():
            x, y, width, height = dibbe[i]
            label = str(naam_ke_labels[items[i]])
            bharosa = bharosa_levels[i]
            rang = (0, 255, 0)  
            cv2.rectangle(photo, (x, y), (x + width, y + height), rang, 2)
            cv2.putText(photo, f'{label} {bharosa:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rang, 2)

    
    cv2.imshow('Live Object Detection', photo)

    
    if cv2.waitKey(1) & 0xFF == ord('n'):
        break

camra.release()
cv2.destroyAllWindows()
