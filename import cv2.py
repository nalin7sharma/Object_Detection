import cv2
import numpy as np

# Paths to model configuration, weights, and object names
yolo_ka_blueprint = "d:/Downloads/YOLO-Real-Time-Object-Detection-master/cfg/yolov3.cfg"
weights_path = "d:/Downloads/YOLO-Real-Time-Object-Detection-master/weight/yolov3.weights"
coco_names_path = "d:/Downloads/YOLO-Real-Time-Object-Detection-master/weight/coco.names"

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(yolo_ka_blueprint, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# coco_names_path ka data load kr rhe hai taaki read kr sake objects ko accordingly......
with open(coco_names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()


output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# webcam start karne k liye instead of webcam if kisi video me detect krna hai toh 0 ki jgah uss video ka path daal denge.....
camera = cv2.VideoCapture(0)

while camera.isOpened():
    ret, photo = camera.read()
    if not ret:
        break

    height, width, channels = photo.shape

    # photo ko yolo ko feed krne ke liye blob bna rhe hai......
    blob = cv2.dnn.blobFromImage(photo, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    # alag alag lists bna rhe hai taaki detected items ko rakh sake......
    items = []
    confidences = []
    boxes = []

    for detection in detections:
        for item in detection:
            scores = item[5:]
            item_id = np.argmax(scores)
            confidence = scores[item_id]
            if confidence > 0.5:  
                center_x = int(item[0] * width)
                center_y = int(item[1] * height)
                w = int(item[2] * width)
                h = int(item[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                items.append(item_id)

    # Non-max suppression laga rhe hai taaki overlapping boxes ko hataye
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # agar koi item detect hota hai toh uspe rectangle aur label draw krenge
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[items[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  
            cv2.rectangle(photo, (x, y), (x + w, y + h), color, 2)
            cv2.putText(photo, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Live Object Detection', photo)

    # n press krne se loop break ho jayega
    if cv2.waitKey(1) & 0xFF == ord('n'):
        break

camera.release()
cv2.destroyAllWindows()
