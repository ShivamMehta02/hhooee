import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort




model = YOLO('yolov11_players.pt')


tracker = DeepSort(max_age=60, n_init=3, nms_max_overlap=1.0, embedder="mobilenet", half=True)




class_id_to_color = {
    0: (0, 255, 255),  
    1: (255, 0, 0),     
    2: (0, 255, 0),    
    3: (0, 0, 255),    
}
default_color = (255, 255, 255)  



input_path = 'data/input.mp4'
output_path = 'output_tracked.mp4'
cap = cv2.VideoCapture(input_path)



width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 25  



fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))




class_id_to_name = {}

while True:
    ret, frame = cap.read()
    print("First frame read:", ret)
    if not ret:
        break

    results = model(frame)
    print("YOLO results:", results)
    detections = [] 
    for result in results:
        boxes = result.boxes
        print("Number of boxes detected:", len(boxes))
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            label = model.names[cls_id] if hasattr(model, 'names') else str(cls_id)
            class_id_to_name[cls_id] = label
            bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(0, min(x2, width-1))
            y2 = max(0, min(y2, height-1))
            if conf < 0.5:
                continue
            box_width = x2 - x1
            box_height = y2 - y1
            if box_width * box_height < 0.0001 * width * height:
                continue
            detections.append((bbox, float(conf), int(cls_id)))


    tracks = tracker.update_tracks(detections, frame=frame)

  
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width-1))
        y2 = max(0, min(y2, height-1))
        class_id = None
        conf = None
    
        if hasattr(track, 'det_class'):
            class_id = track.det_class
    
        # If the class is ball (class_id == 0), shrink the box by 40% (adjust as needed)
        if class_id == 0:
            box_w = x2 - x1
            box_h = y2 - y1
            shrink_factor = 0.4  # 40% smaller
            x1 = int(x1 + box_w * shrink_factor / 2)
            y1 = int(y1 + box_h * shrink_factor / 2)
            x2 = int(x2 - box_w * shrink_factor / 2)
            y2 = int(y2 - box_h * shrink_factor / 2)

        label = f'ID {track_id}'
        if class_id is not None and class_id in class_id_to_name:
            label = f'{class_id_to_name[class_id]} ID {track_id}'
        color = class_id_to_color.get(class_id if class_id is not None else -1, default_color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    out.write(frame)

    cv2.imshow('YOLO+DeepSORT Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print('Detected class IDs and their labels:')
for cls_id, name in class_id_to_name.items():
    print(f'Class ID: {cls_id} -> Label: {name}')
