import argparse
import cv2
import numpy as np
import torch
from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, scale_boxes, check_img_size)
from utils.torch_utils import select_device
import mediapipe as mp

def run(weights='yolov5s.pt', source='0', imgsz=(640, 640), conf_thres=0.25, iou_thres=0.45, device=''):
    # Initialize YOLOv5 model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Initialize MediaPipe face mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

    # Load image/video source
    cap = cv2.VideoCapture(int(source) if source.isnumeric() else source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Prepare image for YOLOv5
        img = cv2.resize(frame, imgsz)
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = torch.from_numpy(np.ascontiguousarray(img)).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # YOLOv5 inference
        pred = model(img, augment=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        # Process detections
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    if names[int(cls)] == "person":
                        x1, y1, x2, y2 = map(int, xyxy)
                        face_roi = frame[y1:y2, x1:x2]

                        # Apply face mesh detection
                        face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                        face_mesh_results = face_mesh.process(face_roi_rgb)

                        # Draw face mesh if detected
                        if face_mesh_results.multi_face_landmarks:
                            for face_landmarks in face_mesh_results.multi_face_landmarks:
                                for landmark in face_landmarks.landmark:
                                    lm_x = int(landmark.x * face_roi.shape[1])
                                    lm_y = int(landmark.y * face_roi.shape[0])
                                    cv2.circle(face_roi, (lm_x, lm_y), 1, (0, 255, 0), -1)

                        # Replace the face ROI in the original frame
                        frame[y1:y2, x1:x2] = face_roi

                        # Draw YOLOv5 detection box
                        label = f'{names[int(cls)]} {conf:.2f}'
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Show the output frame
        cv2.imshow('Face Detection and Mesh', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model path')
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640, 640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IOU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()

    run(weights=opt.weights, source=opt.source, imgsz=opt.imgsz, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, device=opt.device)