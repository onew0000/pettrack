from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from collections import deque
import time
import math

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

class ObjectTracker:
    def __init__(self):
        self.model = YOLO('yolov10n.pt')
        self.class_names = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 트래킹 변수 초기화
        self.tracker = None
        self.selected_class = None
        self.path_coordinates = []
        self.show_path = True
        self.tracker_type = "CSRT"

        self.start_time = None
        self.total_distance = 0
        self.last_coordinate = None
        self.calories_per_meter = 0.8
        self.last_update_time = time.time()
        self.current_speed = 0
        
        self.update_delay = 0.3 
        self.last_metrics_update = time.time()
        
        # 10m x 10m 화각 설정
        self.frame_width = 640
        self.frame_height = 480
        self.real_width = 10.0
        self.real_height = 10.0
        
        self.temp_distance = 0
        self.temp_speed = 0
        
        self.init_tracker()

    def init_tracker(self):
        if self.tracker_type == "CSRT":
            self.tracker = cv2.TrackerCSRT_create()
        elif self.tracker_type == "KCF":
            self.tracker = cv2.TrackerKCF_create()
        elif self.tracker_type == "MOSSE":
            self.tracker = cv2.TrackerMOSSE_create()

    def pixel_to_meter(self, pixel_x, pixel_y):
        meter_x = (pixel_x / self.frame_width) * self.real_width
        meter_y = (pixel_y / self.frame_height) * self.real_height
        return meter_x, meter_y

    def calculate_distance(self, current_coordinate, last_coordinate):
        current_x, current_y = self.pixel_to_meter(current_coordinate[0], current_coordinate[1])
        last_x, last_y = self.pixel_to_meter(last_coordinate[0], last_coordinate[1])
        
        real_distance = math.sqrt(
            (current_x - last_x)**2 + 
            (current_y - last_y)**2
        )
        return real_distance

    def calculate_metrics(self, current_coordinate):
        if self.last_coordinate is None:
            self.last_coordinate = current_coordinate
            return
        
        real_distance = self.calculate_distance(current_coordinate, self.last_coordinate)
        
        self.temp_distance += real_distance
        current_time = time.time()
        time_diff = current_time - self.last_update_time
        
        if time_diff > 0:
            self.temp_speed = real_distance / time_diff  # m/s
        
        if current_time - self.last_metrics_update >= 0.3:
            self.total_distance += self.temp_distance
            
            activity_time = int(time.time() - self.start_time) if self.start_time else 0
            
            calories_burned = self.total_distance * self.calories_per_meter
            
            try:
                metrics = {
                    'distance': round(self.total_distance, 2),
                    'time': activity_time,
                    'speed': round(self.temp_speed, 2),  
                    'calories': round(calories_burned, 2)
                }
                socketio.emit('metrics_update', metrics)
                
                self.temp_distance = 0
                self.last_metrics_update = current_time
                
            except Exception as e:
                print(f"메트릭 전송 오류: {e}")
            
        self.last_coordinate = current_coordinate
        self.last_update_time = current_time

    def start_tracking(self, class_name):
        self.selected_class = class_name
        self.path_coordinates = []
        self.start_time = time.time()
        self.total_distance = 0
        self.last_coordinate = None
        self.temp_distance = 0
        self.temp_speed = 0
        self.last_metrics_update = time.time()
        self.init_tracker()

    def stop_tracking(self):
        self.tracker = None
        self.selected_class = None
        self.path_coordinates = []
        self.start_time = None
        self.total_distance = 0
        self.last_coordinate = None
        self.temp_distance = 0
        self.temp_speed = 0

    def track_object(self, frame):
        self.frame_height, self.frame_width = frame.shape[:2]
        
        if self.selected_class is None:
            return frame

        try:
            results = self.model(frame)[0]
            
            detected = False
            for r in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                if self.class_names[int(class_id)] == self.selected_class:
                    self.init_tracker()
                    bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    try:
                        self.tracker.init(frame, bbox)
                        detected = True
                        center_x = int(x1 + (x2 - x1) / 2)
                        center_y = int(y1 + (y2 - y1) / 2)
                        current_coordinate = (center_x, center_y)
                        self.path_coordinates.append(current_coordinate)
                        self.calculate_metrics(current_coordinate)
                        break
                    except Exception as e:
                        print(f"트래커 초기화 오류: {e}")
                        continue

            if self.tracker and detected:
                try:
                    success, box = self.tracker.update(frame)
                    if success:
                        x, y, w, h = map(int, box)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        center_x = x + w // 2
                        center_y = y + h // 2
                        current_coordinate = (center_x, center_y)
                        self.path_coordinates.append(current_coordinate)
                        self.calculate_metrics(current_coordinate)

                        if len(self.path_coordinates) > 2:
                            density = 0
                            for i in range(1, len(self.path_coordinates)):
                                distance = self.calculate_distance(self.path_coordinates[i], self.path_coordinates[i-1])
                                if distance < 0.5:  
                                    density += 1
                            if density > 2:  
                                color = (0, 0, 255)
                            else:
                                color = (0, 255, 0)
                            for i in range(1, len(self.path_coordinates)):
                                cv2.line(frame, 
                                       self.path_coordinates[i-1], 
                                       self.path_coordinates[i], 
                                       color, 
                                       2)
                        else:
                            for i in range(1, len(self.path_coordinates)):
                                cv2.line(frame, 
                                       self.path_coordinates[i-1], 
                                       self.path_coordinates[i], 
                                       (0, 255, 0), 
                                       2)
                                       
                        meter_x, meter_y = self.pixel_to_meter(center_x, center_y)
                        cv2.putText(frame, 
                                  f'Position: {meter_x:.1f}m, {meter_y:.1f}m', 
                                  (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, 
                                  (0, 255, 0), 
                                  2)
                except Exception as e:
                    print(f"트래킹 오류: {e}")

        except Exception as e:
            print(f"객체 감지 오류: {e}")

        return frame

object_tracker = ObjectTracker()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video-feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start-tracking', methods=['POST'])
def start_tracking():
    try:
        data = request.get_json()
        object_tracker.start_tracking(data.get('class_name', ''))
        return jsonify({'message': 'Tracking started'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop-tracking', methods=['POST'])
def stop_tracking():
    try:
        object_tracker.stop_tracking()
        return jsonify({'message': 'Tracking stopped'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        try:
            frame = object_tracker.track_object(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            
            if not ret:
                continue
                
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"프레임 처리 오류: {e}")
            continue

    cap.release()

if __name__ == '__main__':
    socketio.run(app, debug=True, port=8100)
