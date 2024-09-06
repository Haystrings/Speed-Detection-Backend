from flask import Flask, Response, jsonify, send_from_directory
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import threading

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")

# Line points for reference (modify as needed)
line_pts = [(0, 360), (1280, 360)]
vehicle_speeds = {}

# Pixel to meter ratio and speed threshold
pixel_to_meter_ratio = 0.05
speed_threshold = 50

# Video output path
VIDEO_OUTPUT_PATH = 'output_segment.mp4'
recording = False
video_writer = None
cap = None

def gen_frames():
    global cap, recording, video_writer
    # Video stream source (e.g., from Raspberry Pi)
    video_source = 0  # Replace with your Pi's video stream URL
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            conf = scores[i]
            cls = int(class_ids[i])
            label = results[0].names[cls]

            if label in ["car", "truck", "bus"]:
                speed = (x2 - x1) * pixel_to_meter_ratio * cap.get(cv2.CAP_PROP_FPS) / 3.6
                vehicle_id = f"{cls}-{x1}-{y1}"

                if vehicle_id not in vehicle_speeds:
                    vehicle_speeds[vehicle_id] = {'cumulative_speed': 0, 'frame_count': 0}

                vehicle_speeds[vehicle_id]['cumulative_speed'] += speed
                vehicle_speeds[vehicle_id]['frame_count'] += 1
                avg_speed = vehicle_speeds[vehicle_id]['cumulative_speed'] / vehicle_speeds[vehicle_id]['frame_count']

                color = (0, 255, 0)  # Green
                overspeeding_text = ''
                if avg_speed > speed_threshold:
                    color = (0, 0, 255)  # Red for overspeeding
                    overspeeding_text = 'Overspeeding!'

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"Speed: {avg_speed:.2f} km/h", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if overspeeding_text:
                    cv2.putText(frame, overspeeding_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if recording and video_writer:
            video_writer.write(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording')
def start_recording():
    global recording, video_writer
    if not recording:
        # Fetch the actual frame rate of the video source
        fps = 15
        if fps == 0:  # Fallback in case the frame rate isn't detected
            fps = 30  # Set a default frame rate

        # Create VideoWriter object to save video segment
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
        recording = True
    return jsonify({"status": "Recording started"})


@app.route('/stop_recording')
def stop_recording():
    global recording, video_writer
    if recording:
        recording = False
        video_writer.release()
    return jsonify({"status": "Recording stopped"})

@app.route('/download_video')
def download_video():
    return send_from_directory(directory='.', path=VIDEO_OUTPUT_PATH, as_attachment=True)

@app.route('/speed_data')
def speed_data():
    # Example: Send average speed of all detected vehicles
    avg_speeds = {vehicle: data['cumulative_speed'] / data['frame_count']
                  for vehicle, data in vehicle_speeds.items()}
    return jsonify(avg_speeds)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
