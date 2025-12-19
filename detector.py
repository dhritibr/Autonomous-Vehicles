import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time

class AutonomousDecisionSystem:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.CRITICAL_CLASSES = [0, 1, 2, 3, 5, 7]
        self.VULNERABLE_CLASSES = [0, 1] 
        self.MIN_CONFIDENCE = 0.5
        self.RISK_THRESHOLD_STOP = 3.5
        self.RISK_THRESHOLD_SLOW = 1.2
        self.decision_buffer = deque(maxlen=10)
        
        self.streaming_running = False
        self.latest_action = "--"
        self.webcam_capture = None

    def analyze_frame(self, frame, draw_hud=True):
        """
        Processes frame. 
        draw_hud=True  -> Burns 'ACTION: STOP' onto the image (For saved files/Popups)
        draw_hud=False -> Returns clean video (For Browser Feed)
        """
        results = self.model(frame, conf=self.MIN_CONFIDENCE, classes=self.CRITICAL_CLASSES, verbose=False)
        plotted_frame = results[0].plot()  # Default YOLO colors
        
        height, width, _ = frame.shape
        max_h_norm = 0.0
        vulnerable_count = 0
        
        # Risk Logic
        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls in self.VULNERABLE_CLASSES:
                vulnerable_count += 1
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            h_norm = (y2 - y1) / height
            max_h_norm = max(max_h_norm, h_norm)

        risk_score = (max_h_norm * 5.0) + (vulnerable_count * 2.0)
        
        if risk_score > self.RISK_THRESHOLD_STOP: action = "STOP"
        elif risk_score > self.RISK_THRESHOLD_SLOW: action = "SLOW DOWN"
        else: action = "GO"
        
        self.latest_action = action
        
        # Draw HUD (Only if requested)
        if draw_hud:
            cv2.rectangle(plotted_frame, (0, 0), (width, 80), (0, 0, 0), -1)
            text_color = (0, 255, 0)
            if action == "STOP": text_color = (0, 0, 255)
            elif action == "SLOW DOWN": text_color = (0, 255, 255)
            
            cv2.putText(plotted_frame, f"ACTION: {action}", (20, 55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)
        
        return plotted_frame, action

    def process_image(self, input_path, output_path):
        frame = cv2.imread(input_path)
        processed_frame, action = self.analyze_frame(frame, draw_hud=True) 
        cv2.imwrite(output_path, processed_frame)
        return action

    def process_video(self, input_path, output_path, progress_status=None):
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if progress_status:
            progress_status["total_frames"] = total_frames
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        final_consensus_action = "GO"
        
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            processed_frame, action = self.analyze_frame(frame, draw_hud=True)
            self.decision_buffer.append(action)
            current_action = max(set(self.decision_buffer), key=self.decision_buffer.count)
            final_consensus_action = current_action
            out.write(processed_frame)
            
            frame_count += 1
            if progress_status:
                progress_status["current_frame"] = frame_count
                progress_status["progress"] = int((frame_count / total_frames) * 100)
        
        cap.release()
        out.release()
        return final_consensus_action

    def generate_frames(self):
        self.streaming_running = True
        self.webcam_capture = cv2.VideoCapture(0)
        self.webcam_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.webcam_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.webcam_capture.isOpened(): 
            self.streaming_running = False
            self.webcam_capture = None
            return

        try:
            while self.streaming_running:
                success, frame = self.webcam_capture.read()
                if not success: break
                
                processed_frame, action = self.analyze_frame(frame, draw_hud=False)
                
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            if self.webcam_capture is not None:
                self.webcam_capture.release()
                self.webcam_capture = None
            self.streaming_running = False
            self.latest_action = "--"

    def stop_streaming(self):
        self.streaming_running = False
        if self.webcam_capture is not None:
            self.webcam_capture.release()
            self.webcam_capture = None
        self.latest_action = "--"

    def process_webcam(self):
        # Popup mode
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): return "Error: No Webcam"
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                processed_frame, action = self.analyze_frame(frame, draw_hud=True)
                cv2.imshow("Real-Time Autonomous System (Press Q to Exit)", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        return "Webcam Session Ended"
