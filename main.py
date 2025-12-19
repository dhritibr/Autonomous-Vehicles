import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from detector import AutonomousDecisionSystem

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

system = AutonomousDecisionSystem(model_path="best.pt")

processing_status = {
    "is_processing": False,
    "progress": 0,
    "total_frames": 0,
    "current_frame": 0
}

@app.post("/process-media/")
async def process_media_endpoint(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv']
    
    input_filename = f"{file_id}{ext}"
    input_path = os.path.join(UPLOAD_DIR, input_filename)
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if is_video:
            output_filename = f"processed_{file_id}.mp4"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            processing_status["is_processing"] = True
            processing_status["progress"] = 0
            processing_status["current_frame"] = 0
            
            final_decision = system.process_video(input_path, output_path, processing_status)
            
            processing_status["is_processing"] = False
            processing_status["progress"] = 100
            
            media_type = "video"
        else:
            output_filename = f"processed_{file_id}.jpg"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            final_decision = system.process_image(input_path, output_path)
            media_type = "image"
        
        return {
            "status": "success",
            "media_url": f"http://localhost:8000/outputs/{output_filename}",
            "final_decision": final_decision,
            "media_type": media_type
        }
    except Exception as e:
        processing_status["is_processing"] = False
        return {"status": "error", "message": str(e)}

@app.get("/processing-progress")
def get_processing_progress():
    return processing_status

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(system.generate_frames(), 
                             media_type='multipart/x-mixed-replace; boundary=frame')

@app.post("/stop-feed-signal")
def stop_feed_signal():
    system.stop_streaming()
    return {"status": "stopped"}

@app.get("/current-status")
def get_current_status():
    return {"action": system.latest_action}

@app.post("/start-webcam/")
async def start_webcam_endpoint():
    try:
        result = system.process_webcam()
        return {"status": "success", "message": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Server Ready.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
