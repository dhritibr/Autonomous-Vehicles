import os
import shutil
import uuid
import requests
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from detector import AutonomousDecisionSystem

# --- DYNAMIC CONFIGURATION ---
# Ensure this is your actual GitHub Release direct download link
MODEL_URL = "https://github.com/dhritibr/Ai-in-Autonomous-Vehicles/releases/download/v1.0.0/best.pt"

# Detect if running on Vercel or local laptop
is_cloud = "VERCEL" in os.environ
base_dir = "/tmp" if is_cloud else "."
MODEL_PATH = os.path.join(base_dir, "best.pt")

def download_model_if_needed():
    """Downloads the YOLO model if it doesn't exist in the current environment."""
    if not os.path.exists(MODEL_PATH):
        print(f"📥 Downloading model from GitHub to {MODEL_PATH}...")
        try:
            with requests.get(MODEL_URL, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(MODEL_PATH, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            print("✅ Model download complete.")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This logic runs on server startup
    download_model_if_needed()
    yield

# Initialize FastAPI with the lifespan event
app = FastAPI(lifespan=lifespan)

# Allow all origins for maximum portability across different laptop IPs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup temporary storage paths
UPLOAD_DIR = os.path.join(base_dir, "uploads")
OUTPUT_DIR = os.path.join(base_dir, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Serve the processed output files
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# Initialize the AI detector system
# Note: system needs to be initialized AFTER the model is downloaded in lifespan
# We initialize it here; YOLO will load the file from MODEL_PATH when first used
system = AutonomousDecisionSystem(model_path=MODEL_PATH)

processing_status = {
    "is_processing": False, 
    "progress": 0, 
    "total_frames": 0, 
    "current_frame": 0
}

# --- FRONTEND ROUTES ---

@app.get("/")
async def serve_landing():
    return FileResponse("landing.html")

@app.get("/index.html")
async def serve_app():
    return FileResponse("index.html")

# --- API ENDPOINTS ---

@app.post("/process-media/")
async def process_media_endpoint(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv']
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if is_video:
            output_filename = f"processed_{file_id}.mp4"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            processing_status.update({"is_processing": True, "progress": 0, "current_frame": 0})
            
            # Run YOLO detection logic from detector.py
            final_decision = system.process_video(input_path, output_path, processing_status)
            
            processing_status.update({"is_processing": False, "progress": 100})
            media_type = "video"
        else:
            output_filename = f"processed_{file_id}.jpg"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            final_decision = system.process_image(input_path, output_path)
            media_type = "image"
        
        return {
            "status": "success",
            "media_url": f"/outputs/{output_filename}",
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

if __name__ == "__main__":
    import uvicorn
    print("🚀 Server Ready. Go to http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
