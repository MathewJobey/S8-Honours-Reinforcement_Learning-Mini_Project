import io
import time
import json
from PIL import Image
from flask import Flask, render_template, request, jsonify, Response

import gymnasium as gym
from stable_baselines3 import SAC
import rocket_env # This registers your custom environments!

app = Flask(__name__)
import logging
# Tell the Flask terminal logger to strictly print ERRORS only!
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
# ==========================================
# 1. LOAD THE BRAIN AND ENVIRONMENT
# ==========================================
# We use "rgb_array" so Pygame draws the game invisibly in the background
env = gym.make("Phase3App-v0", render_mode="rgb_array")

# Load your absolute best AI brain! (Adjust the Run folder number if needed)
model = SAC.load("models/Run 6/sac_phase3_final_v0", env=env)

# A temporary mailbox to hold the dashboard numbers before the video starts
pending_launch_data = None
current_telemetry = {"active": False}
# ==========================================
# 2. THE WEB ROUTES
# ==========================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/launch', methods=['POST'])
def launch():
    global pending_launch_data, current_telemetry # <--- Added current_telemetry
    
    # Grab the JSON dictionary sent by the JavaScript and put it in the mailbox
    pending_launch_data = request.json
    
    # ---> THE FIX: Turn the pipeline ON immediately before the video even starts!
    current_telemetry["active"] = True 
    
    print(f"\n>>> INCOMING LAUNCH DETECTED: {pending_launch_data}")
    
    return jsonify({"status": "success"})

# ==========================================
# 3. THE VIDEO GENERATOR (MJPEG STREAM)
# ==========================================
def generate_video_frames():
    global pending_launch_data
    
    # 1. Grab the data from the mailbox and empty it
    data = pending_launch_data
    pending_launch_data = None
    
    # 2. Tell the environment to restart using the exact dashboard numbers
    obs, info = env.reset(options=data)
    done = False
    
    global current_telemetry
    current_telemetry["active"] = True
    
    # 3. The Test Flight Loop
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # ---> NEW: Grab the telemetry and update the global dictionary <---
        live_data = env.unwrapped.get_telemetry()
        live_data["active"] = True
        current_telemetry = live_data
        
        # Grab the raw RGB pixel data from the invisible Pygame window
        frame_pixels = env.render() 
        
        # Compress those pixels into a standard JPEG image
        img = Image.fromarray(frame_pixels)
        buffer = io.BytesIO()
        # New crystal-clear PNG!
        # The 'Perfect JPEG': Blazing fast, but keeps sharp edges!
        img.save(buffer, format="JPEG", quality=100, subsampling=0)
        image_bytes = buffer.getvalue()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')
        
        # Sleep for a tiny fraction of a second to lock the video at roughly 60 FPS
        time.sleep(1.0 / 60.0)

    current_telemetry["active"] = False
    
# ==========================================
# 4. THE TELEMETRY PIPELINE (SSE)
# ==========================================
def generate_telemetry_stream():
    global current_telemetry
    
    while True:
        # Step A: Package the data in the exact text format required by SSE
        json_data = json.dumps(current_telemetry)
        yield f"data: {json_data}\n\n"
        
        # Step B: If the rocket has landed or crashed, close the pipeline
        if current_telemetry.get("active", False) == False:
            break
            
        # Step C: Control the flow rate (10 times a second)
        time.sleep(0.1) 

@app.route('/telemetry_stream')
def telemetry_stream():
    # The special 'text/event-stream' tells the browser to keep the connection open!
    return Response(generate_telemetry_stream(), mimetype='text/event-stream')

@app.route('/video_feed')
def video_feed():
    # This special Response tells the browser: "Keep the connection open! 
    # I am going to send you a mixed bag of continuously replacing pictures."
    return Response(
        generate_video_frames(), 
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    # ---> NEW: Custom Startup Message with Clickable Link <---
    print("\n" + "="*55)
    print(" 🚀 DROPSHIP MISSION CONTROL ONLINE 🚀")
    print(" >>> Click here to open: http://127.0.0.1:5000 <<<")
    print("="*55 + "\n")
    
    # Start the server (Turned off the reloader so it doesn't print the message twice)
    app.run(debug=True, port=5000, use_reloader=False)