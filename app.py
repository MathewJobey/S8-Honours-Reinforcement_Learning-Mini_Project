from flask import Flask, render_template, Response, request, jsonify
import gymnasium as gym
import rocket_env
from stable_baselines3 import PPO
from PIL import Image
import io
import numpy as np
import time

# 1. Start the Flask Server
app = Flask(__name__)

# 2. Load the AI Brain (Make sure this matches your newest 3-thruster brain!)
model = PPO.load("ppo_phase2_descent_v2")

# 3. Memory for the user's slider choices
current_settings = {
    "altitude": 1000.0,
    "speed": -50.0,
    "x_pos": 0.0,
    "should_reset": False 
}

# 4. The Camera Loop
def generate_video():
    """Runs the physics engine in the background and takes pictures."""
    # Ask the environment to give us raw pixels instead of opening a window
    env = gym.make("Phase2Descent-v0", render_mode="rgb_array")
    obs, info = env.reset(options=current_settings)
    
    while True:
        # Check if the user clicked the "INITIATE DROP" button
        if current_settings["should_reset"]:
            obs, info = env.reset(options=current_settings)
            current_settings["should_reset"] = False
            
        # Ask the AI for the best move
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Take a snapshot of the simulation
        frame_array = env.render()
        
        # Convert raw pixels into a JPEG image
        img = Image.fromarray(frame_array.astype('uint8'))
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=80)
        img_bytes = img_buffer.getvalue()
        
        # Send the picture to the web browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')
        
        # If the rocket lands or crashes, pause and then restart
        if terminated or truncated:
            time.sleep(1.5) # Pause for 1.5 seconds so the user can see the result
            obs, info = env.reset(options=current_settings)

# 5. Web Routes
@app.route('/')
def home():
    # Door 1: Give the user the dark-themed webpage
    return render_template('index.html')

@app.route('/launch', methods=['POST'])
def launch():
    # Door 2: Catch the new numbers sent by the JavaScript
    data = request.json
    current_settings["altitude"] = float(data["altitude"])
    current_settings["speed"] = float(data["speed"])
    current_settings["x_pos"] = float(data["x_pos"])
    
    # Flip the switch so the camera loop knows to drop a new rocket
    current_settings["should_reset"] = True 
    return jsonify({"status": "success"})

@app.route('/video_feed')
def video_feed():
    # Door 3: Stream the flipbook of pictures forever
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 6. Run the Server
if __name__ == '__main__':
    # Debug mode automatically restarts the server if you change the code later
    app.run(debug=True, port=5000)