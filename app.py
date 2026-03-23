from flask import Flask, render_template, request, jsonify

# 1. Start the Flask Server
app = Flask(__name__)

# --- TEMPORARILY DISABLED ---
# We commented out the AI and physics engine so the server doesn't 
# crash looking for a brain file that doesn't exist yet!
# import gymnasium as gym
# from stable_baselines3 import PPO
# model = PPO.load("ppo_phase2_descent_v2")

@app.route('/')
def home():
    # Door 1: Give the user the dark-themed webpage
    return render_template('index.html')

@app.route('/launch', methods=['POST'])
def launch():
    # Door 2: Catch the new numbers sent by the JavaScript
    data = request.json
    
    # Print the numbers to the terminal so we can prove the UI works!
    print(f"\n--- LAUNCH INITIATED ---")
    print(f"Altitude: {data['altitude']} m")
    print(f"Speed: {data['speed']} m/s")
    print(f"X-Offset: {data['x_pos']} m")
    print(f"Angle: {data['angle']} deg\n") # <--- NEW!
    
    return jsonify({"status": "success"})

@app.route('/video_feed')
def video_feed():
    # Door 3: Placeholder for the video feed.
    # We will put the PyGame camera loop back in here later!
    return "Video Player Offline"

# Run the Server
if __name__ == '__main__':
    app.run(debug=True, port=5000)