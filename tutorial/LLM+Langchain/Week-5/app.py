import os
import base64
import uuid
from flask import Flask, request, render_template, jsonify, session
from werkzeug.utils import secure_filename
import requests

app = Flask(__name__)
app.secret_key = "super-secret-key"  # Needed for session

IMAGE_DIR = "image_psychic"
os.makedirs(IMAGE_DIR, exist_ok=True)


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_image_prompt(image_str):
    
    return {"type": "image",
            "template": {"url": f"data:image/jpeg;base64,{image_str}"}}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Step 1: Upload images + prompt, but don't process yet"""
    uploaded_files = request.files.getlist("images")
    user_prompt = request.form.get("prompt", "")

    # Generate a session ID to group this request
    session_id = str(uuid.uuid4())
    session["session_id"] = session_id

    image_files = []
    for file in uploaded_files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(IMAGE_DIR, f"{session_id}_{filename}")
        file.save(file_path)
        image_files.append(f"{session_id}_{filename}")

    # Save data in session (lightweight, not for production-scale)
    session["uploaded_files"] = image_files
    session["user_prompt"] = user_prompt

    return jsonify({"message": "Images uploaded successfully. Click Generate to continue."})


@app.route("/generate", methods=["POST"])
def generate():
    """Step 2: Build human_template and call backend"""
    image_files = session.get("uploaded_files", [])
    user_prompt = session.get("user_prompt", "")

    if not image_files or not user_prompt:
        return jsonify({"ai_response": "No uploaded images or prompt found. Please upload first."})

    # Build human_template
    human_template = []
    text_prompt_template = {"template": "{question}", "input_variables": ["question"]}
    human_template.append(text_prompt_template)

    for image_file in image_files:
        image_str = image_to_base64(os.path.join(IMAGE_DIR, image_file))
        human_template.append(build_image_prompt(image_str))
        os.remove(os.path.join(IMAGE_DIR, image_file))

    payload = {
        "human": human_template,
        "question": user_prompt
    }

    # Send to backend AI service
    try:
        resp = requests.post(
            "http://localhost:5000/app_image_psychic/invoke",
            json={"input": payload},
            timeout=180
        )
        ai_response = resp.json().get("output", {}).get("content", "No response")
    except Exception as e:
        ai_response = f"Error contacting backend: {e}"

    return jsonify({"ai_response": ai_response})


if __name__ == "__main__":
    app.run(port=8000, debug=True)