from cProfile import label
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, session
import os
from utils import extract_frames, predict_video, get_grad_cam, save_gradcam_overlay
import csv
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import json
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

UPLOAD_FOLDER = 'uploads'
CHART_FOLDER = 'static/charts'
GRADCAM_FOLDER = 'static/gradcam'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHART_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)


app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Dummy user database (for demo)
USERS = {
    'user': {'password': 'userpass', 'role': 'user'},
    'admin': {'password': 'adminpass', 'role': 'admin'}
}

model = load_model('forest_fire_model_mobilenetv2.h5')


# Home page: redirect to login if not logged in, else to dashboard
@app.route('/', methods=['GET'])
def index():
    if 'username' in session:
        if session.get('role') == 'admin':
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('user_dashboard'))
    return redirect(url_for('login'))

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = USERS.get(username)
        if user and user['password'] == password:
            session['username'] = username
            session['role'] = user['role']
            if user['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('user_dashboard'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')


# Logout
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# User dashboard (upload/predict)
@app.route('/user/dashboard', methods=['GET'])
def user_dashboard():
    if session.get('role') != 'user':
        return redirect(url_for('login'))
    return render_template('user_dashboard.html', username=session.get('username'))


# Admin dashboard
@app.route('/admin/dashboard', methods=['GET'])
def admin_dashboard():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
    # List uploaded files
    uploaded_files = os.listdir(UPLOAD_FOLDER)
    # User stats
    user_count = 0
    if 'USERS' in globals():
        user_count = len(USERS)
    # Video stats
    video_count = len(uploaded_files)
    # Prediction stats and all history
    prediction_count = 0
    all_history = []
    fire_alert = False
    flag_path = 'fire_alert.flag'

    if os.path.exists('user_history.csv'):
        with open('user_history.csv', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            all_history = list(reader)
            prediction_count = len(all_history)

    if os.path.exists(flag_path):
        try:
            with open(flag_path, 'r') as f:
                content = f.read().strip()
            if content == '1':
                fire_alert = True
                os.remove(flag_path)  # Clear after showing
        except Exception as e:
            print("Error reading fire alert flag:", e)

    return render_template('admin_dashboard.html', username=session.get('username'), uploaded_files=uploaded_files, user_count=user_count, video_count=video_count, prediction_count=prediction_count, all_history=all_history,fire_alert=fire_alert)

# Admin delete any user's prediction history
@app.route('/admin/history/delete', methods=['POST'])
def admin_delete_history():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
    username = request.form['username']
    video_filename = request.form['video_filename']
    label = request.form['label']
    gradcam_filename = request.form['gradcam_filename']
    # Read all rows, filter out the one to delete
    rows = []
    if os.path.exists('user_history.csv'):
        with open('user_history.csv', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not (row[0] == username and row[1] == video_filename and row[2] == label and row[3] == gradcam_filename):
                    rows.append(row)
    # Write back filtered rows
    with open('user_history.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return redirect(url_for('admin_dashboard'))


# Upload route (user only)
@app.route('/upload', methods=['POST'])
def upload():
    if session.get('role') != 'user':
        return redirect(url_for('login'))
    video = request.files['video']
    if video:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
        video.save(video_path)
        session['video_filename'] = video.filename

        # Extract first frame and save as chart
        frames = extract_frames(video_path, max_frames=1)
        if frames:
            frame = frames[0]
            chart_path = os.path.join(CHART_FOLDER, f"{os.path.splitext(video.filename)[0]}_frame.jpg")
            cv2.imwrite(chart_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            chart_url = '/' + chart_path.replace('\\', '/')
        else:
            chart_url = None

        return render_template('show_chart.html', chart_path=chart_url, video_filename=video.filename)
    return redirect(url_for('user_dashboard'))


@app.route('/show_graph')
def show_graph():
    if not os.path.exists('history.json'):
        return render_template('show_graph.html', plot_data=None)
    with open('history.json', 'r') as f:
        history = json.load(f)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(history['accuracy'], label='Train')
    axs[0].plot(history['val_accuracy'], label='Validation')
    axs[0].set_title('Accuracy Over Epochs')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    axs[1].plot(history['loss'], label='Train')
    axs[1].plot(history['val_loss'], label='Validation')
    axs[1].set_title('Loss Over Epochs')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    plt.tight_layout()

    img = io.BytesIO()
    FigureCanvas(fig).print_png(img)
    img.seek(0)
    plot_data = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)

    return render_template('show_graph.html', plot_data=plot_data)

# Admin visualizations page (base64 method like user graph)
@app.route('/admin/visualizations', methods=['GET'])
def admin_visualizations():
    def img_to_base64(path):
        import base64
        if not os.path.exists(path):
            return None
        with open(path, 'rb') as img_f:
            return base64.b64encode(img_f.read()).decode('utf-8')

    img_paths = {
        'accuracy_loss': os.path.join(app.root_path, 'static', 'charts', 'accuracy_loss.png'),
        'confusion_matrix': os.path.join(app.root_path, 'static', 'charts', 'confusion_matrix.png'),
        'feature_map': os.path.join(app.root_path, 'static', 'charts', 'feature_map.png'),
        'saliency_map': os.path.join(app.root_path, 'static', 'charts', 'saliency_map.png'),
        'tsne': os.path.join(app.root_path, 'static', 'charts', 'tsne.png')
    }
    images = {k: img_to_base64(v) for k, v in img_paths.items()}
    return render_template('admin_visualizations.html', images=images)
# Predict route (user only, handles file upload and prediction)
@app.route('/predict', methods=['POST'])
def predict():
    if session.get('role') != 'user':
        return redirect(url_for('login'))

    # If video file is uploaded directly
    if 'video' in request.files:
        video = request.files['video']
        if video:
            video_filename = video.filename
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
            video.save(video_path)
        else:
            return redirect(url_for('user_dashboard'))
    else:
        # If only filename is posted (from show_chart.html)
        video_filename = request.form['video_filename']
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)

    # Create a fire alert flag file if fire is detected
    label, _ = predict_video(video_path, model)

# Ensure exact match before triggering alert
    if label.strip() == '🔥 FIRE':
        with open('fire_alert.flag', 'w') as f:
            f.write('1')
    else:
    # Optional: clear any previous alert
        if os.path.exists('fire_alert.flag'):
            os.remove('fire_alert.flag')

    frames = extract_frames(video_path, max_frames=1)
    if frames:
        cam = get_grad_cam(frames[0].astype('float32') / 255.0, model)
        gradcam_filename = f"{os.path.splitext(video_filename)[0]}_gradcam.jpg"
        gradcam_path = os.path.join(GRADCAM_FOLDER, gradcam_filename)
        save_gradcam_overlay(frames[0]/255.0, cam, gradcam_path)
        gradcam_url = '/' + gradcam_path.replace('\\', '/')
    else:
        gradcam_url = None
        gradcam_filename = ''
    video_url = url_for('uploaded_file', filename=video_filename)
    # Save user prediction history
    username = session.get('username')
    with open('user_history.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([username, video_filename, label, gradcam_filename])
    return render_template('result.html', label=label, gradcam_path=gradcam_url, video_path=video_url)


# User prediction history page
@app.route('/user/history', methods=['GET'])
def user_history():
    if session.get('role') != 'user':
        return redirect(url_for('login'))
    username = session.get('username')
    history = []
    if os.path.exists('user_history.csv'):
        with open('user_history.csv', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            history = [row for row in reader if row[0] == username]
    return render_template('user_history.html', history=history)

# Delete a history entry
@app.route('/user/history/delete', methods=['POST'])
def delete_history():
    if session.get('role') != 'user':
        return redirect(url_for('login'))
    username = session.get('username')
    video_filename = request.form['video_filename']
    label = request.form['label']
    gradcam_filename = request.form['gradcam_filename']
    # Read all rows, filter out the one to delete
    rows = []
    if os.path.exists('user_history.csv'):
        with open('user_history.csv', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not (row[0] == username and row[1] == video_filename and row[2] == label and row[3] == gradcam_filename):
                    rows.append(row)
    # Write back filtered rows
    with open('user_history.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return redirect(url_for('user_history'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)