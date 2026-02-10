import cv2
import numpy as np
import random
import os
from tensorflow.keras.models import load_model
import tensorflow as tf

IMG_SIZE = 128
FRAMES_PER_VIDEO = 15

def extract_frames(video_path, max_frames=FRAMES_PER_VIDEO, augment=False):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // max_frames)
    for i in range(max_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        if augment and random.random() < 0.5:
            frame = cv2.flip(frame, 1)
        frames.append(frame)
    cap.release()
    return frames

def predict_video(video_path, model):
    frames = extract_frames(video_path, max_frames=FRAMES_PER_VIDEO)
    frames = np.array(frames).astype('float32') / 255.0
    preds = model.predict(frames)
    avg_pred = np.mean(preds, axis=0)
    label = '🔥 FIRE' if np.argmax(avg_pred) == 1 else '🌲 NO FIRE'
    return label, avg_pred

def get_grad_cam(img_array, model, layer_name='Conv_1'):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    img_input = np.expand_dims(img_array, axis=0)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_input)
        loss = predictions[:, 1]  # for fire class
    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(conv_outputs.shape[1:3], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[0, :, :, i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    return cam

def save_gradcam_overlay(frame, cam, save_path):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.uint8(frame * 255), 0.6, heatmap, 0.4, 0)
    cv2.imwrite(save_path, overlay)