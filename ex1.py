import subprocess
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

def plot_histogram(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    histogram = histogram.cumsum()
    return histogram

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def calculate_histograms(frames, histograms, output_folder):
    for frame in frames:
        hist = plot_histogram(frame)
        histograms.append(hist)

def find_minimal_similarity(histograms):
    histograms_diff = []
    for i in range(len(histograms) - 1):
        diff = np.linalg.norm(histograms[i] - histograms[i+1])
        histograms_diff.append(diff)
    j = np.argmax(histograms_diff)
    return j, j+1

def main(video_path, video_type):
    """
    Main entry point for ex1
    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame number for which the scene cut was detected (i.e. the last frame index of the first scene and the first frame index of the second scene)
    """
    filename = os.path.basename(video_path)
    video_name, suffix = os.path.splitext(filename)
    output_folder = './output/' + video_name
    frames = extract_frames(video_path)
    histograms = []
    calculate_histograms(frames, histograms, output_folder)
    return find_minimal_similarity(histograms)
