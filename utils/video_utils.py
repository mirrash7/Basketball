"""
A module for reading and writing video files.

This module provides utility functions to load video frames into memory and save
processed frames back to video files, with support for common video formats.
"""

import cv2
import os
import numpy as np

def read_video(video_path):
    """
    Read all frames from a video file into memory.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        tuple: (list of video frames as numpy arrays, frame rate)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Get the original frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames, fps

def read_video_memory_efficient(video_path, max_frames=None, resize_factor=1.0):
    """
    Read video frames with memory-efficient processing.
    
    Args:
        video_path (str): Path to the input video file.
        max_frames (int, optional): Maximum number of frames to read.
        resize_factor (float): Factor to resize frames (1.0 = original size, 0.5 = half size).
        
    Returns:
        tuple: (list of video frames as numpy arrays, frame rate)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Get the original frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize frame if needed
        if resize_factor != 1.0:
            height, width = frame.shape[:2]
            new_width = int(width * resize_factor)
            new_height = int(height * resize_factor)
            frame = cv2.resize(frame, (new_width, new_height))
        
        frames.append(frame)
        frame_count += 1
        
        # Stop if max_frames reached
        if max_frames and frame_count >= max_frames:
            break
    
    cap.release()
    return frames, fps

def save_video(ouput_video_frames, output_video_path, fps=24):
    """
    Save a sequence of frames as a video file.

    Creates necessary directories if they don't exist and writes frames using XVID codec.

    Args:
        ouput_video_frames (list): List of frames to save.
        output_video_path (str): Path where the video should be saved.
        fps (float): Frame rate for the output video (default: 24).
    """
    # Check if we have frames to save
    if not ouput_video_frames or len(ouput_video_frames) == 0:
        raise ValueError("No frames provided to save_video function")
    
    # If folder doesn't exist, create it
    if not os.path.exists(os.path.dirname(output_video_path)):
        os.makedirs(os.path.dirname(output_video_path))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()