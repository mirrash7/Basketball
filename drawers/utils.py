"""
A utility module providing functions for drawing shapes on video frames.

This module includes functions to draw triangles and ellipses on frames, which can be used
to represent various annotations such as player positions or ball locations in sports analysis.
"""

import cv2 
import numpy as np
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

def draw_traingle(frame, bbox, color, track_id=None, confidence=None):
    """
    Draws an upside-down filled triangle above the player's head with optional player ID and confidence.
    If the triangle is blue (BGR: blue channel dominant), the player ID font is white; otherwise, black.
    Args:
        frame (numpy.ndarray): The frame on which to draw the triangle.
        bbox (tuple): A tuple representing the bounding box (x1, y1, x2, y2).
        color (tuple): The color of the triangle in BGR format (team color).
        track_id (int, optional): The player ID to display above the triangle. Defaults to None.
        confidence (float, optional): The confidence score to display above the triangle. Only shown for actual detections. Defaults to None.
    Returns:
        numpy.ndarray: The frame with the triangle and player ID drawn on it.
    """
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    x_center = (x1 + x2) // 2
    triangle_y = y1 - 25  # 25 pixels above the top of bounding box
    triangle_width = 20
    triangle_height = 15
    # Upside-down triangle: base on top, tip pointing down
    triangle_points = np.array([
        [x_center - triangle_width//2, triangle_y],      # Top left
        [x_center + triangle_width//2, triangle_y],      # Top right
        [x_center, triangle_y + triangle_height],        # Bottom (tip)
    ], dtype=np.int32)
    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
    
    # Display confidence above the triangle (only for actual detections)
    if confidence is not None and confidence > 0.1:  # Lower threshold to show more detections
        conf_text = f"{confidence:.2f}"
        (conf_width, conf_height), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        conf_x = x_center - conf_width // 2
        conf_y = triangle_y - 15  # Above the triangle
        
        # Draw confidence text in the same color as the triangle (no background)
        cv2.putText(frame, conf_text, (conf_x, conf_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    if track_id is not None:
        # Adjust text position based on whether confidence is displayed
        text_offset = 25 if confidence is not None and confidence > 0.1 else 10
        text_y = triangle_y - text_offset
        text = str(track_id)
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_x = x_center - text_width // 2
        cv2.rectangle(frame, 
                     (text_x - 5, text_y - text_height - 5), 
                     (text_x + text_width + 5, text_y + 5), 
                     color, cv2.FILLED)
        cv2.rectangle(frame, 
                     (text_x - 5, text_y - text_height - 5), 
                     (text_x + text_width + 5, text_y + 5), 
                     (0, 0, 0), 1)
        # Automatically choose white or black font based on background brightness
        # Calculate brightness: 0.299*R + 0.587*G + 0.114*B (standard luminance formula)
        brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]  # BGR to RGB
        font_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)  # Black on light, white on dark
        cv2.putText(frame, text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, font_color, 2)
    return frame

def draw_ellipse(frame,bbox,color,track_id=None):
    """
    Draws an ellipse and an optional rectangle with a track ID on the given frame at the specified bounding box location.

    Args:
        frame (numpy.ndarray): The frame on which to draw the ellipse.
        bbox (tuple): A tuple representing the bounding box (x, y, width, height).
        color (tuple): The color of the ellipse in BGR format.
        track_id (int, optional): The track ID to display inside a rectangle. Defaults to None.

    Returns:
        numpy.ndarray: The frame with the ellipse and optional track ID drawn on it.
    """
    y2 = int(bbox[3])
    x_center, _ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)

    cv2.ellipse(
        frame,
        center=(x_center,y2),
        axes=(int(width), int(0.35*width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color = color,
        thickness=2,
        lineType=cv2.LINE_4
    )

    rectangle_width = 40
    rectangle_height=20
    x1_rect = x_center - rectangle_width//2
    x2_rect = x_center + rectangle_width//2
    y1_rect = (y2- rectangle_height//2) +15
    y2_rect = (y2+ rectangle_height//2) +15

    if track_id is not None:
        cv2.rectangle(frame,
                        (int(x1_rect),int(y1_rect) ),
                        (int(x2_rect),int(y2_rect)),
                        color,
                        cv2.FILLED)
        
        x1_text = x1_rect+12
        if track_id > 99:
            x1_text -=10
        
        cv2.putText(
            frame,
            f"{track_id}",
            (int(x1_text),int(y1_rect+15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,0),
            2
        )

    return frame

def draw_bbox(frame, bbox, color, track_id=None, confidence=None):
    """
    Draws a rectangular bounding box on the given frame.

    Args:
        frame (numpy.ndarray): The frame on which to draw the bounding box.
        bbox (tuple): A tuple representing the bounding box (x1, y1, x2, y2).
        color (tuple): The color of the bounding box in BGR format.
        track_id (int, optional): The track ID to display. Defaults to None.
        confidence (float, optional): The confidence score to display. Defaults to None.

    Returns:
        numpy.ndarray: The frame with the bounding box drawn on it.
    """
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Draw the bounding box rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Prepare text to display
    text_parts = []
    if track_id is not None:
        text_parts.append(f"ID:{track_id}")
    if confidence is not None:
        text_parts.append(f"{confidence:.2f}")
    
    if text_parts:
        text = " ".join(text_parts)
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Position text above the bounding box
        text_x = x1
        text_y = y1 - 10 if y1 > 20 else y1 + text_height + 10
        
        # Draw background rectangle for text
        cv2.rectangle(frame, 
                     (text_x, text_y - text_height - 5), 
                     (text_x + text_width + 10, text_y + 5), 
                     color, cv2.FILLED)
        
        # Draw text
        cv2.putText(frame, text, (text_x + 5, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame