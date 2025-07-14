from ultralytics import YOLO
import supervision as sv
import numpy as np
import torch
import copy
import sys 
sys.path.append('../')
from utils import read_stub, save_stub


class CourtKeypointDetector:
    """
    The CourtKeypointDetector class uses a YOLO model to detect court keypoints in image frames. 
    It also provides functionality to draw these detected keypoints on the frames.
    """
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.inversion_stats = {
            'total_frames': 0,
            'inversions_detected': 0,
            'corrections_applied': 0
        }
    
    def get_inversion_statistics(self):
        """
        Get statistics about keypoint inversions detected and corrected.
        
        Returns:
            dict: Statistics about inversions
        """
        return self.inversion_stats.copy()
    
    def reset_inversion_statistics(self):
        """
        Reset inversion statistics.
        """
        self.inversion_stats = {
            'total_frames': 0,
            'inversions_detected': 0,
            'corrections_applied': 0
        }
    
    def check_and_correct_keypoint_inversion(self, keypoints):
        """
        Check if keypoints are inverted and correct them if necessary.
        
        Args:
            keypoints: Keypoints object from YOLO detection
            
        Returns:
            Corrected keypoints object
        """
        self.inversion_stats['total_frames'] += 1
        
        if not hasattr(keypoints, 'xy') or keypoints.xy is None or len(keypoints.xy.tolist()) == 0:
            return keypoints
            
        frame_keypoints = keypoints.xy.tolist()[0]
        
        # Debug: Print all detected keypoints
        # print(f"DEBUG: Frame keypoints before correction:")
        for i, point in enumerate(frame_keypoints):
            if point[0] > 0 and point[1] > 0:
                # print(f"  Point {i}: ({point[0]:.1f}, {point[1]:.1f})")
                pass
        
        # Define the keypoint pairs that should be switched if inversion is detected
        keypoint_pairs = [(0, 15), (1, 14), (2, 13), (3, 12), (4, 11), (5, 10), (8, 16), (9, 17)]
        
        # Check 1: Rightmost of points 0,1,2,3,4,5 should be to the left of leftmost of points 8,9
        left_points = [0, 1, 2, 3, 4, 5]
        middle_points = [8, 9]
        
        # Get detected left points
        detected_left = [(i, frame_keypoints[i]) for i in left_points 
                        if frame_keypoints[i][0] > 0 and frame_keypoints[i][1] > 0]
        detected_middle = [(i, frame_keypoints[i]) for i in middle_points 
                          if frame_keypoints[i][0] > 0 and frame_keypoints[i][1] > 0]
        
        inversion_detected = False
        
        if detected_left and detected_middle:
            # Find rightmost of left points
            rightmost_left_x = max(point[1][0] for point in detected_left)
            # Find leftmost of middle points
            leftmost_middle_x = min(point[1][0] for point in detected_middle)
            
            # Check if rightmost left point is to the right of leftmost middle point
            if rightmost_left_x > leftmost_middle_x:
                inversion_detected = True
                # print(f"Keypoint inversion detected: left points ({rightmost_left_x:.1f}) > middle points ({leftmost_middle_x:.1f})")
        
        # Check 2: Leftmost of points 10,11,12,13,14,15 should be to the right of rightmost of points 16,17
        right_points = [10, 11, 12, 13, 14, 15]
        right_middle_points = [16, 17]
        
        # Get detected right points
        detected_right = [(i, frame_keypoints[i]) for i in right_points 
                         if frame_keypoints[i][0] > 0 and frame_keypoints[i][1] > 0]
        detected_right_middle = [(i, frame_keypoints[i]) for i in right_middle_points 
                                if frame_keypoints[i][0] > 0 and frame_keypoints[i][1] > 0]
        
        if detected_right and detected_right_middle:
            # Find leftmost of right points
            leftmost_right_x = min(point[1][0] for point in detected_right)
            # Find rightmost of right middle points
            rightmost_right_middle_x = max(point[1][0] for point in detected_right_middle)
            
            # Check if leftmost right point is to the left of rightmost right middle point
            if leftmost_right_x < rightmost_right_middle_x:
                inversion_detected = True
                # print(f"Keypoint inversion detected: right points ({leftmost_right_x:.1f}) < right middle points ({rightmost_right_middle_x:.1f})")
        
        # If inversion is detected, swap the keypoint pairs
        if inversion_detected:
            self.inversion_stats['inversions_detected'] += 1
            # print("Correcting keypoint inversion...")
            
            # Create a copy of the data tensor
            try:
                modified_data = keypoints.data.clone()
                # print(f"DEBUG: Successfully cloned keypoints data, shape: {modified_data.shape}")
            except Exception as e:
                # print(f"Warning: Could not clone keypoints data: {e}")
                return keypoints
            
            corrections_made = 0
            
            # Swap the detected keypoints according to pairs
            for pair in keypoint_pairs:
                idx1, idx2 = pair
                
                # Check if at least one point in the pair is detected
                point1_detected = frame_keypoints[idx1][0] > 0 and frame_keypoints[idx1][1] > 0
                point2_detected = frame_keypoints[idx2][0] > 0 and frame_keypoints[idx2][1] > 0
                
                # Check if at least one point in the pair is detected
                # print(f"DEBUG: Pair {idx1}-{idx2}: Point {idx1} detected: {point1_detected}, Point {idx2} detected: {point2_detected}")
                
                if point1_detected or point2_detected:
                    # print(f"DEBUG: Attempting to swap keypoints {idx1} and {idx2}")
                    
                    if point1_detected and point2_detected:
                        # Both points detected - swap their coordinates
                        # print(f"DEBUG: Both points detected - swapping coordinates")
                        # print(f"DEBUG: Before swap - {idx1}: {modified_data[0, idx1].tolist()}, {idx2}: {modified_data[0, idx2].tolist()}")
                        
                        try:
                            temp_data = modified_data[0, idx1].clone()
                            modified_data[0, idx1] = modified_data[0, idx2].clone()
                            modified_data[0, idx2] = temp_data
                            corrections_made += 1
                            # print(f"DEBUG: After swap - {idx1}: {modified_data[0, idx1].tolist()}, {idx2}: {modified_data[0, idx2].tolist()}")
                            # print(f"Swapped keypoints {idx1} and {idx2}")
                        except Exception as e:
                            # print(f"Warning: Could not swap keypoints {idx1} and {idx2}: {e}")
                            continue
                    
                    elif point1_detected:
                        # Only point1 detected - move its coordinates to point2 and clear point1
                        # print(f"DEBUG: Only point {idx1} detected - moving coordinates to point {idx2}")
                        # print(f"DEBUG: Before move - {idx1}: {modified_data[0, idx1].tolist()}, {idx2}: {modified_data[0, idx2].tolist()}")
                        
                        try:
                            # Copy point1 coordinates to point2
                            modified_data[0, idx2] = modified_data[0, idx1].clone()
                            # Clear point1 (set to 0,0,0)
                            modified_data[0, idx1] = torch.tensor([0.0, 0.0, 0.0])
                            corrections_made += 1
                            # print(f"DEBUG: After move - {idx1}: {modified_data[0, idx1].tolist()}, {idx2}: {modified_data[0, idx2].tolist()}")
                            # print(f"Moved keypoint {idx1} coordinates to {idx2}")
                        except Exception as e:
                            # print(f"Warning: Could not move keypoint {idx1} to {idx2}: {e}")
                            continue
                    
                    elif point2_detected:
                        # Only point2 detected - move its coordinates to point1 and clear point2
                        # print(f"DEBUG: Only point {idx2} detected - moving coordinates to point {idx1}")
                        # print(f"DEBUG: Before move - {idx1}: {modified_data[0, idx1].tolist()}, {idx2}: {modified_data[0, idx2].tolist()}")
                        
                        try:
                            # Copy point2 coordinates to point1
                            modified_data[0, idx1] = modified_data[0, idx2].clone()
                            # Clear point2 (set to 0,0,0)
                            modified_data[0, idx2] = torch.tensor([0.0, 0.0, 0.0])
                            corrections_made += 1
                            # print(f"DEBUG: After move - {idx1}: {modified_data[0, idx1].tolist()}, {idx2}: {modified_data[0, idx2].tolist()}")
                            # print(f"Moved keypoint {idx2} coordinates to {idx1}")
                        except Exception as e:
                            # print(f"Warning: Could not move keypoint {idx2} to {idx1}: {e}")
                            continue
                else:
                    # print(f"DEBUG: Skipping pair {idx1}-{idx2} - neither point detected")
                    pass
            
            # print(f"DEBUG: Total corrections made: {corrections_made}")
            
            # Create new keypoints object with modified data
            corrected_keypoints = self.create_new_keypoints_object(keypoints, modified_data)
            if corrected_keypoints is None:
                # print("Warning: Could not create corrected keypoints object, returning original")
                return keypoints
            
            # Debug: Print corrected keypoints
            if hasattr(corrected_keypoints, 'xy') and corrected_keypoints.xy is not None:
                corrected_frame_keypoints = corrected_keypoints.xy.tolist()[0]
                # print(f"DEBUG: Frame keypoints after correction:")
                for i, (x, y) in enumerate(corrected_frame_keypoints):
                    if x > 0 and y > 0:
                        # print(f"  Point {i}: ({x:.1f}, {y:.1f})")
                        pass
                
                # Check if any coordinates actually changed
                changes_detected = 0
                for i in range(len(frame_keypoints)):
                    if (frame_keypoints[i][0] > 0 and frame_keypoints[i][1] > 0 and 
                        corrected_frame_keypoints[i][0] > 0 and corrected_frame_keypoints[i][1] > 0):
                        if (abs(frame_keypoints[i][0] - corrected_frame_keypoints[i][0]) > 0.1 or 
                            abs(frame_keypoints[i][1] - corrected_frame_keypoints[i][1]) > 0.1):
                            changes_detected += 1
                            # print(f"DEBUG: Point {i} changed from ({frame_keypoints[i][0]:.1f}, {frame_keypoints[i][1]:.1f}) to ({corrected_frame_keypoints[i][0]:.1f}, {corrected_frame_keypoints[i][1]:.1f})")
                
                # print(f"DEBUG: Total coordinate changes detected: {changes_detected}")
            
            self.inversion_stats['corrections_applied'] += corrections_made
            
            # Validate the corrected keypoints
            if self.validate_corrected_keypoints(corrected_keypoints):
                # print(f"Keypoint correction successful - {corrections_made} pairs swapped")
                return corrected_keypoints
            else:
                # print("Warning: Keypoint correction may have introduced invalid spatial relationships")
                # print("DEBUG: Returning corrected keypoints anyway for testing purposes")
                return corrected_keypoints
        
        return keypoints
    
    def visualize_keypoint_correction(self, frame, original_keypoints, corrected_keypoints):
        """
        Visualize the keypoint correction for debugging purposes.
        
        Args:
            frame: Original video frame
            original_keypoints: Keypoints before correction
            corrected_keypoints: Keypoints after correction
            
        Returns:
            Frame with keypoints drawn
        """
        import cv2
        
        # Create a copy of the frame
        vis_frame = frame.copy()
        
        if hasattr(original_keypoints, 'xy') and original_keypoints.xy is not None:
            original_xy = original_keypoints.xy.tolist()[0]
            for i, (x, y) in enumerate(original_xy):
                if x > 0 and y > 0:
                    # Draw original keypoints in red
                    cv2.circle(vis_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv2.putText(vis_frame, str(i), (int(x)+5, int(y)-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if hasattr(corrected_keypoints, 'xy') and corrected_keypoints.xy is not None:
            corrected_xy = corrected_keypoints.xy.tolist()[0]
            for i, (x, y) in enumerate(corrected_xy):
                if x > 0 and y > 0:
                    # Draw corrected keypoints in green
                    cv2.circle(vis_frame, (int(x), int(y)), 8, (0, 255, 0), 2)
                    cv2.putText(vis_frame, str(i), (int(x)+5, int(y)-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis_frame
    
    def get_court_keypoints(self, frames,read_from_stub=False, stub_path=None):
        """
        Detect court keypoints for a batch of frames using the YOLO model. If requested, 
        attempts to read previously detected keypoints from a stub file before running the model.

        Args:
            frames (list of numpy.ndarray): A list of frames (images) on which to detect keypoints.
            read_from_stub (bool, optional): Indicates whether to read keypoints from a stub file 
                instead of running the detection model. Defaults to False.
            stub_path (str, optional): The file path for the stub file. If None, a default path may be used. 
                Defaults to None.

        Returns:
            list: A list of detected keypoints for each input frame.
        """
        court_keypoints = read_stub(read_from_stub,stub_path)
        if court_keypoints is not None:
            if len(court_keypoints) == len(frames):
                # Apply keypoint inversion correction to cached results
                corrected_keypoints = []
                for keypoints in court_keypoints:
                    corrected = self.check_and_correct_keypoint_inversion(keypoints)
                    corrected_keypoints.append(corrected)
                return corrected_keypoints
        
        batch_size=20
        court_keypoints = []
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.5)
            for detection in detections_batch:
                # Apply keypoint inversion correction
                corrected_keypoints = self.check_and_correct_keypoint_inversion(detection.keypoints)
                court_keypoints.append(corrected_keypoints)

        save_stub(stub_path,court_keypoints)
        
        # Print statistics
        stats = self.get_inversion_statistics()
        if stats['total_frames'] > 0:
            inversion_rate = (stats['inversions_detected'] / stats['total_frames']) * 100
            # print(f"Keypoint inversion statistics:")
            # print(f"  Total frames processed: {stats['total_frames']}")
            # print(f"  Inversions detected: {stats['inversions_detected']} ({inversion_rate:.1f}%)")
            # print(f"  Keypoint pairs corrected: {stats['corrections_applied']}")
        
        return court_keypoints
    
    def validate_corrected_keypoints(self, keypoints):
        """
        Validate that corrected keypoints follow expected spatial relationships.
        
        Args:
            keypoints: Keypoints object to validate
            
        Returns:
            bool: True if keypoints are valid, False otherwise
        """
        if not hasattr(keypoints, 'xy') or keypoints.xy is None or len(keypoints.xy.tolist()) == 0:
            return False
            
        frame_keypoints = keypoints.xy.tolist()[0]
        
        # Check basic spatial relationships
        validations = []
        
        # Check 1: Left points should be to the left of middle points
        left_points = [0, 1, 2, 3, 4, 5]
        middle_points = [8, 9]
        
        detected_left = [(i, frame_keypoints[i]) for i in left_points 
                        if frame_keypoints[i][0] > 0 and frame_keypoints[i][1] > 0]
        detected_middle = [(i, frame_keypoints[i]) for i in middle_points 
                          if frame_keypoints[i][0] > 0 and frame_keypoints[i][1] > 0]
        
        if detected_left and detected_middle:
            rightmost_left_x = max(point[1][0] for point in detected_left)
            leftmost_middle_x = min(point[1][0] for point in detected_middle)
            validation1 = rightmost_left_x < leftmost_middle_x
            validations.append(validation1)
            # print(f"DEBUG: Validation 1 - Left points should be left of middle: {validation1} ({rightmost_left_x:.1f} < {leftmost_middle_x:.1f})")
        
        # Check 2: Right points should be to the right of right middle points
        right_points = [10, 11, 12, 13, 14, 15]
        right_middle_points = [16, 17]
        
        detected_right = [(i, frame_keypoints[i]) for i in right_points 
                         if frame_keypoints[i][0] > 0 and frame_keypoints[i][1] > 0]
        detected_right_middle = [(i, frame_keypoints[i]) for i in right_middle_points 
                                if frame_keypoints[i][0] > 0 and frame_keypoints[i][1] > 0]
        
        if detected_right and detected_right_middle:
            leftmost_right_x = min(point[1][0] for point in detected_right)
            rightmost_right_middle_x = max(point[1][0] for point in detected_right_middle)
            validation2 = leftmost_right_x > rightmost_right_middle_x
            validations.append(validation2)
            # print(f"DEBUG: Validation 2 - Right points should be right of right middle: {validation2} ({leftmost_right_x:.1f} > {rightmost_right_middle_x:.1f})")
        
        # Check 3: Top points should be above bottom points (basic vertical check)
        top_points = [0, 8, 15]
        bottom_points = [5, 6, 10]
        
        detected_top = [(i, frame_keypoints[i]) for i in top_points 
                       if frame_keypoints[i][0] > 0 and frame_keypoints[i][1] > 0]
        detected_bottom = [(i, frame_keypoints[i]) for i in bottom_points 
                          if frame_keypoints[i][0] > 0 and frame_keypoints[i][1] > 0]
        
        if detected_top and detected_bottom:
            bottommost_top_y = max(point[1][1] for point in detected_top)
            topmost_bottom_y = min(point[1][1] for point in detected_bottom)
            validation3 = bottommost_top_y < topmost_bottom_y
            validations.append(validation3)
            # print(f"DEBUG: Validation 3 - Top points should be above bottom: {validation3} ({bottommost_top_y:.1f} < {topmost_bottom_y:.1f})")
        
        # Return True if all validations pass
        result = all(validations) if validations else True
        # print(f"DEBUG: Overall validation result: {result} (validations: {validations})")
        return result
    
    def create_new_keypoints_object(self, original_keypoints, modified_data):
        """
        Create a new keypoints object with modified data.
        
        Args:
            original_keypoints: Original keypoints object
            modified_data: Modified data tensor
            
        Returns:
            New keypoints object
        """
        try:
            # Try to create new object with same type
            new_keypoints = type(original_keypoints)(modified_data, original_keypoints.orig_shape)
            return new_keypoints
        except Exception as e:
            # print(f"Could not create new keypoints object: {e}")
            return None