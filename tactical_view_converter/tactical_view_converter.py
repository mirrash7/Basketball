import os
import sys
import pathlib
import numpy as np
import cv2
from copy import deepcopy
from .homography import Homography

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,"../"))
from utils import get_foot_position,measure_distance

class TacticalViewConverter:
    def __init__(self, court_image_path):
        self.court_image_path = court_image_path
        self.width = 300
        self.height= 161

        self.actual_width_in_meters=28
        self.actual_height_in_meters=15 
        
        # Store previous frame keypoints for fallback
        self.previous_frame_keypoints = None
        self.frames_with_insufficient_keypoints = 0
        self.total_frames_processed = 0
        
        # Motion smoothing parameters
        self.player_positions_history = {}  # Store previous positions for each player
        self.smoothing_window_size = 5  # Number of frames to average over
        self.max_velocity = 8.0  # Maximum realistic velocity in tactical view units per frame
        self.min_velocity_threshold = 0.5  # Minimum movement to consider as actual motion
        self.smoothing_alpha = 0.7  # Exponential smoothing factor (0-1)

        self.key_points = [
            # left edge
            (0,0),
            (0,int((0.91/self.actual_height_in_meters)*self.height)),
            (0,int((5.18/self.actual_height_in_meters)*self.height)),
            (0,int((10/self.actual_height_in_meters)*self.height)),
            (0,int((14.1/self.actual_height_in_meters)*self.height)),
            (0,int(self.height)),

            # Middle line
            (int(self.width/2),self.height),
            (int(self.width/2),0),
            
            # Left Free throw line
            (int((5.79/self.actual_width_in_meters)*self.width),int((5.18/self.actual_height_in_meters)*self.height)),
            (int((5.79/self.actual_width_in_meters)*self.width),int((10/self.actual_height_in_meters)*self.height)),

            # right edge
            (self.width,int(self.height)),
            (self.width,int((14.1/self.actual_height_in_meters)*self.height)),
            (self.width,int((10/self.actual_height_in_meters)*self.height)),
            (self.width,int((5.18/self.actual_height_in_meters)*self.height)),
            (self.width,int((0.91/self.actual_height_in_meters)*self.height)),
            (self.width,0),

            # Right Free throw line
            (int(((self.actual_width_in_meters-5.79)/self.actual_width_in_meters)*self.width),int((5.18/self.actual_height_in_meters)*self.height)),
            (int(((self.actual_width_in_meters-5.79)/self.actual_width_in_meters)*self.width),int((10/self.actual_height_in_meters)*self.height)),
        ]

    def validate_keypoints(self, keypoints_list):
        """
        Validates detected keypoints by comparing their proportional distances
        to the tactical view keypoints.
        
        Args:
            keypoints_list (List[List[Tuple[float, float]]]): A list containing keypoints for each frame.
                Each outer list represents a frame.
                Each inner list contains keypoints as (x, y) tuples.
                A keypoint of (0, 0) indicates that the keypoint is not detected for that frame.
        
        Returns:
            List[bool]: A list indicating whether each frame's keypoints are valid.
        """

        keypoints_list = deepcopy(keypoints_list)

        for frame_idx, frame_keypoints in enumerate(keypoints_list):
            # Check if frame_keypoints has xy attribute and is not empty
            if not hasattr(frame_keypoints, 'xy') or frame_keypoints.xy is None or len(frame_keypoints.xy.tolist()) == 0:
                continue
                
            frame_keypoints = frame_keypoints.xy.tolist()[0]
            
            # Get indices of detected keypoints (not (0, 0))
            detected_indices = [i for i, kp in enumerate(frame_keypoints) if kp[0] >0 and kp[1]>0]
            
            # Need at least 3 detected keypoints to validate proportions
            if len(detected_indices) < 3:
                continue
            
            invalid_keypoints = []
            # Validate each detected keypoint
            for i in detected_indices:
                # Skip if this is (0, 0)
                if frame_keypoints[i][0] == 0 and frame_keypoints[i][1] == 0:
                    continue

                # Choose two other random detected keypoints
                other_indices = [idx for idx in detected_indices if idx != i and idx not in invalid_keypoints]
                if len(other_indices) < 2:
                    continue

                # Take first two other indices for simplicity
                j, k = other_indices[0], other_indices[1]

                # Calculate distances between detected keypoints
                d_ij = measure_distance(frame_keypoints[i], frame_keypoints[j])
                d_ik = measure_distance(frame_keypoints[i], frame_keypoints[k])
                
                # Calculate distances between corresponding tactical keypoints
                t_ij = measure_distance(self.key_points[i], self.key_points[j])
                t_ik = measure_distance(self.key_points[i], self.key_points[k])

                # Calculate and compare proportions with 50% error margin
                if t_ij > 0 and t_ik > 0:
                    prop_detected = d_ij / d_ik if d_ik > 0 else float('inf')
                    prop_tactical = t_ij / t_ik if t_ik > 0 else float('inf')

                    error = (prop_detected - prop_tactical) / prop_tactical
                    error = abs(error)

                    if error >0.8:  # 80% error margin                        
                        keypoints_list[frame_idx].xy[0][i] *= 0
                        keypoints_list[frame_idx].xyn[0][i] *= 0
                        invalid_keypoints.append(i)
            
        return keypoints_list

    def transform_players_to_tactical_view(self, keypoints_list, player_tracks):
        """
        Transform player positions from video frame coordinates to tactical view coordinates.
        
        Args:
            keypoints_list (list): List of detected court keypoints for each frame.
            player_tracks (list): List of dictionaries containing player tracking information for each frame,
                where each dictionary maps player IDs to their bounding box coordinates.
        
        Returns:
            list: List of dictionaries where each dictionary maps player IDs to their (x, y) positions
                in the tactical view coordinate system. The list index corresponds to the frame number.
        """
        tactical_player_positions = []
        
        for frame_idx, (frame_keypoints, frame_tracks) in enumerate(zip(keypoints_list, player_tracks)):
            self.total_frames_processed += 1
            
            # Initialize empty dictionary for this frame
            tactical_positions = {}

            # Check if frame_keypoints has xy attribute and is not empty
            if not hasattr(frame_keypoints, 'xy') or frame_keypoints.xy is None or len(frame_keypoints.xy.tolist()) == 0:
                # Try to use previous frame keypoints
                if self.previous_frame_keypoints is not None:
                    # print(f"Frame {frame_idx}: No keypoints detected, using previous frame keypoints")
                    frame_keypoints = self.previous_frame_keypoints
                else:
                    # print(f"Frame {frame_idx}: No keypoints detected and no previous frame available")
                    tactical_player_positions.append(tactical_positions)
                    continue

            frame_keypoints_data = frame_keypoints.xy.tolist()[0]

            # Skip frames with insufficient keypoints
            if frame_keypoints_data is None or len(frame_keypoints_data) == 0:
                # Try to use previous frame keypoints
                if self.previous_frame_keypoints is not None:
                    # print(f"Frame {frame_idx}: Empty keypoints, using previous frame keypoints")
                    frame_keypoints = self.previous_frame_keypoints
                    frame_keypoints_data = frame_keypoints.xy.tolist()[0]
                else:
                    # print(f"Frame {frame_idx}: Empty keypoints and no previous frame available")
                    tactical_player_positions.append(tactical_positions)
                    continue
            
            # Get detected keypoints for this frame
            detected_keypoints = frame_keypoints_data
            
            # Filter out undetected keypoints (those with coordinates (0,0))
            valid_indices = [i for i, kp in enumerate(detected_keypoints) if kp[0] > 0 and kp[1] > 0]
            
            # Need at least 4 points for a reliable homography
            if len(valid_indices) < 4:
                # Try to use previous frame keypoints
                if self.previous_frame_keypoints is not None:
                    # print(f"Frame {frame_idx}: Insufficient keypoints ({len(valid_indices)}), using previous frame keypoints")
                    self.frames_with_insufficient_keypoints += 1
                    frame_keypoints = self.previous_frame_keypoints
                    frame_keypoints_data = frame_keypoints.xy.tolist()[0]
                    detected_keypoints = frame_keypoints_data
                    valid_indices = [i for i, kp in enumerate(detected_keypoints) if kp[0] > 0 and kp[1] > 0]
                    
                    # If still insufficient, skip this frame
                    if len(valid_indices) < 4:
                        # print(f"Frame {frame_idx}: Still insufficient keypoints after using previous frame, skipping")
                        tactical_player_positions.append(tactical_positions)
                        continue
                else:
                    # print(f"Frame {frame_idx}: Insufficient keypoints ({len(valid_indices)}) and no previous frame available")
                    tactical_player_positions.append(tactical_positions)
                    continue
            
            # Store current frame keypoints for next iteration (if they're sufficient)
            try:
                # Create a copy of the keypoints object for storage
                if hasattr(frame_keypoints, 'data'):
                    # Store the data tensor and orig_shape directly
                    import torch
                    
                    # Clone the data tensor
                    cloned_data = frame_keypoints.data.clone()
                    
                    # Store as a simple dictionary with the necessary attributes
                    self.previous_frame_keypoints = type('KeypointsCopy', (), {
                        'data': cloned_data,
                        'orig_shape': frame_keypoints.orig_shape,
                        'xy': cloned_data[:, :, :2],
                        'xyn': cloned_data[:, :, :2] / torch.tensor([frame_keypoints.orig_shape[1], frame_keypoints.orig_shape[0]]),
                        'conf': cloned_data[:, :, 2] if cloned_data.shape[2] > 2 else None
                    })()
                else:
                    # Fallback: store the original object
                    self.previous_frame_keypoints = frame_keypoints
                    
            except Exception as e:
                # print(f"Warning: Could not store keypoints for frame {frame_idx}: {e}")
                # Still store the original object as fallback
                self.previous_frame_keypoints = frame_keypoints
            
            # Create source and target point arrays for homography
            source_points = np.array([detected_keypoints[i] for i in valid_indices], dtype=np.float32)
            target_points = np.array([self.key_points[i] for i in valid_indices], dtype=np.float32)
            
            try:
                # Create homography transformer
                homography = Homography(source_points, target_points)
                
                # Transform each player's position
                for player_id, player_data in frame_tracks.items():
                    # Skip referees - only show players on tactical view
                    if player_data.get("class", "Player") == "Ref":
                        continue
                        
                    bbox = player_data["bbox"]
                    # Use bottom center of bounding box as player position
                    player_position = np.array([get_foot_position(bbox)])
                    # Transform to tactical view coordinates
                    tactical_position = homography.transform_points(player_position)

                    # If tactical position is not in the tactical view, skip
                    if tactical_position[0][0] < 0 or tactical_position[0][0] > self.width or tactical_position[0][1] < 0 or tactical_position[0][1] > self.height:
                        continue

                    # Apply motion smoothing to create realistic movement
                    raw_position = tactical_position[0].tolist()
                    smoothed_position = self.smooth_player_motion(player_id, raw_position, frame_idx)
                    
                    tactical_positions[player_id] = smoothed_position
                    
            except (ValueError, cv2.error) as e:
                # If homography fails, continue with empty dictionary
                # print(f"Frame {frame_idx}: Homography failed: {e}")
                pass
            
            tactical_player_positions.append(tactical_positions)
        
        # Print statistics
        if self.total_frames_processed > 0:
            percentage = (self.frames_with_insufficient_keypoints / self.total_frames_processed) * 100
            # print(f"Tactical view statistics: {self.frames_with_insufficient_keypoints}/{self.total_frames_processed} frames used previous keypoints ({percentage:.1f}%)")
        
        return tactical_player_positions

    def smooth_player_motion(self, player_id, new_position, frame_idx):
        """
        Apply motion smoothing to player positions to create realistic movement.
        
        Args:
            player_id: ID of the player
            new_position: New detected position (x, y)
            frame_idx: Current frame index
            
        Returns:
            tuple: Smoothed position (x, y)
        """
        if player_id not in self.player_positions_history:
            # First time seeing this player, initialize history
            self.player_positions_history[player_id] = {
                'positions': [new_position],
                'frame_indices': [frame_idx],
                'last_smoothed': new_position
            }
            return new_position
        
        history = self.player_positions_history[player_id]
        
        # Add new position to history
        history['positions'].append(new_position)
        history['frame_indices'].append(frame_idx)
        
        # Keep only recent positions within window
        if len(history['positions']) > self.smoothing_window_size:
            history['positions'] = history['positions'][-self.smoothing_window_size:]
            history['frame_indices'] = history['frame_indices'][-self.smoothing_window_size:]
        
        # Apply velocity-based filtering
        smoothed_position = self._apply_velocity_filtering(player_id, new_position, history)
        
        # Apply exponential smoothing
        final_position = self._apply_exponential_smoothing(player_id, smoothed_position, history)
        
        # Update last smoothed position
        history['last_smoothed'] = final_position
        
        return final_position
    
    def _apply_velocity_filtering(self, player_id, new_position, history):
        """
        Filter out unrealistic movements based on velocity constraints.
        
        Args:
            player_id: ID of the player
            new_position: New detected position
            history: Player's position history
            
        Returns:
            tuple: Velocity-filtered position
        """
        if len(history['positions']) < 2:
            return new_position
        
        last_position = history['last_smoothed']
        
        # Calculate velocity (distance per frame)
        import math
        dx = new_position[0] - last_position[0]
        dy = new_position[1] - last_position[1]
        velocity = math.sqrt(dx*dx + dy*dy)
        
        # If velocity is too high, limit it
        if velocity > self.max_velocity:
            # Scale down the movement to max velocity
            scale_factor = self.max_velocity / velocity
            dx *= scale_factor
            dy *= scale_factor
            filtered_position = (last_position[0] + dx, last_position[1] + dy)
        else:
            filtered_position = new_position
        
        return filtered_position
    
    def _apply_exponential_smoothing(self, player_id, filtered_position, history):
        """
        Apply exponential smoothing to reduce jitter.
        
        Args:
            player_id: ID of the player
            filtered_position: Velocity-filtered position
            history: Player's position history
            
        Returns:
            tuple: Exponentially smoothed position
        """
        last_smoothed = history['last_smoothed']
        
        # Apply exponential smoothing
        smoothed_x = self.smoothing_alpha * filtered_position[0] + (1 - self.smoothing_alpha) * last_smoothed[0]
        smoothed_y = self.smoothing_alpha * filtered_position[1] + (1 - self.smoothing_alpha) * last_smoothed[1]
        
        return (smoothed_x, smoothed_y)
    
    def _apply_median_filtering(self, positions):
        """
        Apply median filtering to remove outliers.
        
        Args:
            positions: List of recent positions
            
        Returns:
            tuple: Median filtered position
        """
        if len(positions) < 3:
            return positions[-1] if positions else (0, 0)
        
        # Separate x and y coordinates
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        # Calculate median
        x_coords.sort()
        y_coords.sort()
        median_x = x_coords[len(x_coords) // 2]
        median_y = y_coords[len(y_coords) // 2]
        
        return (median_x, median_y)

    def reset_motion_smoothing(self):
        """
        Reset motion smoothing history for all players.
        Call this when processing a new video or when you want to clear smoothing state.
        """
        self.player_positions_history = {}
        # print("Motion smoothing history reset")
    
    def configure_smoothing(self, window_size=None, max_velocity=None, smoothing_alpha=None):
        """
        Configure motion smoothing parameters.
        
        Args:
            window_size: Number of frames to average over (default: 5)
            max_velocity: Maximum realistic velocity in tactical view units per frame (default: 8.0)
            smoothing_alpha: Exponential smoothing factor 0-1 (default: 0.7)
        """
        if window_size is not None:
            self.smoothing_window_size = window_size
        if max_velocity is not None:
            self.max_velocity = max_velocity
        if smoothing_alpha is not None:
            self.smoothing_alpha = max(0.0, min(1.0, smoothing_alpha))  # Clamp between 0 and 1
        
        # print(f"Motion smoothing configured: window_size={self.smoothing_window_size}, "
        #       f"max_velocity={self.max_velocity}, smoothing_alpha={self.smoothing_alpha}")

