from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import sys 
sys.path.append('../')
from utils import read_stub, save_stub


class KalmanBallFilter:
    """Kalman filter for ball tracking with motion prediction"""
    
    def __init__(self):
        # State: [x, y, vx, vy] - position and velocity
        self.dt = 1.0  # Time step
        self.A = np.array([[1, 0, self.dt, 0],
                          [0, 1, 0, self.dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])  # State transition matrix
        
        self.H = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]])  # Observation matrix
        
        # Process noise covariance
        self.Q = np.diag([10, 10, 50, 50])  # Position and velocity noise
        
        # Measurement noise covariance
        self.R = np.diag([20, 20])  # Position measurement noise
        
        self.x = None  # State estimate
        self.P = None  # Error covariance
        
    def init(self, measurement):
        """Initialize filter with first measurement"""
        x, y = measurement[:2]
        self.x = np.array([x, y, 0, 0])  # Initial velocity = 0
        self.P = np.diag([100, 100, 100, 100])  # High initial uncertainty
        
    def predict(self):
        """Predict next state"""
        if self.x is None:
            return None
            
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[:2]  # Return predicted position
        
    def update(self, measurement):
        """Update with new measurement"""
        if self.x is None:
            self.init(measurement)
            return self.x[:2]
            
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update
        z = measurement[:2]  # Measurement
        y = z - self.H @ self.x  # Innovation
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return self.x[:2]


class BallTracker:
    """
    A class that handles basketball detection and tracking using YOLO.

    This class provides methods to detect the ball in video frames, process detections
    in batches, and refine tracking results through filtering and interpolation.
    """
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.kalman_filter = KalmanBallFilter()
        self.track_history = []
        self.confidence_threshold = 0.5
        self.min_track_length = 5

    def detect_frames(self, frames):
        """
        Detect the ball in a sequence of frames using batch processing.

        Args:
            frames (list): List of video frames to process.

        Returns:
            list: YOLO detection results for each frame.
        """
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=self.confidence_threshold)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Get ball tracking results for a sequence of frames with improved fallback mechanisms.

        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries containing ball tracking information for each frame.
        """
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                # Ensure visibility even for cached results
                tracks = self.ensure_ball_visibility(tracks, frames)
                return tracks

        # Use adaptive detection for better results
        detections = self.adaptive_ball_detection(frames, initial_confidence=0.3)

        tracks = []
        last_valid_bbox = None

        for frame_num, detection in enumerate(detections):
            tracks.append({})
            chosen_bbox = None
            max_confidence = 0
            
            if detection is not None:
                cls_names = detection.names
                cls_names_inv = {v: k for k, v in cls_names.items()}

                # Convert to supervision Detection format
                detection_supervision = sv.Detections.from_ultralytics(detection)
                
                for frame_detection in detection_supervision:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]
                    confidence = frame_detection[2]
                    
                    if cls_id == cls_names_inv.get('Ball', -1):
                        if max_confidence < confidence:
                            chosen_bbox = bbox
                            max_confidence = confidence

            if chosen_bbox is not None:
                tracks[frame_num][1] = {"bbox": chosen_bbox, "confidence": max_confidence}
                last_valid_bbox = chosen_bbox
            else:
                # No detection - try prediction
                if frame_num > 0 and last_valid_bbox is not None:
                    # Simple prediction based on last position
                    predicted_bbox, pred_confidence = self.predict_ball_position(tracks, frame_num)
                    if predicted_bbox is not None:
                        tracks[frame_num][1] = {"bbox": predicted_bbox, "predicted": True, "confidence": pred_confidence}
                        last_valid_bbox = predicted_bbox
                    else:
                        # Use last valid position
                        tracks[frame_num][1] = {"bbox": last_valid_bbox, "fallback": True, "confidence": 0.5}
                elif last_valid_bbox is not None:
                    # Use last valid position
                    tracks[frame_num][1] = {"bbox": last_valid_bbox, "fallback": True, "confidence": 0.5}
                else:
                    # No previous position - use center of frame
                    h, w = frames[frame_num].shape[:2]
                    center_bbox = [w//2 - 10, h//2 - 10, w//2 + 10, h//2 + 10]
                    tracks[frame_num][1] = {"bbox": center_bbox, "center_fallback": True, "confidence": 0.2}
                    last_valid_bbox = center_bbox

        # Final visibility check
        tracks = self.ensure_ball_visibility(tracks, frames)
        
        save_stub(stub_path, tracks)
        
        return tracks

    def apply_kalman_filtering(self, ball_positions):
        """
        Apply Kalman filtering to smooth ball trajectory and predict positions.
        
        Args:
            ball_positions (list): List of ball positions across frames.
            
        Returns:
            list: Smoothed ball positions with Kalman filtering applied.
        """
        filtered_positions = []
        self.kalman_filter = KalmanBallFilter()  # Reset filter
        
        for i, frame_data in enumerate(ball_positions):
            current_bbox = frame_data.get(1, {}).get('bbox', [])
            
            if len(current_bbox) == 0:
                # No detection - use prediction
                predicted_pos = self.kalman_filter.predict()
                if predicted_pos is not None:
                    # Create bbox around predicted position
                    x, y = predicted_pos
                    bbox_size = 20  # Default ball size
                    filtered_bbox = [x - bbox_size//2, y - bbox_size//2, 
                                   x + bbox_size//2, y + bbox_size//2]
                    filtered_positions.append({1: {"bbox": filtered_bbox, "predicted": True}})
                else:
                    filtered_positions.append({})
            else:
                # Update filter with detection
                filtered_pos = self.kalman_filter.update(current_bbox)
                if filtered_pos is not None:
                    # Create bbox around filtered position
                    x, y = filtered_pos
                    bbox_size = current_bbox[2] - current_bbox[0]  # Use original size
                    filtered_bbox = [x - bbox_size//2, y - bbox_size//2, 
                                   x + bbox_size//2, y + bbox_size//2]
                    
                    # Preserve original confidence if it exists
                    original_confidence = frame_data.get(1, {}).get('confidence', None)
                    if original_confidence is not None:
                        filtered_positions.append({1: {"bbox": filtered_bbox, "confidence": original_confidence}})
                    else:
                        filtered_positions.append({1: {"bbox": filtered_bbox}})
                else:
                    # Preserve original confidence if it exists
                    original_confidence = frame_data.get(1, {}).get('confidence', None)
                    if original_confidence is not None:
                        filtered_positions.append({1: {"bbox": current_bbox, "confidence": original_confidence}})
                    else:
                        filtered_positions.append({1: {"bbox": current_bbox}})
        
        return filtered_positions

    def remove_wrong_detections(self,ball_positions):
        """
        Filter out incorrect ball detections based on maximum allowed movement distance.

        Args:
            ball_positions (list): List of detected ball positions across frames.

        Returns:
            list: Filtered ball positions with incorrect detections removed.
        """
        
        maximum_allowed_distance = 25
        last_good_frame_index = -1

        for i in range(len(ball_positions)):
            current_box = ball_positions[i].get(1, {}).get('bbox', [])

            if len(current_box) == 0:
                continue

            if last_good_frame_index == -1:
                # First valid detection
                last_good_frame_index = i
                continue

            last_good_box = ball_positions[last_good_frame_index].get(1, {}).get('bbox', [])
            frame_gap = i - last_good_frame_index
            adjusted_max_distance = maximum_allowed_distance * frame_gap

            if np.linalg.norm(np.array(last_good_box[:2]) - np.array(current_box[:2])) > adjusted_max_distance:
                ball_positions[i] = {}
            else:
                last_good_frame_index = i

        return ball_positions

    def interpolate_ball_positions(self,ball_positions):
        """
        Interpolate missing ball positions to create smooth tracking results.

        Args:
            ball_positions (list): List of ball positions with potential gaps.

        Returns:
            list: List of ball positions with interpolated values filling the gaps.
        """
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def advanced_interpolation(self, ball_positions, method='cubic'):
        """
        Advanced interpolation with multiple methods for smoother trajectories.
        
        Args:
            ball_positions (list): Ball positions across frames.
            method (str): Interpolation method ('linear', 'cubic', 'polynomial').
            
        Returns:
            list: Interpolated ball positions.
        """
        positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        
        # Extract valid positions and their indices
        valid_indices = []
        valid_positions = []
        
        for i, pos in enumerate(positions):
            if len(pos) == 4:
                valid_indices.append(i)
                valid_positions.append(pos)
        
        if len(valid_positions) < 2:
            return ball_positions
        
        # Convert to numpy arrays
        valid_indices = np.array(valid_indices)
        valid_positions = np.array(valid_positions)
        
        # Interpolate each coordinate separately
        interpolated_positions = []
        
        for coord_idx in range(4):  # x1, y1, x2, y2
            coord_values = valid_positions[:, coord_idx]
            
            if method == 'linear':
                from scipy.interpolate import interp1d
                f = interp1d(valid_indices, coord_values, kind='linear', 
                           bounds_error=False, fill_value='extrapolate')
                interpolated_coord = f(np.arange(len(ball_positions)))
                
            elif method == 'cubic':
                from scipy.interpolate import CubicSpline
                if len(valid_indices) >= 4:
                    cs = CubicSpline(valid_indices, coord_values)
                    interpolated_coord = cs(np.arange(len(ball_positions)))
                else:
                    # Fall back to linear if not enough points
                    from scipy.interpolate import interp1d
                    f = interp1d(valid_indices, coord_values, kind='linear', 
                               bounds_error=False, fill_value='extrapolate')
                    interpolated_coord = f(np.arange(len(ball_positions)))
                    
            elif method == 'polynomial':
                if len(valid_indices) >= 3:
                    # Fit polynomial
                    coeffs = np.polyfit(valid_indices, coord_values, min(3, len(valid_indices)-1))
                    poly = np.poly1d(coeffs)
                    interpolated_coord = poly(np.arange(len(ball_positions)))
                else:
                    # Fall back to linear
                    from scipy.interpolate import interp1d
                    f = interp1d(valid_indices, coord_values, kind='linear', 
                               bounds_error=False, fill_value='extrapolate')
                    interpolated_coord = f(np.arange(len(ball_positions)))
            
            interpolated_positions.append(interpolated_coord)
        
        # Reconstruct bbox format
        result = []
        for i in range(len(ball_positions)):
            bbox = [interpolated_positions[j][i] for j in range(4)]
            
            # Preserve original confidence if it exists
            original_confidence = ball_positions[i].get(1, {}).get('confidence', None)
            if original_confidence is not None:
                result.append({1: {"bbox": bbox, "confidence": original_confidence}})
            else:
                result.append({1: {"bbox": bbox}})
        
        return result

    def smooth_trajectory(self, ball_positions, window_size=5):
        """
        Apply temporal smoothing to ball trajectory.
        
        Args:
            ball_positions (list): Ball positions across frames.
            window_size (int): Size of smoothing window.
            
        Returns:
            list: Smoothed ball positions.
        """
        if len(ball_positions) < window_size:
            return ball_positions
        
        positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        smoothed_positions = []
        
        for i in range(len(positions)):
            if len(positions[i]) == 0:
                smoothed_positions.append(ball_positions[i])
                continue
            
            # Get window of positions
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(positions), i + window_size // 2 + 1)
            
            window_positions = []
            for j in range(start_idx, end_idx):
                if len(positions[j]) == 4:
                    window_positions.append(positions[j])
            
            if len(window_positions) == 0:
                smoothed_positions.append(ball_positions[i])
                continue
            
            # Calculate weighted average (center frame gets higher weight)
            weights = []
            for j in range(start_idx, end_idx):
                if len(positions[j]) == 4:
                    # Gaussian-like weight centered on current frame
                    distance = abs(j - i)
                    weight = np.exp(-distance**2 / (2 * (window_size//4)**2))
                    weights.append(weight)
                else:
                    weights.append(0)
            
            if sum(weights) == 0:
                smoothed_positions.append(ball_positions[i])
                continue
            
            # Normalize weights
            weights = np.array(weights) / sum(weights)
            
            # Calculate weighted average
            smoothed_bbox = np.zeros(4)
            for j, pos in enumerate(window_positions):
                smoothed_bbox += weights[j] * np.array(pos)
            
            # Preserve original confidence if it exists
            original_confidence = ball_positions[i].get(1, {}).get('confidence', None)
            if original_confidence is not None:
                smoothed_positions.append({1: {"bbox": smoothed_bbox.tolist(), "confidence": original_confidence}})
            else:
                smoothed_positions.append({1: {"bbox": smoothed_bbox.tolist()}})
        
        return smoothed_positions

    def track_ball_with_motion_model(self, frames, read_from_stub=False, stub_path=None):
        """
        Enhanced ball tracking with motion prediction and Kalman filtering.
        
        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.
            
        Returns:
            list: Enhanced ball tracking results with motion prediction.
        """
        # Get initial detections
        tracks = self.get_object_tracks(frames, read_from_stub, stub_path)
        
        # Apply Kalman filtering
        tracks = self.apply_kalman_filtering(tracks)
        
        # Remove outliers
        tracks = self.remove_wrong_detections(tracks)
        
        # Interpolate gaps
        tracks = self.interpolate_ball_positions(tracks)
        
        return tracks

    def get_multiple_ball_candidates(self, frames):
        """
        Get multiple ball candidates per frame for robust tracking.
        
        Args:
            frames (list): List of video frames to process.
            
        Returns:
            list: List of frames with multiple ball candidates.
        """
        detections = self.detect_frames(frames)
        candidates_per_frame = []
        
        for frame_num, detection in enumerate(detections):
            try:
                cls_names = detection.names
                cls_names_inv = {v:k for k,v in cls_names.items()}
                detection_supervision = sv.Detections.from_ultralytics(detection)
                
                frame_candidates = []
                for frame_detection in detection_supervision:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]
                    confidence = frame_detection[2]
                    
                    if cls_id == cls_names_inv.get('Ball', -1):  # Use get() with default
                        frame_candidates.append({
                            'bbox': bbox,
                            'confidence': confidence,
                            'center': [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
                        })
                
                # Sort by confidence and keep top candidates
                frame_candidates.sort(key=lambda x: x['confidence'], reverse=True)
                candidates_per_frame.append(frame_candidates[:3])  # Keep top 3 candidates
                
            except Exception as e:
                # If there's any error processing this frame, add empty candidates
                print(f"Warning: Error processing frame {frame_num}: {e}")
                candidates_per_frame.append([])
        
        return candidates_per_frame

    def select_best_trajectory(self, candidates_per_frame):
        """
        Select the best ball trajectory from multiple candidates.
        
        Args:
            candidates_per_frame (list): List of ball candidates per frame.
            
        Returns:
            list: Best ball trajectory across frames.
        """
        if not candidates_per_frame:
            return []
            
        # Dynamic programming to find best trajectory
        n_frames = len(candidates_per_frame)
        max_candidates = max(len(frame_candidates) for frame_candidates in candidates_per_frame)
        
        if max_candidates == 0:
            return [{} for _ in range(n_frames)]
            
        dp = [[float('inf')] * max_candidates for _ in range(n_frames)]
        prev = [[-1] * max_candidates for _ in range(n_frames)]
        
        # Initialize first frame
        for j, candidate in enumerate(candidates_per_frame[0]):
            dp[0][j] = 0  # No cost for first frame
            
        # Fill DP table
        for i in range(1, n_frames):
            for j, curr_candidate in enumerate(candidates_per_frame[i]):
                for k, prev_candidate in enumerate(candidates_per_frame[i-1]):
                    if dp[i-1][k] == float('inf'):
                        continue
                        
                    # Calculate transition cost
                    distance = np.linalg.norm(
                        np.array(curr_candidate['center']) - np.array(prev_candidate['center'])
                    )
                    confidence_cost = 1 - curr_candidate['confidence']
                    total_cost = dp[i-1][k] + distance + confidence_cost
                    
                    if total_cost < dp[i][j]:
                        dp[i][j] = total_cost
                        prev[i][j] = k
        
        # Backtrack to find best trajectory
        best_trajectory = []
        
        # Find best ending position
        if len(candidates_per_frame[-1]) > 0:
            curr_j = np.argmin(dp[-1])
        else:
            curr_j = -1
        
        for i in range(n_frames-1, -1, -1):
            if curr_j >= 0 and curr_j < len(candidates_per_frame[i]):
                best_trajectory.append({1: {"bbox": candidates_per_frame[i][curr_j]['bbox']}})
            else:
                best_trajectory.append({})
            
            if curr_j >= 0 and i > 0:
                curr_j = prev[i][curr_j]
            else:
                curr_j = -1
            
        return list(reversed(best_trajectory))

    def adaptive_confidence_threshold(self, frame_candidates):
        """
        Adaptively adjust confidence threshold based on detection quality.
        
        Args:
            frame_candidates (list): Ball candidates for current frame.
            
        Returns:
            float: Adaptive confidence threshold.
        """
        if not frame_candidates:
            return self.confidence_threshold
            
        # If we have multiple high-confidence detections, be more selective
        high_conf_candidates = [c for c in frame_candidates if c['confidence'] > 0.7]
        if len(high_conf_candidates) > 1:
            return 0.7
        elif len(high_conf_candidates) == 1:
            return 0.5
        else:
            return 0.3  # Lower threshold if detections are scarce

    def apply_physics_constraints(self, ball_positions, fps=30):
        """
        Apply basketball-specific physics constraints to ball motion.
        
        Args:
            ball_positions (list): Ball positions across frames.
            fps (int): Frames per second of the video.
            
        Returns:
            list: Physics-constrained ball positions.
        """
        if len(ball_positions) < 3:
            return ball_positions
            
        constrained_positions = []
        dt = 1.0 / fps
        g = 9.81  # Gravity (m/s²)
        max_velocity = 15.0  # Maximum ball velocity (m/s)
        
        for i, frame_data in enumerate(ball_positions):
            current_bbox = frame_data.get(1, {}).get('bbox', [])
            
            if len(current_bbox) == 0:
                constrained_positions.append(frame_data)
                continue
                
            # Get current position
            x, y = (current_bbox[0] + current_bbox[2])/2, (current_bbox[1] + current_bbox[3])/2
            
            # Calculate velocity from previous frames
            if i >= 2:
                prev_bbox = ball_positions[i-1].get(1, {}).get('bbox', [])
                prev_prev_bbox = ball_positions[i-2].get(1, {}).get('bbox', [])
                
                if len(prev_bbox) > 0 and len(prev_prev_bbox) > 0:
                    prev_x, prev_y = (prev_bbox[0] + prev_bbox[2])/2, (prev_bbox[1] + prev_bbox[3])/2
                    prev_prev_x, prev_prev_y = (prev_prev_bbox[0] + prev_prev_bbox[2])/2, (prev_prev_bbox[1] + prev_prev_bbox[3])/2
                    
                    # Calculate velocity
                    vx = (prev_x - prev_prev_x) / dt
                    vy = (prev_y - prev_prev_y) / dt
                    
                    # Apply velocity constraints
                    velocity_magnitude = np.sqrt(vx**2 + vy**2)
                    if velocity_magnitude > max_velocity:
                        scale = max_velocity / velocity_magnitude
                        vx *= scale
                        vy *= scale
                    
                    # Predict position with physics
                    predicted_x = prev_x + vx * dt
                    predicted_y = prev_y + vy * dt - 0.5 * g * dt**2  # Add gravity effect
                    
                    # Blend detection with physics prediction
                    alpha = 0.7  # Weight for detection vs physics
                    constrained_x = alpha * x + (1 - alpha) * predicted_x
                    constrained_y = alpha * y + (1 - alpha) * predicted_y
                    
                    # Update bbox
                    bbox_size = current_bbox[2] - current_bbox[0]
                    constrained_bbox = [
                        constrained_x - bbox_size/2,
                        constrained_y - bbox_size/2,
                        constrained_x + bbox_size/2,
                        constrained_y + bbox_size/2
                    ]
                    
                    # Preserve original confidence if it exists
                    original_confidence = frame_data.get(1, {}).get('confidence', None)
                    if original_confidence is not None:
                        constrained_positions.append({1: {"bbox": constrained_bbox, "confidence": original_confidence}})
                    else:
                        constrained_positions.append({1: {"bbox": constrained_bbox}})
                else:
                    constrained_positions.append(frame_data)
            else:
                constrained_positions.append(frame_data)
        
        return constrained_positions

    def detect_bounce_points(self, ball_positions, fps=30):
        """
        Detect bounce points in ball trajectory for better tracking.
        
        Args:
            ball_positions (list): Ball positions across frames.
            fps (int): Frames per second of the video.
            
        Returns:
            list: Bounce point indicators.
        """
        bounce_points = []
        dt = 1.0 / fps
        
        for i in range(2, len(ball_positions) - 2):
            current_bbox = ball_positions[i].get(1, {}).get('bbox', [])
            prev_bbox = ball_positions[i-1].get(1, {}).get('bbox', [])
            next_bbox = ball_positions[i+1].get(1, {}).get('bbox', [])
            
            if len(current_bbox) == 0 or len(prev_bbox) == 0 or len(next_bbox) == 0:
                bounce_points.append(False)
                continue
                
            # Calculate vertical velocity
            current_y = (current_bbox[1] + current_bbox[3]) / 2
            prev_y = (prev_bbox[1] + prev_bbox[3]) / 2
            next_y = (next_bbox[1] + next_bbox[3]) / 2
            
            vy_prev = (current_y - prev_y) / dt
            vy_next = (next_y - current_y) / dt
            
            # Detect bounce: velocity changes sign and magnitude
            if vy_prev < 0 and vy_next > 0 and abs(vy_next) > 0.5 * abs(vy_prev):
                bounce_points.append(True)
            else:
                bounce_points.append(False)
        
        # Pad with False for first and last frames
        return [False, False] + bounce_points + [False, False]

    def enhanced_tracking_pipeline(self, frames, read_from_stub=False, stub_path=None):
        """
        Complete enhanced ball tracking pipeline with guaranteed ball visibility.
        
        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.
            
        Returns:
            list: Enhanced ball tracking results with guaranteed visibility.
        """
        try:
            # Validate model first
            if not self._validate_model():
                print("Warning: Ball detection model validation failed, using basic tracking")
                return self.get_object_tracks(frames, read_from_stub, stub_path)
            
            # Use adaptive detection for better initial results
            print("Running adaptive ball detection...")
            adaptive_detections = self.adaptive_ball_detection(frames, initial_confidence=0.4)
            
            # Convert detections to tracks format
            tracks = []
            for detection in adaptive_detections:
                if detection is None:
                    tracks.append({})
                else:
                    cls_names = detection.names
                    cls_names_inv = {v: k for k, v in cls_names.items()}
                    detection_supervision = sv.Detections.from_ultralytics(detection)
                    
                    chosen_bbox = None
                    max_confidence = 0
                    
                    for frame_detection in detection_supervision:
                        bbox = frame_detection[0].tolist()
                        cls_id = frame_detection[3]
                        confidence = frame_detection[2]
                        
                        if cls_id == cls_names_inv.get('Ball', -1):
                            if max_confidence < confidence:
                                chosen_bbox = bbox
                                max_confidence = confidence
                    
                    if chosen_bbox is not None:
                        tracks.append({1: {"bbox": chosen_bbox}})
                    else:
                        tracks.append({})
            
            # Get multiple candidates for better trajectory selection
            print("Getting multiple ball candidates...")
            candidates_per_frame = self.get_multiple_ball_candidates(frames)
            
            # Select best trajectory
            print("Selecting best trajectory...")
            tracks = self.select_best_trajectory(candidates_per_frame)
            
            # Apply Kalman filtering for smooth prediction
            print("Applying Kalman filtering...")
            tracks = self.apply_kalman_filtering(tracks)
            
            # Apply physics constraints
            print("Applying physics constraints...")
            tracks = self.apply_physics_constraints(tracks)
            
            # Remove outliers
            print("Removing outliers...")
            tracks = self.remove_wrong_detections(tracks)
            
            # Advanced interpolation
            print("Applying advanced interpolation...")
            tracks = self.advanced_interpolation(tracks, method='cubic')
            
            # Smooth trajectory
            print("Smoothing trajectory...")
            tracks = self.smooth_trajectory(tracks, window_size=5)
            
            # Ensure ball is always visible (most important step)
            print("Ensuring ball visibility...")
            tracks = self.ensure_ball_visibility(tracks, frames)
            
            # Final validation - make sure every frame has a ball position
            final_tracks = []
            last_valid_bbox = None
            last_valid_confidence = None
            
            for track in tracks:
                bbox = track.get(1, {}).get('bbox', [])
                confidence = track.get(1, {}).get('confidence', 1.0)
                
                if len(bbox) == 4:
                    final_tracks.append(track)
                    last_valid_bbox = bbox
                    last_valid_confidence = confidence
                else:
                    # Use last valid position as fallback
                    if last_valid_bbox is not None:
                        final_tracks.append({1: {"bbox": last_valid_bbox, "final_fallback": True, "confidence": last_valid_confidence * 0.5}})
                    else:
                        # Ultimate fallback - center of frame
                        h, w = frames[0].shape[:2]
                        center_bbox = [w//2 - 10, h//2 - 10, w//2 + 10, h//2 + 10]
                        final_tracks.append({1: {"bbox": center_bbox, "ultimate_fallback": True, "confidence": 0.1}})
            
            print(f"Enhanced tracking completed. Ball visible in {sum(1 for t in final_tracks if len(t.get(1, {}).get('bbox', [])) == 4)}/{len(final_tracks)} frames")
            
            return final_tracks
            
        except Exception as e:
            print(f"Warning: Enhanced tracking failed, falling back to basic tracking: {e}")
            # Fallback to basic tracking with visibility guarantee
            tracks = self.get_object_tracks(frames, read_from_stub, stub_path)
            tracks = self.remove_wrong_detections(tracks)
            tracks = self.interpolate_ball_positions(tracks)
            tracks = self.ensure_ball_visibility(tracks, frames)
            return tracks

    def _validate_model(self):
        """
        Validate that the ball detection model is working correctly.
        
        Returns:
            bool: True if model is valid, False otherwise.
        """
        try:
            # Test with a dummy frame to check model output
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            result = self.model.predict(dummy_frame, conf=0.1, verbose=False)
            
            if len(result) == 0:
                return False
                
            # Check if 'Ball' class exists in model
            cls_names = result[0].names
            if 'Ball' not in cls_names:
                print(f"Warning: 'Ball' class not found in model. Available classes: {list(cls_names.keys())}")
                return False
                
            return True
            
        except Exception as e:
            print(f"Model validation failed: {e}")
            return False

    def test_tracking(self, test_frames=None):
        """
        Test the ball tracking functionality with a simple test.
        
        Args:
            test_frames (list): Optional test frames. If None, creates dummy frames.
            
        Returns:
            bool: True if tracking works, False otherwise.
        """
        try:
            if test_frames is None:
                # Create dummy test frames
                test_frames = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(5)]
            
            # Test basic tracking
            tracks = self.get_object_tracks(test_frames)
            if len(tracks) != len(test_frames):
                print(f"Warning: Track length mismatch. Expected {len(test_frames)}, got {len(tracks)}")
                return False
            
            # Test enhanced tracking
            enhanced_tracks = self.enhanced_tracking_pipeline(test_frames)
            if len(enhanced_tracks) != len(test_frames):
                print(f"Warning: Enhanced track length mismatch. Expected {len(test_frames)}, got {len(enhanced_tracks)}")
                return False
            
            print("Ball tracking test passed!")
            return True
            
        except Exception as e:
            print(f"Ball tracking test failed: {e}")
            return False

    def predict_ball_position(self, ball_positions, frame_idx, search_radius=50):
        """
        Predict ball position based on previous frames and search in that area.
        
        Args:
            ball_positions (list): Ball positions across frames
            frame_idx (int): Current frame index
            search_radius (int): Radius to search around predicted position
            
        Returns:
            tuple: (predicted_bbox, confidence) or (None, 0.0)
        """
        if frame_idx < 2:
            return None, 0.0
            
        # Get last 3 valid positions
        recent_positions = []
        for i in range(max(0, frame_idx-3), frame_idx):
            bbox = ball_positions[i].get(1, {}).get('bbox', [])
            if len(bbox) == 4:
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                recent_positions.append((center_x, center_y))
        
        if len(recent_positions) < 2:
            return None, 0.0
        
        # Calculate velocity and predict next position
        if len(recent_positions) >= 2:
            # Simple linear prediction
            dx = recent_positions[-1][0] - recent_positions[-2][0]
            dy = recent_positions[-1][1] - recent_positions[-2][1]
            
            predicted_x = recent_positions[-1][0] + dx
            predicted_y = recent_positions[-1][1] + dy
            
            # Create predicted bbox
            bbox_size = 20  # Default ball size
            predicted_bbox = [
                predicted_x - bbox_size//2,
                predicted_y - bbox_size//2,
                predicted_x + bbox_size//2,
                predicted_y + bbox_size//2
            ]
            
            return predicted_bbox, 0.8  # Medium confidence for prediction
        
        return None, 0.0

    def search_ball_in_region(self, frame, center_x, center_y, search_radius=50, confidence_threshold=0.3):
        """
        Search for ball in a specific region with lower confidence threshold.
        
        Args:
            frame (numpy.ndarray): Video frame
            center_x (float): Center X coordinate to search around
            center_y (float): Center Y coordinate to search around
            search_radius (int): Radius to search around center
            confidence_threshold (float): Lower confidence threshold for search
            
        Returns:
            tuple: (bbox, confidence) or (None, 0.0)
        """
        try:
            # Crop frame to search region
            x1 = max(0, int(center_x - search_radius))
            y1 = max(0, int(center_y - search_radius))
            x2 = min(frame.shape[1], int(center_x + search_radius))
            y2 = min(frame.shape[0], int(center_y + search_radius))
            
            if x1 >= x2 or y1 >= y2:
                return None, 0.0
            
            # Crop frame
            cropped_frame = frame[y1:y2, x1:x2]
            
            # Detect ball in cropped region with lower confidence
            result = self.model.predict(cropped_frame, conf=confidence_threshold, verbose=False)
            
            if len(result) == 0:
                return None, 0.0
            
            # Get best ball detection
            cls_names = result[0].names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            detection_supervision = sv.Detections.from_ultralytics(result[0])
            
            best_bbox = None
            best_confidence = 0.0
            
            for detection in detection_supervision:
                bbox = detection[0].tolist()
                cls_id = detection[3]
                confidence = detection[2]
                
                if cls_id == cls_names_inv.get('Ball', -1) and confidence > best_confidence:
                    # Adjust bbox coordinates back to original frame
                    adjusted_bbox = [
                        bbox[0] + x1,
                        bbox[1] + y1,
                        bbox[2] + x1,
                        bbox[3] + y1
                    ]
                    best_bbox = adjusted_bbox
                    best_confidence = confidence
            
            return best_bbox, best_confidence
            
        except Exception as e:
            print(f"Error searching for ball in region: {e}")
            return None, 0.0

    def ensure_ball_visibility(self, ball_positions, frames):
        """
        Ensure ball is always visible by using prediction and fallback mechanisms.
        
        Args:
            ball_positions (list): Ball positions across frames
            frames (list): Video frames
            
        Returns:
            list: Ball positions with guaranteed visibility
        """
        guaranteed_positions = []
        last_valid_position = None
        
        for frame_idx, (frame_data, frame) in enumerate(zip(ball_positions, frames)):
            current_bbox = frame_data.get(1, {}).get('bbox', [])
            
            if len(current_bbox) == 4:
                # Valid detection found - preserve original data (don't add flags)
                # This ensures confidence can be displayed for actual detections
                guaranteed_positions.append(frame_data)
                last_valid_position = current_bbox
            else:
                # No detection - try prediction and search
                predicted_bbox, pred_confidence = self.predict_ball_position(ball_positions, frame_idx)
                
                if predicted_bbox is not None:
                    # Search around predicted position
                    center_x = (predicted_bbox[0] + predicted_bbox[2]) / 2
                    center_y = (predicted_bbox[1] + predicted_bbox[3]) / 2
                    
                    found_bbox, found_confidence = self.search_ball_in_region(
                        frame, center_x, center_y, search_radius=80, confidence_threshold=0.2
                    )
                    
                    if found_bbox is not None and found_confidence > 0.3:
                        # Found ball in predicted region - this is an actual detection
                        guaranteed_positions.append({1: {"bbox": found_bbox, "confidence": found_confidence}})
                        last_valid_position = found_bbox
                    else:
                        # Use predicted position
                        guaranteed_positions.append({1: {"bbox": predicted_bbox, "predicted": True, "confidence": pred_confidence}})
                        last_valid_position = predicted_bbox
                else:
                    # No prediction possible - use last valid position
                    if last_valid_position is not None:
                        guaranteed_positions.append({1: {"bbox": last_valid_position, "fallback": True, "confidence": 0.5}})
                    else:
                        # No previous position - use center of frame as fallback
                        h, w = frame.shape[:2]
                        center_x, center_y = w // 2, h // 2
                        fallback_bbox = [center_x - 10, center_y - 10, center_x + 10, center_y + 10]
                        guaranteed_positions.append({1: {"bbox": fallback_bbox, "ultimate_fallback": True, "confidence": 0.1}})
        
        return guaranteed_positions

    def adaptive_ball_detection(self, frames, initial_confidence=0.5):
        """
        Adaptive ball detection that adjusts confidence threshold based on detection success.
        
        Args:
            frames (list): Video frames to process
            initial_confidence (float): Initial confidence threshold
            
        Returns:
            list: Ball detection results
        """
        detections = []
        current_confidence = initial_confidence
        consecutive_failures = 0
        consecutive_successes = 0
        
        for frame_idx, frame in enumerate(frames):
            # Try detection with current confidence
            result = self.model.predict(frame, conf=current_confidence, verbose=False)
            
            if len(result) == 0:
                consecutive_failures += 1
                consecutive_successes = 0
                
                # Lower confidence if too many failures
                if consecutive_failures >= 3:
                    current_confidence = max(0.1, current_confidence * 0.8)
                    consecutive_failures = 0
                    
                # Try again with lower confidence
                result = self.model.predict(frame, conf=current_confidence, verbose=False)
            else:
                consecutive_successes += 1
                consecutive_failures = 0
                
                # Increase confidence if successful
                if consecutive_successes >= 5:
                    current_confidence = min(0.9, current_confidence * 1.1)
                    consecutive_successes = 0
            
            detections.append(result[0] if len(result) > 0 else None)
        
        return detections