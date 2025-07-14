"""
Configuration parameters for enhanced ball tracking.
"""

# Kalman Filter Parameters
KALMAN_CONFIG = {
    'position_noise': 10,      # Process noise for position
    'velocity_noise': 50,      # Process noise for velocity
    'measurement_noise': 20,   # Measurement noise
    'initial_uncertainty': 100 # Initial state uncertainty
}

# Physics Constraints
PHYSICS_CONFIG = {
    'gravity': 9.81,           # Gravity (m/s²)
    'max_velocity': 15.0,      # Maximum ball velocity (m/s)
    'detection_weight': 0.7,   # Weight for detection vs physics prediction
    'bounce_velocity_threshold': 0.5  # Velocity ratio threshold for bounce detection
}

# Tracking Parameters
TRACKING_CONFIG = {
    'confidence_threshold': 0.5,    # Base confidence threshold
    'max_candidates_per_frame': 3,  # Maximum ball candidates per frame
    'max_allowed_distance': 25,     # Maximum allowed movement between frames
    'min_track_length': 5,          # Minimum track length to consider valid
    'interpolation_method': 'linear' # Interpolation method: 'linear', 'cubic', 'spline'
}

# Multi-Hypothesis Tracking
MHT_CONFIG = {
    'max_hypotheses': 10,      # Maximum number of tracking hypotheses
    'pruning_threshold': 0.1,  # Threshold for pruning unlikely hypotheses
    'association_threshold': 50, # Distance threshold for association
    'track_termination_frames': 10  # Frames without detection before terminating track
}

# Adaptive Parameters
ADAPTIVE_CONFIG = {
    'enable_adaptive_threshold': True,  # Enable adaptive confidence threshold
    'enable_physics_constraints': True, # Enable physics-based constraints
    'enable_bounce_detection': True,    # Enable bounce point detection
    'enable_kalman_filtering': True,    # Enable Kalman filtering
    'enable_multi_hypothesis': True     # Enable multi-hypothesis tracking
}

# Performance Tuning
PERFORMANCE_CONFIG = {
    'batch_size': 20,          # Batch size for YOLO inference
    'enable_caching': True,    # Enable result caching
    'cache_expiry_frames': 1000, # Cache expiry in frames
    'parallel_processing': False # Enable parallel processing (experimental)
} 