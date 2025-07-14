"""
Overlay configuration presets for basketball video analysis.

This module provides preset configurations for different types of analysis,
allowing easy switching between different visualization modes.
"""

# Default configuration - show everything
DEFAULT_CONFIG = {
    'show_player_tracks': True,
    'show_ball_tracks': True,
    'show_court_keypoints': True,
    'show_frame_number': True,
    'show_team_ball_control': True,
    'show_passes_interceptions': True,
    'show_speed_distance': True,
    'show_tactical_view': True,
}

# Minimal configuration - basic tracking only
MINIMAL_CONFIG = {
    'show_player_tracks': True,
    'show_ball_tracks': True,
    'show_court_keypoints': False,
    'show_frame_number': False,
    'show_team_ball_control': False,
    'show_passes_interceptions': False,
    'show_speed_distance': False,
    'show_tactical_view': False,
}

# Coaching configuration - focus on tactical analysis
COACHING_CONFIG = {
    'show_player_tracks': True,
    'show_ball_tracks': True,
    'show_court_keypoints': True,
    'show_frame_number': False,
    'show_team_ball_control': True,
    'show_passes_interceptions': True,
    'show_speed_distance': False,
    'show_tactical_view': True,
}

# Performance analysis configuration
PERFORMANCE_CONFIG = {
    'show_player_tracks': True,
    'show_ball_tracks': False,
    'show_court_keypoints': False,
    'show_frame_number': True,
    'show_team_ball_control': False,
    'show_passes_interceptions': False,
    'show_speed_distance': True,
    'show_tactical_view': True,
}

# Broadcast configuration - clean, professional look
BROADCAST_CONFIG = {
    'show_player_tracks': False,
    'show_ball_tracks': True,
    'show_court_keypoints': True,
    'show_frame_number': False,
    'show_team_ball_control': True,
    'show_passes_interceptions': True,
    'show_speed_distance': False,
    'show_tactical_view': False,
}

# Debug configuration - show everything for development
DEBUG_CONFIG = {
    'show_player_tracks': True,
    'show_ball_tracks': True,
    'show_court_keypoints': True,
    'show_frame_number': True,
    'show_team_ball_control': True,
    'show_passes_interceptions': True,
    'show_speed_distance': True,
    'show_tactical_view': True,
}

# Available presets
PRESETS = {
    'default': DEFAULT_CONFIG,
    'minimal': MINIMAL_CONFIG,
    'coaching': COACHING_CONFIG,
    'performance': PERFORMANCE_CONFIG,
    'broadcast': BROADCAST_CONFIG,
    'debug': DEBUG_CONFIG,
}

def get_preset_config(preset_name):
    """
    Get a preset configuration by name.
    
    Args:
        preset_name (str): Name of the preset ('default', 'minimal', 'coaching', etc.)
    
    Returns:
        dict: Configuration dictionary for the specified preset
    """
    return PRESETS.get(preset_name.lower(), DEFAULT_CONFIG)

def list_available_presets():
    """
    List all available preset configurations.
    
    Returns:
        list: List of preset names
    """
    return list(PRESETS.keys()) 