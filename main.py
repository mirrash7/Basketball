import os
import argparse
import gc  # Add garbage collection
import psutil  # Add memory monitoring
from utils import read_video, save_video, read_video_memory_efficient
from trackers import PlayerTracker, BallTracker
from team_assigner import TeamAssigner
from team_assigner.team_assigner_Orig import TeamAssigner as OriginalTeamAssigner
from court_keypoint_detector import CourtKeypointDetector
from ball_aquisition import BallAquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from tactical_view_converter import TacticalViewConverter
from speed_and_distance_calculator import SpeedAndDistanceCalculator
from drawers import (
    PlayerTracksDrawer, 
    BallTracksDrawer,
    CourtKeypointDrawer,
    TeamBallControlDrawer,
    FrameNumberDrawer,
    PassInterceptionDrawer,
    TacticalViewDrawer,
    SpeedAndDistanceDrawer
)
from configs import(
    STUBS_DEFAULT_PATH,
    PLAYER_DETECTOR_PATH,
    BALL_DETECTOR_PATH,
    COURT_KEYPOINT_DETECTOR_PATH,
    OUTPUT_VIDEO_PATH
)
from configs.overlay_config import get_preset_config, list_available_presets

def parse_args():
    parser = argparse.ArgumentParser(description='Basketball Video Analysis')
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('--output_video', type=str, default=OUTPUT_VIDEO_PATH, 
                        help='Path to output video file')
    parser.add_argument('--stub_path', type=str, default=STUBS_DEFAULT_PATH,
                        help='Path to stub directory')
    
    # Preset configuration
    parser.add_argument('--preset', type=str, choices=list_available_presets(),
                        help='Use a preset configuration for overlays')
    
    # Overlay control arguments
    parser.add_argument('--show_player_tracks', action='store_true', default=True,
                        help='Show player bounding boxes and tracking')
    parser.add_argument('--no_player_tracks', dest='show_player_tracks', action='store_false',
                        help='Hide player bounding boxes and tracking')
    
    parser.add_argument('--show_ball_tracks', action='store_true', default=True,
                        help='Show ball trajectory')
    parser.add_argument('--no_ball_tracks', dest='show_ball_tracks', action='store_false',
                        help='Hide ball trajectory')
    
    parser.add_argument('--show_court_keypoints', action='store_true', default=True,
                        help='Show court keypoints and lines')
    parser.add_argument('--no_court_keypoints', dest='show_court_keypoints', action='store_false',
                        help='Hide court keypoints and lines')
    
    parser.add_argument('--show_frame_number', action='store_true', default=True,
                        help='Show frame number overlay')
    parser.add_argument('--no_frame_number', dest='show_frame_number', action='store_false',
                        help='Hide frame number overlay')
    
    parser.add_argument('--show_team_ball_control', action='store_true', default=True,
                        help='Show team ball possession indicators')
    parser.add_argument('--no_team_ball_control', dest='show_team_ball_control', action='store_false',
                        help='Hide team ball possession indicators')
    
    parser.add_argument('--show_passes_interceptions', action='store_true', default=True,
                        help='Show pass and interception arrows')
    parser.add_argument('--no_passes_interceptions', dest='show_passes_interceptions', action='store_false',
                        help='Hide pass and interception arrows')
    
    parser.add_argument('--show_speed_distance', action='store_true', default=True,
                        help='Show player speed and distance metrics')
    parser.add_argument('--no_speed_distance', dest='show_speed_distance', action='store_false',
                        help='Hide player speed and distance metrics')
    
    parser.add_argument('--show_tactical_view', action='store_true', default=True,
                        help='Show tactical overhead view')
    parser.add_argument('--no_tactical_view', dest='show_tactical_view', action='store_false',
                        help='Hide tactical overhead view')
    
    # Team assignment control
    parser.add_argument('--enable_team_assignment', action='store_true', default=True,
                        help='Enable team assignment for players')
    parser.add_argument('--disable_team_assignment', dest='enable_team_assignment', action='store_false',
                        help='Disable team assignment for players')
    
    # Team assignment configuration
    parser.add_argument('--team_assigner_version', type=str, choices=['current', 'original'], default='current',
                        help='Choose team assigner version: "current" (improved) or "original" (basic)')
    parser.add_argument('--max_players_per_team', type=int, default=5,
                        help='Maximum players allowed per team (default: 5)')
    parser.add_argument('--team_confidence_threshold', type=float, default=0.6,
                        help='Minimum confidence threshold for team assignment (default: 0.6)')
    parser.add_argument('--team_1_description', type=str, default="white basketball jersey",
                        help='Description of Team 1 jersey appearance (default: "white basketball jersey")')
    parser.add_argument('--team_2_description', type=str, default="dark blue basketball jersey",
                        help='Description of Team 2 jersey appearance (default: "dark blue basketball jersey")')
    
    # Memory management options
    parser.add_argument('--memory_efficient', action='store_true', default=False,
                        help='Enable memory-efficient processing (resize frames and process in batches)')
    parser.add_argument('--resize_factor', type=float, default=0.5,
                        help='Frame resize factor for memory efficiency (0.5 = half size, 1.0 = original)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum number of frames to process (for testing)')
    parser.add_argument('--ultra_memory_efficient', action='store_true', default=False,
                        help='Enable ultra memory-efficient processing (process in very small batches)')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for memory-efficient processing (default: 10)')
    
    return parser.parse_args()

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def process_frames_in_batches(frames, batch_size, processor_func, *args, **kwargs):
    """
    Process frames in batches to reduce memory usage.
    
    Args:
        frames (list): List of frames to process
        batch_size (int): Number of frames to process at once
        processor_func (function): Function to apply to each batch
        *args, **kwargs: Additional arguments for processor_func
        
    Returns:
        list: Processed frames
    """
    processed_frames = []
    
    for i in range(0, len(frames), batch_size):
        batch_end = min(i + batch_size, len(frames))
        batch_frames = frames[i:batch_end]
        
        # Process this batch
        batch_result = processor_func(batch_frames, *args, **kwargs)
        processed_frames.extend(batch_result)
        
        # Force garbage collection
        gc.collect()
        
        # Print progress
        if i % (batch_size * 5) == 0:
            print(f"Processed {i}/{len(frames)} frames. Memory: {get_memory_usage():.1f} MB")
    
    return processed_frames

def main():
    args = parse_args()
    
    # Apply preset configuration if specified
    if args.preset:
        preset_config = get_preset_config(args.preset)
        print(f"Applying preset: {args.preset}")
        for key, value in preset_config.items():
            setattr(args, key, value)
    
    # Debug: Print overlay configuration
    print("Overlay Configuration:")
    overlay_flags = [
        'show_player_tracks', 'show_ball_tracks', 'show_court_keypoints',
        'show_frame_number', 'show_team_ball_control', 'show_passes_interceptions',
        'show_speed_distance', 'show_tactical_view'
    ]
    for flag in overlay_flags:
        status = "✅" if getattr(args, flag) else "❌"
        print(f"  {status} {flag}: {getattr(args, flag)}")
    
    # Print team assignment status
    team_status = "✅" if args.enable_team_assignment else "❌"
    print(f"  {team_status} enable_team_assignment: {args.enable_team_assignment}")
    
    # Ensure at least one overlay is enabled
    any_enabled = any(getattr(args, flag) for flag in overlay_flags)
    if not any_enabled:
        print("Warning: No overlays enabled, enabling player tracks by default")
        args.show_player_tracks = True
    
    # Read Video
    if args.memory_efficient:
        print(f"Using memory-efficient processing with resize factor: {args.resize_factor}")
        video_frames, fps = read_video_memory_efficient(args.input_video, 
                                                       max_frames=args.max_frames,
                                                       resize_factor=args.resize_factor)
    else:
        video_frames, fps = read_video(args.input_video)
        if args.max_frames:
            video_frames = video_frames[:args.max_frames]
    
    print(f"Loaded {len(video_frames)} frames. Memory usage: {get_memory_usage():.1f} MB")
    
    # Set batch size for memory-efficient processing
    if args.ultra_memory_efficient:
        batch_size = min(args.batch_size, 5)  # Very small batches for ultra memory efficiency
        print(f"Using ultra memory-efficient processing with batch size: {batch_size}")
    elif args.memory_efficient:
        batch_size = args.batch_size
        print(f"Using memory-efficient processing with batch size: {batch_size}")
    else:
        batch_size = None  # Process all frames at once
    
    ## Initialize Tracker
    player_tracker = PlayerTracker(PLAYER_DETECTOR_PATH)
    ball_tracker = BallTracker(BALL_DETECTOR_PATH)

    ## Initialize Keypoint Detector
    court_keypoint_detector = CourtKeypointDetector(COURT_KEYPOINT_DETECTOR_PATH)

    # Run Detectors
    player_tracks = player_tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path=os.path.join(args.stub_path, 'player_track_stubs.pkl')
                                      )
    
    ## Run KeyPoint Extractor
    court_keypoints_per_frame = court_keypoint_detector.get_court_keypoints(video_frames,
                                                                    read_from_stub=True,
                                                                    stub_path=os.path.join(args.stub_path, 'court_key_points_stub.pkl')
                                                                    )

    # Enhanced Ball Tracking Pipeline
    print("Running enhanced ball tracking...")
    try:
        ball_tracks = ball_tracker.enhanced_tracking_pipeline(video_frames,
                                                             read_from_stub=True,
                                                             stub_path=os.path.join(args.stub_path, 'ball_track_stubs.pkl')
                                                            )
    except Exception as e:
        print(f"Enhanced ball tracking failed: {e}")
        print("Falling back to basic ball tracking...")
        ball_tracks = ball_tracker.get_object_tracks(video_frames,
                                                     read_from_stub=True,
                                                     stub_path=os.path.join(args.stub_path, 'ball_track_stubs.pkl')
                                                    )
        # Remove Wrong Ball Detections
        ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
        # Interpolate Ball Tracks
        ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    # Assign Player Teams
    if args.enable_team_assignment:
        print("Running team assignment...")
        print(f"Team assigner version: {args.team_assigner_version}")
        print(f"Team assignment configuration:")
        print(f"  Max players per team: {args.max_players_per_team}")
        print(f"  Confidence threshold: {args.team_confidence_threshold}")
        print(f"  Team 1 description: '{args.team_1_description}'")
        print(f"  Team 2 description: '{args.team_2_description}'")
        
        if args.team_assigner_version == 'original':
            # Use original team assigner (simpler, no max players limit)
            print("Using original team assigner (basic version)")
            team_assigner = OriginalTeamAssigner(
                team_1_class_name=args.team_1_description,
                team_2_class_name=args.team_2_description
            )
        else:
            # Use current team assigner (improved version with constraints)
            print("Using current team assigner (improved version)")
            team_assigner = TeamAssigner(
                team_1_class_name=args.team_1_description,
                team_2_class_name=args.team_2_description,
                max_players_per_team=args.max_players_per_team,
                confidence_threshold=args.team_confidence_threshold
            )
        
        player_assignment = team_assigner.get_player_teams_across_frames(video_frames,
                                                                        player_tracks,
                                                                        read_from_stub=True,
                                                                        stub_path=os.path.join(args.stub_path, 'player_assignment_stub.pkl')
                                                                        )
    else:
        print("Team assignment disabled, using default team assignments...")
        # Create default team assignments (all players assigned to team 1)
        player_assignment = []
        for frame_tracks in player_tracks:
            frame_assignment = {}
            for track_id in frame_tracks.keys():
                # Only assign teams to actual players, not refs or hoops
                if frame_tracks[track_id].get("class", "Player") == "Player":
                    frame_assignment[track_id] = 1  # Default to team 1
            player_assignment.append(frame_assignment)

    # Ball Acquisition
    ball_aquisition_detector = BallAquisitionDetector()
    ball_aquisition = ball_aquisition_detector.detect_ball_possession(player_tracks,ball_tracks)

    # Detect Passes
    pass_and_interception_detector = PassAndInterceptionDetector()
    passes = pass_and_interception_detector.detect_passes(ball_aquisition,player_assignment)
    interceptions = pass_and_interception_detector.detect_interceptions(ball_aquisition,player_assignment)

    # Tactical View
    tactical_view_converter = TacticalViewConverter(
        court_image_path="./images/basketball_court.png"
    )

    court_keypoints_per_frame = tactical_view_converter.validate_keypoints(court_keypoints_per_frame)
    tactical_player_positions = tactical_view_converter.transform_players_to_tactical_view(court_keypoints_per_frame,player_tracks)

    # Speed and Distance Calculator
    speed_and_distance_calculator = SpeedAndDistanceCalculator(
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.actual_width_in_meters,
        tactical_view_converter.actual_height_in_meters
    )
    player_distances_per_frame = speed_and_distance_calculator.calculate_distance(tactical_player_positions)
    player_speed_per_frame = speed_and_distance_calculator.calculate_speed(player_distances_per_frame)

    # Draw output   
    # Initialize Drawers
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    court_keypoint_drawer = CourtKeypointDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()
    frame_number_drawer = FrameNumberDrawer()
    pass_and_interceptions_drawer = PassInterceptionDrawer()
    tactical_view_drawer = TacticalViewDrawer()
    speed_and_distance_drawer = SpeedAndDistanceDrawer()

    # Start with original frames
    output_video_frames = video_frames
    print(f"Initial frames: {len(output_video_frames)}")

    ## Draw object Tracks (conditional)
    if args.show_player_tracks:
        print("Drawing player tracks...")
        output_video_frames = player_tracks_drawer.draw(output_video_frames, 
                                                        player_tracks,
                                                        player_assignment,
                                                        ball_aquisition)
        print(f"After player tracks: {len(output_video_frames)}. Memory: {get_memory_usage():.1f} MB")
        gc.collect()  # Force garbage collection
    
    if args.show_ball_tracks:
        print("Drawing ball tracks...")
        output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)
        print(f"After ball tracks: {len(output_video_frames)}. Memory: {get_memory_usage():.1f} MB")
        gc.collect()  # Force garbage collection

    ## Draw KeyPoints (conditional)
    if args.show_court_keypoints:
        print("Drawing court keypoints...")
        output_video_frames = court_keypoint_drawer.draw(output_video_frames, court_keypoints_per_frame)
        print(f"After court keypoints: {len(output_video_frames)}. Memory: {get_memory_usage():.1f} MB")
        gc.collect()  # Force garbage collection

    ## Draw Frame Number (conditional)
    if args.show_frame_number:
        print("Drawing frame numbers...")
        output_video_frames = frame_number_drawer.draw(output_video_frames)
        print(f"After frame numbers: {len(output_video_frames)}. Memory: {get_memory_usage():.1f} MB")
        gc.collect()  # Force garbage collection

    # Draw Team Ball Control (conditional)
    if args.show_team_ball_control:
        print("Drawing team ball control...")
        output_video_frames = team_ball_control_drawer.draw(output_video_frames,
                                                            player_assignment,
                                                            ball_aquisition)
        print(f"After team ball control: {len(output_video_frames)}. Memory: {get_memory_usage():.1f} MB")
        gc.collect()  # Force garbage collection

    # Draw Passes and Interceptions (conditional)
    if args.show_passes_interceptions:
        print("Drawing passes and interceptions...")
        output_video_frames = pass_and_interceptions_drawer.draw(output_video_frames,
                                                                 passes,
                                                                 interceptions)
        print(f"After passes/interceptions: {len(output_video_frames)}. Memory: {get_memory_usage():.1f} MB")
        gc.collect()  # Force garbage collection
    
    # Speed and Distance Drawer (conditional)
    if args.show_speed_distance:
        print("Drawing speed and distance...")
        output_video_frames = speed_and_distance_drawer.draw(output_video_frames,
                                                             player_tracks,
                                                             player_distances_per_frame,
                                                             player_speed_per_frame
                                                             )
        print(f"After speed/distance: {len(output_video_frames)}. Memory: {get_memory_usage():.1f} MB")
        gc.collect()  # Force garbage collection

    ## Draw Tactical View (conditional)
    if args.show_tactical_view:
        print("Drawing tactical view...")
        
        if batch_size is not None:
            # Use batch processing for memory efficiency
            def tactical_view_processor(batch_frames):
                return tactical_view_drawer.draw(batch_frames,
                                                tactical_view_converter.court_image_path,
                                                tactical_view_converter.width,
                                                tactical_view_converter.height,
                                                tactical_view_converter.key_points,
                                                tactical_player_positions,
                                                player_assignment,
                                                ball_aquisition)
            
            output_video_frames = process_frames_in_batches(
                output_video_frames, 
                batch_size, 
                tactical_view_processor
            )
        else:
            # Process all frames at once
            output_video_frames = tactical_view_drawer.draw(output_video_frames,
                                                            tactical_view_converter.court_image_path,
                                                            tactical_view_converter.width,
                                                            tactical_view_converter.height,
                                                            tactical_view_converter.key_points,
                                                            tactical_player_positions,
                                                            player_assignment,
                                                            ball_aquisition)
        
        print(f"After tactical view: {len(output_video_frames)}. Memory: {get_memory_usage():.1f} MB")
        gc.collect()  # Force garbage collection

    print(f"Final frame count: {len(output_video_frames)}")
    
    # Save video
    # Ensure we have frames to save
    if not output_video_frames or len(output_video_frames) == 0:
        print("Warning: No overlays enabled, using original frames")
        output_video_frames = video_frames
    
    save_video(output_video_frames, args.output_video, fps)

if __name__ == '__main__':
    main()
    