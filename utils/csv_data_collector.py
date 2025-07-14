import csv
import os
import numpy as np
from typing import List, Dict, Any, Optional

class CSVDataCollector:
    """
    Collects and exports tracking data to CSV format.
    Tracks court keypoints, ball positions, and player positions for each frame.
    """
    
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.data = []
        
    def collect_frame_data(self, 
                          frame_idx: int,
                          court_keypoints: Optional[Any],
                          ball_tracks: Dict[int, Dict[str, Any]],
                          player_tracks: Dict[int, Dict[str, Any]],
                          player_assignment: Optional[Dict[int, int]] = None):
        """
        Collect data for a single frame.
        
        Args:
            frame_idx: Frame index
            court_keypoints: Court keypoints object (YOLO format)
            ball_tracks: Dictionary of ball tracking data for this frame
            player_tracks: Dictionary of player tracking data for this frame
            player_assignment: Dictionary mapping player IDs to team assignments
        """
        frame_data = {
            'frame': frame_idx,
            'court_keypoints': [],
            'ball_data': [],
            'player_data': []
        }
        
        # Collect court keypoints
        if court_keypoints is not None and hasattr(court_keypoints, 'xy') and court_keypoints.xy is not None:
            keypoints_xy = court_keypoints.xy
            if hasattr(keypoints_xy, 'cpu'):
                keypoints_xy = keypoints_xy.cpu().numpy()
            
            for kp_idx, (x, y) in enumerate(keypoints_xy[0]):
                if x > 0 and y > 0:  # Only include detected keypoints
                    frame_data['court_keypoints'].append({
                        'keypoint_id': kp_idx,
                        'x': float(x),
                        'y': float(y)
                    })
        
        # Collect ball data
        for ball_id, ball_data in ball_tracks.items():
            bbox = ball_data.get('bbox', [])
            if len(bbox) >= 4:
                x = (bbox[0] + bbox[2]) / 2  # Center x
                y = (bbox[1] + bbox[3]) / 2  # Center y
                frame_data['ball_data'].append({
                    'ball_id': ball_id,
                    'x': float(x),
                    'y': float(y),
                    'confidence': float(ball_data.get('conf', 0.0))
                })
        
        # Collect player data
        for player_id, player_data in player_tracks.items():
            bbox = player_data.get('bbox', [])
            if len(bbox) >= 4:
                x = (bbox[0] + bbox[2]) / 2  # Center x
                y = (bbox[1] + bbox[3]) / 2  # Center y
                
                # Get team assignment
                team_id = player_assignment.get(player_id, 0) if player_assignment else 0
                
                frame_data['player_data'].append({
                    'player_id': player_id,
                    'x': float(x),
                    'y': float(y),
                    'team_id': team_id,
                    'class': player_data.get('class', 'Player'),
                    'confidence': float(player_data.get('conf', 0.0))
                })
        
        self.data.append(frame_data)
    
    def export_to_csv(self):
        """
        Export collected data to CSV files.
        Creates separate files for court keypoints, ball data, and player data.
        """
        if not self.data:
            print("No data to export")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Export court keypoints
        court_keypoints_path = self.output_path.replace('.csv', '_court_keypoints.csv')
        with open(court_keypoints_path, 'w', newline='') as csvfile:
            fieldnames = ['frame', 'keypoint_id', 'x', 'y']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for frame_data in self.data:
                for kp in frame_data['court_keypoints']:
                    writer.writerow({
                        'frame': frame_data['frame'],
                        'keypoint_id': kp['keypoint_id'],
                        'x': kp['x'],
                        'y': kp['y']
                    })
        
        # Export ball data
        ball_data_path = self.output_path.replace('.csv', '_ball_data.csv')
        with open(ball_data_path, 'w', newline='') as csvfile:
            fieldnames = ['frame', 'ball_id', 'x', 'y', 'confidence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for frame_data in self.data:
                for ball in frame_data['ball_data']:
                    writer.writerow({
                        'frame': frame_data['frame'],
                        'ball_id': ball['ball_id'],
                        'x': ball['x'],
                        'y': ball['y'],
                        'confidence': ball['confidence']
                    })
        
        # Export player data
        player_data_path = self.output_path.replace('.csv', '_player_data.csv')
        with open(player_data_path, 'w', newline='') as csvfile:
            fieldnames = ['frame', 'player_id', 'x', 'y', 'team_id', 'class', 'confidence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for frame_data in self.data:
                for player in frame_data['player_data']:
                    writer.writerow({
                        'frame': frame_data['frame'],
                        'player_id': player['player_id'],
                        'x': player['x'],
                        'y': player['y'],
                        'team_id': player['team_id'],
                        'class': player['class'],
                        'confidence': player['confidence']
                    })
        
        print(f"CSV data exported to:")
        print(f"  Court keypoints: {court_keypoints_path}")
        print(f"  Ball data: {ball_data_path}")
        print(f"  Player data: {player_data_path}")
    
    def export_combined_csv(self):
        """
        Export all data to a single combined CSV file.
        """
        if not self.data:
            print("No data to export")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        with open(self.output_path, 'w', newline='') as csvfile:
            fieldnames = ['frame', 'data_type', 'object_id', 'x', 'y', 'team_id', 'class', 'confidence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for frame_data in self.data:
                # Write court keypoints
                for kp in frame_data['court_keypoints']:
                    writer.writerow({
                        'frame': frame_data['frame'],
                        'data_type': 'court_keypoint',
                        'object_id': kp['keypoint_id'],
                        'x': kp['x'],
                        'y': kp['y'],
                        'team_id': '',
                        'class': '',
                        'confidence': ''
                    })
                
                # Write ball data
                for ball in frame_data['ball_data']:
                    writer.writerow({
                        'frame': frame_data['frame'],
                        'data_type': 'ball',
                        'object_id': ball['ball_id'],
                        'x': ball['x'],
                        'y': ball['y'],
                        'team_id': '',
                        'class': '',
                        'confidence': ball['confidence']
                    })
                
                # Write player data
                for player in frame_data['player_data']:
                    writer.writerow({
                        'frame': frame_data['frame'],
                        'data_type': 'player',
                        'object_id': player['player_id'],
                        'x': player['x'],
                        'y': player['y'],
                        'team_id': player['team_id'],
                        'class': player['class'],
                        'confidence': player['confidence']
                    })
        
        print(f"Combined CSV data exported to: {self.output_path}") 