import cv2 
import numpy as np
import gc

class TacticalViewDrawer:
    def __init__(self, team_1_color=[255, 245, 238], team_2_color=[128, 0, 0]):
        self.start_x = 20
        self.start_y = 40
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color

    def draw(self, 
             video_frames, 
             court_image_path, 
             width,
             height,
             tactical_court_keypoints,
             tactical_player_positions=None,
             player_assignment=None,
             ball_acquisition=None):
        """
        Draw tactical view with court keypoints and player positions.
        Memory-efficient version that processes frames in small batches.
        
        Args:
            video_frames (list): List of video frames to draw on.
            court_image_path (str): Path to the court image.
            width (int): Width of the tactical view.
            height (int): Height of the tactical view.
            tactical_court_keypoints (list): List of court keypoints in tactical view.
            tactical_player_positions (list, optional): List of dictionaries mapping player IDs to 
                their positions in tactical view coordinates.
            player_assignment (list, optional): List of dictionaries mapping player IDs to team assignments.
            ball_acquisition (list, optional): List indicating which player has the ball in each frame.
            
        Returns:
            list: List of frames with tactical view drawn on them.
        """
        court_image = cv2.imread(court_image_path)
        court_image = cv2.resize(court_image, (width, height))

        output_video_frames = []
        
        # Process frames in very small batches to minimize memory usage
        batch_size = 5  # Process only 5 frames at a time
        
        for batch_start in range(0, len(video_frames), batch_size):
            batch_end = min(batch_start + batch_size, len(video_frames))
            
            # Process this batch
            batch_output = self._process_batch(
                video_frames[batch_start:batch_end],
                court_image,
                width,
                height,
                tactical_court_keypoints,
                tactical_player_positions[batch_start:batch_end] if tactical_player_positions else None,
                player_assignment[batch_start:batch_end] if player_assignment else None,
                ball_acquisition[batch_start:batch_end] if ball_acquisition else None,
                batch_start
            )
            
            output_video_frames.extend(batch_output)
            
            # Force garbage collection after each batch
            gc.collect()
            
            # Print progress
            if batch_start % 50 == 0:
                print(f"Processed {batch_start}/{len(video_frames)} frames for tactical view")

        return output_video_frames

    def _process_batch(self, batch_frames, court_image, width, height, tactical_court_keypoints, 
                      tactical_player_positions, player_assignment, ball_acquisition, batch_start):
        """Process a small batch of frames to minimize memory usage."""
        batch_output = []
        
        for frame_idx, frame in enumerate(batch_frames):
            global_frame_idx = batch_start + frame_idx
            
            # Create a copy of the frame (necessary for modifications)
            frame_copy = frame.copy()
            
            y1 = self.start_y
            y2 = self.start_y + height
            x1 = self.start_x
            x2 = self.start_x + width
            
            alpha = 0.6  # Transparency factor
            
            # Apply court overlay
            overlay_region = frame_copy[y1:y2, x1:x2]
            cv2.addWeighted(court_image, alpha, overlay_region, 1 - alpha, 0, overlay_region)
            
            # Draw court keypoints
            if tactical_court_keypoints:
                for keypoint_index, keypoint in enumerate(tactical_court_keypoints):
                    x, y = keypoint
                    x += self.start_x
                    y += self.start_y
                    cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)
                    cv2.putText(frame_copy, str(keypoint_index), (x, y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw player positions in tactical view if available
            if (tactical_player_positions and player_assignment and 
                frame_idx < len(tactical_player_positions) and frame_idx < len(player_assignment)):
                
                frame_positions = tactical_player_positions[frame_idx]
                frame_assignments = player_assignment[frame_idx]
                player_with_ball = (ball_acquisition[frame_idx] if ball_acquisition and 
                                   frame_idx < len(ball_acquisition) else -1)
                
                for player_id, position in frame_positions.items():
                    # Get player's team
                    team_id = frame_assignments.get(player_id, 1)  # Default to team 1 if not assigned
                    
                    # Set color based on team
                    color = self.team_1_color if team_id == 1 else self.team_2_color
                    
                    # Adjust position to overlay coordinates
                    x, y = int(position[0]) + self.start_x, int(position[1]) + self.start_y
                    
                    # Draw player circle
                    player_radius = 8
                    cv2.circle(frame_copy, (x, y), player_radius, color, -1)
                    
                    # Highlight player with ball
                    if player_id == player_with_ball:
                        cv2.circle(frame_copy, (x, y), player_radius+3, (0, 0, 255), 2)
            
            batch_output.append(frame_copy)
        
        return batch_output
