from .utils import draw_ellipse, draw_traingle, draw_bbox
import cv2

class PlayerTracksDrawer:
    """
    A class responsible for drawing player tracks and ball possession indicators on video frames.

    Attributes:
        default_player_team_id (int): Default team ID used when a player's team is not specified.
        team_1_color (list): RGB color used to represent Team 1 players.
        team_2_color (list): RGB color used to represent Team 2 players.
    """
    def __init__(self, team_1_color=[255, 245, 238], team_2_color=[128, 0, 0]):
        """
        Initialize the PlayerTracksDrawer with specified team colors.

        Args:
            team_1_color (list, optional): RGB color for Team 1. Defaults to [255, 245, 238].
            team_2_color (list, optional): RGB color for Team 2. Defaults to [128, 0, 0].
        """
        self.default_player_team_id = 1
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color

    def draw(self, video_frames, tracks, player_assignment, ball_aquisition):
        """
        Draw player tracks and ball possession indicators on a list of video frames.

        Args:
            video_frames (list): A list of frames (as NumPy arrays or image objects) on which to draw.
            tracks (list): A list of dictionaries where each dictionary contains player tracking information
                for the corresponding frame.
            player_assignment (list): A list of dictionaries indicating team assignments for each player
                in the corresponding frame.
            ball_aquisition (list): A list indicating which player has possession of the ball in each frame.

        Returns:
            list: A list of frames with player tracks and ball possession indicators drawn on them.
        """

        for frame_num, frame in enumerate(video_frames):
            player_dict = tracks[frame_num]
            player_assignment_for_frame = player_assignment[frame_num]
            player_id_has_ball = ball_aquisition[frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                obj_class = player.get("class", "player")  # default to "player" if not present
                confidence = player.get("confidence", None)

                if obj_class == "Hoop":
                    # Draw bounding box for hoops without ID or confidence
                    draw_bbox(frame, player["bbox"], (0, 255, 255), None, None)  # yellow for hoop, no text
                elif obj_class == "Ref":
                    # Draw pink upside-down triangles for referees with ref IDs
                    ref_color = [255, 0, 255]  # Pink color for referees
                    draw_traingle(frame, player["bbox"], ref_color, track_id)
                else:
                    # Draw triangles for players with team colors and player IDs
                    team_id = player_assignment_for_frame.get(track_id, self.default_player_team_id)
                    color = self.team_1_color if team_id == 1 else self.team_2_color
                    
                    # Draw triangle with player ID
                    draw_traingle(frame, player["bbox"], color, track_id)
                    
                    # Add ball possession indicator if player has the ball
                    if track_id == player_id_has_ball:
                        # Draw a small circle or highlight to indicate ball possession
                        x1, y1, x2, y2 = player["bbox"]
                        x_center = int((x1 + x2) / 2)
                        y_center = int((y1 + y2) / 2)
                        
                        # Draw a small filled circle to indicate ball possession
                        cv2.circle(frame, (x_center, y_center), 8, (0, 0, 255), -1)  # Red circle
                        cv2.circle(frame, (x_center, y_center), 8, (255, 255, 255), 2)  # White border

        return video_frames
        
