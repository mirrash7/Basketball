from .utils import draw_traingle

class BallTracksDrawer:
    """
    A drawer class responsible for drawing ball tracks on video frames.

    Attributes:
        ball_pointer_color (tuple): The color used to draw the ball pointers (in BGR format).
    """

    def __init__(self):
        """
        Initialize the BallTracksDrawer instance with default settings.
        """
        self.ball_pointer_color = (0, 255, 0)

    def draw(self, video_frames, tracks):
        """
        Draws ball pointers on each video frame based on provided tracking information.

        Args:
            video_frames (list): A list of video frames (as NumPy arrays or image objects).
            tracks (list): A list of dictionaries where each dictionary contains ball information
                for the corresponding frame.

        Returns:
            list: A list of processed video frames with drawn ball pointers.
        """
        print(f"Ball drawer: Processing {len(tracks)} track frames")
        
        for frame_num, frame in enumerate(video_frames):
            if frame_num >= len(tracks):
                print(f"Warning: Frame {frame_num} has no track data")
                continue
                
            ball_dict = tracks[frame_num]
            print(f"Frame {frame_num}: Ball dict keys: {list(ball_dict.keys())}")

            # Draw ball 
            for ball_id, ball in ball_dict.items():
                if ball["bbox"] is None:
                    continue
                
                print(f"  Ball {ball_id}: bbox={ball['bbox']}, confidence={ball.get('confidence', 'None')}")
                
                # Extract confidence from ball data
                confidence = None
                if "confidence" in ball:
                    # Only show confidence for actual detections (not predicted/fallback)
                    flags = [key for key in ball.keys() if key in ["predicted", "fallback", "predicted_found", "final_fallback", "ultimate_fallback", "center_fallback"]]
                    if not flags:
                        confidence = ball["confidence"]
                        print(f"    ✅ Will display confidence: {confidence}")
                    else:
                        print(f"    ❌ Confidence blocked by flags: {flags}")
                else:
                    print(f"    ❌ No confidence in ball data")
                
                draw_traingle(frame, ball["bbox"], self.ball_pointer_color, confidence=confidence)
            
        return video_frames