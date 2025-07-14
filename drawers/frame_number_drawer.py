import cv2
class FrameNumberDrawer:
    def __init__(self):
        pass

    def draw(self, frames):
        """
        Draws frame numbers on a given list of frames.

        Args:
            frames (list): A list of frames (as NumPy arrays or image objects) on which to draw.

        Returns:
            list: A list of frames with frame numbers drawn on them.
        """
        for i in range(len(frames)):
            # Draw frame number in top-left corner
            cv2.putText(frames[i], f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frames