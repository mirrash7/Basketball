import numpy as np
import cv2 

class Homography:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape.")
        if source.shape[1] != 2:
            raise ValueError("Source and target points must be 2D coordinates.")
        
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        
        self.m, _ = cv2.findHomography(source, target)
        if self.m is None:
            raise ValueError("Homography matrix could not be calculated.")
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")
        
        points = points.reshape(-1, 1, 2).astype(np.float32)
        points = cv2.perspectiveTransform(points, self.m)
        return points.reshape(-1, 2).astype(np.float32)

    def create_court_mask(self, frame_shape, keypoints):
        """Create binary mask of court area using homography"""
        # Create tactical view court outline
        court_outline = np.array([
            [0, 0], [0, self.height], [self.width, self.height], 
            [self.width, 0]
        ], dtype=np.float32)
        
        # Transform back to video coordinates using reverse homography
        homography = Homography(self.key_points, keypoints)
        video_court_outline = homography.transform_points(court_outline)
        
        # Create mask
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [video_court_outline.astype(np.int32)], 255)
        return mask

    def filter_court_detections(self, detections, mask):
        """Remove detections outside court area"""
        filtered = []
        for det in detections:
            bbox_center = get_bbox_center(det.bbox)
            if mask[int(bbox_center[1]), int(bbox_center[0])] > 0:
                filtered.append(det)
        return filtered

    def filter_player_detections(self, detections, mask):
        """Strict filtering - players must be on court"""
        return [det for det in detections if self.is_on_court(det.bbox, mask)]





