from PIL import Image
import cv2
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict, Counter
import sys 
sys.path.append('../')
from utils import read_stub, save_stub

class TeamAssigner:
    """
    A class that assigns players to teams based on their jersey colors using visual analysis.

    This improved version enforces:
    - Maximum 5 players per team at any time
    - Temporal consistency (players can't switch teams during play)
    - Conflict resolution for ambiguous cases
    - Consensus-based team assignment across multiple frames
    """
    
    def __init__(self,
                 team_1_class_name="white basketball jersey",
                 team_2_class_name="dark blue basketball jersey",
                 max_players_per_team=5,
                 consensus_frames=5,
                 confidence_threshold=0.6):
        """
        Initialize the TeamAssigner with specified team jersey descriptions.

        Args:
            team_1_class_name (str): Description of Team 1's jersey appearance.
            team_2_class_name (str): Description of Team 2's jersey appearance.
            max_players_per_team (int): Maximum players allowed per team.
            consensus_frames (int): Number of frames to use for consensus voting.
            confidence_threshold (float): Minimum confidence for team assignment.
        """
        self.team_colors = {}
        self.player_team_dict = {}        
        self.player_confidence_dict = {}
        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name
        self.max_players_per_team = max_players_per_team
        self.consensus_frames = consensus_frames
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.processor = None

    def load_model(self):
        """
        Loads the pre-trained vision model for jersey color classification.
        """
        if self.model is None:
            self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
            self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
            print(f"Team assigner loaded with classes: '{self.team_1_class_name}' and '{self.team_2_class_name}'")

    def get_player_color(self, frame, bbox):
        """
        Analyzes the jersey color of a player within the given bounding box.

        Args:
            frame (numpy.ndarray): The video frame containing the player.
            bbox (tuple): Bounding box coordinates of the player.

        Returns:
            tuple: (team_id, confidence_score) where team_id is 1 or 2
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure valid bounding box
        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
            return None, 0.0
            
        # Extract player image
        image = frame[y1:y2, x1:x2]
        
        if image.size == 0:
            return None, 0.0

        # Convert to PIL Image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        classes = [self.team_1_class_name, self.team_2_class_name]

        inputs = self.processor(text=classes, images=pil_image, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1) 

        # Get confidence and team assignment
        confidence = probs.max().item()
        team_id = 1 if probs.argmax(dim=1)[0] == 0 else 2

        return team_id, confidence

    def get_player_consensus_team(self, player_id, frames, bboxes):
        """
        Get team assignment for a player using consensus across multiple frames.

        Args:
            player_id (int): Player ID
            frames (list): List of frames containing the player
            bboxes (list): List of bounding boxes for the player

        Returns:
            tuple: (team_id, confidence) or (None, 0.0) if no consensus
        """
        votes = {1: [], 2: []}  # Store confidence scores for each team
        
        for frame, bbox in zip(frames, bboxes):
            team_id, confidence = self.get_player_color(frame, bbox)
            if team_id is not None and confidence > self.confidence_threshold:
                votes[team_id].append(confidence)
        
        # Calculate average confidence for each team
        team_avg_confidence = {}
        for team_id, confidences in votes.items():
            if confidences:
                team_avg_confidence[team_id] = np.mean(confidences)
        
        if not team_avg_confidence:
            return None, 0.0
            
        # Return team with highest average confidence
        best_team = max(team_avg_confidence, key=team_avg_confidence.get)
        return best_team, team_avg_confidence[best_team]

    def validate_team_assignments(self, frame_assignments):
        """
        Validate and fix team assignments to ensure max 5 players per team.

        Args:
            frame_assignments (list): List of team assignments for each frame

        Returns:
            list: Corrected team assignments
        """
        corrected_assignments = []
        
        for frame_idx, frame_assignment in enumerate(frame_assignments):
            corrected_frame = frame_assignment.copy()
            
            # Count players per team
            team_counts = Counter(frame_assignment.values())
            
            # If any team has more than max players, resolve conflicts
            if team_counts[1] > self.max_players_per_team or team_counts[2] > self.max_players_per_team:
                print(f"Frame {frame_idx}: Team 1 has {team_counts[1]} players, Team 2 has {team_counts[2]} players")
                
                # Get players for each team
                team_1_players = [pid for pid, team in frame_assignment.items() if team == 1]
                team_2_players = [pid for pid, team in frame_assignment.items() if team == 2]
                
                # Resolve conflicts by reassigning players with lower confidence
                if team_counts[1] > self.max_players_per_team:
                    self._resolve_team_conflict(corrected_frame, team_1_players, 1, frame_idx)
                    
                if team_counts[2] > self.max_players_per_team:
                    self._resolve_team_conflict(corrected_frame, team_2_players, 2, frame_idx)
            
            corrected_assignments.append(corrected_frame)
            
        return corrected_assignments

    def _resolve_team_conflict(self, frame_assignment, team_players, team_id, frame_idx):
        """
        Resolve team conflicts by reassigning players with lowest confidence.
        
        Args:
            frame_assignment (dict): Current frame assignments
            team_players (list): List of player IDs in the team
            team_id (int): Team ID that has too many players
            frame_idx (int): Current frame index
        """
        # Get confidence scores for players in this team
        player_confidences = []
        for player_id in team_players:
            confidence = self.player_confidence_dict.get(player_id, 0.0)
            player_confidences.append((player_id, confidence))
        
        # Sort by confidence (lowest first)
        player_confidences.sort(key=lambda x: x[1])
        
        # Reassign players with lowest confidence to the other team
        excess_players = len(team_players) - self.max_players_per_team
        other_team = 3 - team_id  # 1 -> 2, 2 -> 1
        
        for i in range(excess_players):
            player_id, confidence = player_confidences[i]
            frame_assignment[player_id] = other_team
            print(f"Frame {frame_idx}: Reassigned player {player_id} from team {team_id} to team {other_team} (confidence: {confidence:.3f})")

    def enforce_temporal_consistency(self, frame_assignments):
        """
        Enforce temporal consistency by preventing players from switching teams.

        Args:
            frame_assignments (list): List of team assignments for each frame

        Returns:
            list: Temporally consistent team assignments
        """
        # Track player team history
        player_team_history = {}
        consistent_assignments = []
        
        for frame_idx, frame_assignment in enumerate(frame_assignments):
            consistent_frame = {}
            
            for player_id, team_id in frame_assignment.items():
                if player_id in player_team_history:
                    # Player has been assigned before - maintain consistency
                    consistent_frame[player_id] = player_team_history[player_id]
                else:
                    # New player - assign team and record
                    consistent_frame[player_id] = team_id
                    player_team_history[player_id] = team_id
            
            consistent_assignments.append(consistent_frame)
        
        return consistent_assignments

    def get_player_teams_across_frames(self, video_frames, player_tracks, read_from_stub=False, stub_path=None):
        """
        Processes all video frames to assign teams to players with improved logic.

        Args:
            video_frames (list): List of video frames to process.
            player_tracks (list): List of player tracking information for each frame.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries mapping player IDs to team assignments for each frame.
        """
        
        player_assignment = read_stub(read_from_stub, stub_path)
        if player_assignment is not None:
            if len(player_assignment) == len(video_frames):
                return player_assignment

        self.load_model()

        # Step 1: Initial team assignment with consensus
        print("Performing initial team assignment with consensus...")
        initial_assignments = []
        player_frames = defaultdict(list)
        player_bboxes = defaultdict(list)
        
        # Collect player appearances across frames
        for frame_num, player_track in enumerate(player_tracks):        
            for player_id, track in player_track.items():
                obj_class = track.get("class", "Player")
                if obj_class == "Player":
                    player_frames[player_id].append(video_frames[frame_num])
                    player_bboxes[player_id].append(track['bbox'])
        
        # Get consensus team assignment for each player
        for player_id in player_frames:
            frames = player_frames[player_id]
            bboxes = player_bboxes[player_id]
            
            # Use consensus across multiple frames
            team_id, confidence = self.get_player_consensus_team(player_id, frames, bboxes)
            
            if team_id is not None:
                self.player_team_dict[player_id] = team_id
                self.player_confidence_dict[player_id] = confidence
                print(f"Player {player_id}: assigned to team {team_id} (confidence: {confidence:.3f})")
            else:
                # Default assignment if no consensus
                self.player_team_dict[player_id] = 1
                self.player_confidence_dict[player_id] = 0.0
                print(f"Player {player_id}: no consensus, defaulting to team 1")
        
        # Step 2: Create frame-by-frame assignments
        for frame_num, player_track in enumerate(player_tracks):
            frame_assignment = {}
            for player_id, track in player_track.items():
                obj_class = track.get("class", "Player")
                if obj_class == "Player" and player_id in self.player_team_dict:
                    frame_assignment[player_id] = self.player_team_dict[player_id]
            initial_assignments.append(frame_assignment)
        
        # Step 3: Validate team sizes
        print("Validating team assignments...")
        validated_assignments = self.validate_team_assignments(initial_assignments)
        
        # Step 4: Enforce temporal consistency
        print("Enforcing temporal consistency...")
        final_assignments = self.enforce_temporal_consistency(validated_assignments)
        
        # Step 5: Final validation
        print("Performing final validation...")
        final_assignments = self.validate_team_assignments(final_assignments)
        
        # Print final statistics
        self._print_assignment_statistics(final_assignments)
        
        save_stub(stub_path, final_assignments)
        return final_assignments

    def _print_assignment_statistics(self, assignments):
        """
        Print statistics about team assignments.
        
        Args:
            assignments (list): Final team assignments
        """
        total_frames = len(assignments)
        team_1_counts = []
        team_2_counts = []
        
        for frame_assignment in assignments:
            team_counts = Counter(frame_assignment.values())
            team_1_counts.append(team_counts.get(1, 0))
            team_2_counts.append(team_counts.get(2, 0))
        
        avg_team_1 = np.mean(team_1_counts)
        avg_team_2 = np.mean(team_2_counts)
        max_team_1 = max(team_1_counts)
        max_team_2 = max(team_2_counts)
        
        print(f"Team assignment statistics:")
        print(f"  Average players per frame - Team 1: {avg_team_1:.1f}, Team 2: {avg_team_2:.1f}")
        print(f"  Maximum players per frame - Team 1: {max_team_1}, Team 2: {max_team_2}")
        print(f"  Total unique players assigned: {len(self.player_team_dict)}")