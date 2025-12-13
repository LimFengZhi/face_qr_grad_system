import cv2 as cv
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import time

class PersonTracker:
    """
    Person detection and tracking using YOLOv8 + Simple Centroid Tracking
    
    Features:
    - Detect people using YOLOv8
    - Track people across frames with unique IDs
    - Define verification zone
    - Trigger face/QR scan only when person is in zone
    """
    
    def __init__(self, model_name: str = 'yolov8n.pt', confidence: float = 0.5, 
                 verification_zone: Tuple[float, float, float, float] = None,
                 max_tracks: int = 1):
        """
        Initialize person tracker.
        
        Args:
            model_name: YOLOv8 model name ('yolov8n.pt', 'yolov8s.pt', etc.)
            confidence: Detection confidence threshold
            verification_zone: (x1_ratio, y1_ratio, x2_ratio, y2_ratio) - ratios of frame size
                              e.g., (0.3, 0.2, 0.7, 0.9) = center zone
            max_tracks: Maximum number of people to track simultaneously (default: 1)
        """
        self.confidence = confidence
        self.verification_zone = verification_zone or (0.25, 0.1, 0.75, 0.95)
        self.model = None
        self.model_name = model_name
        self.max_tracks = max_tracks
        
        # Tracking state
        self.tracks = {}  # track_id: {'bbox': (x1,y1,x2,y2), 'centroid': (cx,cy), 'last_seen': time}
        self.next_track_id = 1
        self.max_disappeared = 30  # frames before removing track
        self.max_distance = 100  # max pixel distance to match tracks
        
        # Verification state per track
        self.track_states = defaultdict(lambda: {
            'in_zone': False,
            'face_verified': False,
            'face_id': None,
            'qr_verified': False,
            'qr_id': None,
            'verified': False,
            'frame_count_in_zone': 0
        })
        
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 model"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            print(f"✓ Person Tracker loaded: {self.model_name}")
        except ImportError:
            print("❌ ultralytics not installed. Run: pip install ultralytics")
            self.model = None
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            self.model = None
    
    def detect_people(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect people in frame.
        
        Returns:
            List of detections: [{'bbox': (x1,y1,x2,y2), 'confidence': float, 'centroid': (cx,cy)}]
        """
        if self.model is None:
            return []
        
        try:
            # Run YOLO detection (class 0 = person)
            results = self.model(frame, classes=[0], conf=self.confidence, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                    
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    bbox = (int(x1), int(y1), int(x2), int(y2))
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    detections.append({
                        'bbox': bbox,
                        'confidence': conf,
                        'centroid': (cx, cy)
                    })
            
            # Sort by: 1) bounding box area (larger = closer), 2) bottom y (tie-breaker)
            def proximity_score(det):
                bbox = det['bbox']
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                bottom_y = bbox[3]
                return (area, bottom_y)  # Prioritize area, then bottom position
            
            detections.sort(key=proximity_score, reverse=True)
            detections = detections[:self.max_tracks]
            
            return detections
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def update_tracks(self, detections: List[Dict]) -> Dict:
        """
        Update tracks with new detections using centroid tracking.
        
        Returns:
            Updated tracks dictionary
        """
        current_time = time.time()
        
        if len(detections) == 0:
            # Mark all tracks as disappeared
            tracks_to_remove = []
            for track_id, track in self.tracks.items():
                if current_time - track['last_seen'] > self.max_disappeared / 30:  # ~1 second
                    tracks_to_remove.append(track_id)
            
            for track_id in tracks_to_remove:
                del self.tracks[track_id]
                if track_id in self.track_states:
                    del self.track_states[track_id]
            
            return self.tracks
        
        # If no existing tracks, create new ones
        if len(self.tracks) == 0:
            for det in detections[:self.max_tracks]:
                self.tracks[self.next_track_id] = {
                    'bbox': det['bbox'],
                    'centroid': det['centroid'],
                    'last_seen': current_time
                }
                self.next_track_id += 1
            return self.tracks
        
        # Match detections to existing tracks
        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid]['centroid'] for tid in track_ids]
        det_centroids = [d['centroid'] for d in detections]
        
        # Calculate distance matrix
        used_tracks = set()
        used_detections = set()
        matches = []
        
        for i, det_c in enumerate(det_centroids):
            min_dist = float('inf')
            min_track_idx = -1
            
            for j, track_c in enumerate(track_centroids):
                if j in used_tracks:
                    continue
                dist = np.sqrt((det_c[0] - track_c[0])**2 + (det_c[1] - track_c[1])**2)
                if dist < min_dist and dist < self.max_distance:
                    min_dist = dist
                    min_track_idx = j
            
            if min_track_idx >= 0:
                matches.append((i, min_track_idx))
                used_tracks.add(min_track_idx)
                used_detections.add(i)
        
        # Update matched tracks
        for det_idx, track_idx in matches:
            track_id = track_ids[track_idx]
            self.tracks[track_id] = {
                'bbox': detections[det_idx]['bbox'],
                'centroid': detections[det_idx]['centroid'],
                'last_seen': current_time
            }
        
        # Create new tracks for unmatched detections (if under max_tracks)
        for i, det in enumerate(detections):
            if i not in used_detections and len(self.tracks) < self.max_tracks:
                self.tracks[self.next_track_id] = {
                    'bbox': det['bbox'],
                    'centroid': det['centroid'],
                    'last_seen': current_time
                }
                self.next_track_id += 1
        
        # Remove old tracks
        tracks_to_remove = []
        for track_idx, track_id in enumerate(track_ids):
            if track_idx not in used_tracks:
                if current_time - self.tracks[track_id]['last_seen'] > self.max_disappeared / 30:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            if track_id in self.track_states:
                del self.track_states[track_id]
        
        return self.tracks
    
    def get_verification_zone_pixels(self, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Convert zone ratios to pixel coordinates"""
        h, w = frame_shape[:2]
        x1 = int(self.verification_zone[0] * w)
        y1 = int(self.verification_zone[1] * h)
        x2 = int(self.verification_zone[2] * w)
        y2 = int(self.verification_zone[3] * h)
        return (x1, y1, x2, y2)
    
    def is_in_verification_zone(self, bbox: Tuple[int, int, int, int], 
                                 frame_shape: Tuple[int, int]) -> bool:
        """Check if person's center is in verification zone"""
        zone = self.get_verification_zone_pixels(frame_shape)
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        
        return (zone[0] <= cx <= zone[2]) and (zone[1] <= cy <= zone[3])
    
    def get_person_in_zone(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Get the person currently in verification zone.
        
        Returns:
            Dict with track_id, bbox, state or None if no one in zone
        """
        detections = self.detect_people(frame)
        self.update_tracks(detections)
        
        # Find person in verification zone
        for track_id, track in self.tracks.items():
            bbox = track['bbox']
            if self.is_in_verification_zone(bbox, frame.shape):
                self.track_states[track_id]['in_zone'] = True
                self.track_states[track_id]['frame_count_in_zone'] += 1
                
                return {
                    'track_id': track_id,
                    'bbox': bbox,
                    'centroid': track['centroid'],
                    'state': self.track_states[track_id],
                    'roi': frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                }
            else:
                self.track_states[track_id]['in_zone'] = False
                self.track_states[track_id]['frame_count_in_zone'] = 0
        
        return None
    
    def update_verification_state(self, track_id: int, face_id: str = None, qr_id: str = None):
        """Update verification state for a track"""
        if face_id:
            self.track_states[track_id]['face_verified'] = True
            self.track_states[track_id]['face_id'] = face_id
        
        if qr_id:
            self.track_states[track_id]['qr_verified'] = True
            self.track_states[track_id]['qr_id'] = qr_id
        
        # Check if fully verified
        state = self.track_states[track_id]
        if state['face_verified'] and state['qr_verified']:
            if state['face_id'] == state['qr_id']:
                state['verified'] = True
    
    def reset_track_state(self, track_id: int):
        """Reset verification state for a track"""
        self.track_states[track_id] = {
            'in_zone': False,
            'face_verified': False,
            'face_id': None,
            'qr_verified': False,
            'qr_id': None,
            'verified': False,
            'frame_count_in_zone': 0
        }
    
    def draw_tracking(self, frame: np.ndarray, show_zone: bool = True, 
                      show_tracks: bool = True) -> np.ndarray:
        """
        Draw tracking visualization on frame.
        
        Args:
            frame: Input frame
            show_zone: Draw verification zone
            show_tracks: Draw person bounding boxes and IDs
            
        Returns:
            Annotated frame
        """
        disp = frame.copy()
        
        # Draw verification zone
        if show_zone:
            zone = self.get_verification_zone_pixels(frame.shape)
            cv.rectangle(disp, (zone[0], zone[1]), (zone[2], zone[3]), (0, 255, 255), 2)
            cv.putText(disp, "VERIFICATION ZONE", (zone[0] + 10, zone[1] + 25), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw tracks
        if show_tracks:
            for track_id, track in self.tracks.items():
                bbox = track['bbox']
                state = self.track_states[track_id]
                
                # Color based on state
                if state['in_zone']:
                    color = (255, 0, 255)  # Purple - in zone
                    label = "IN ZONE"
                else:
                    color = (255, 0, 0)  # Blue - outside zone
                    label = None  # Don't show label when outside
                
                # Draw bounding box
                cv.rectangle(disp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                # Draw label
                cv.putText(disp, label, (bbox[0], bbox[1] - 10), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw verification status
                if state['in_zone']:
                    status_y = bbox[3] + 20
                    face_status = f"Face: {state['face_id'] or 'scanning...'}"
                    qr_status = f"QR: {state['qr_id'] or 'scanning...'}"
                    cv.putText(disp, face_status, (bbox[0], status_y), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    cv.putText(disp, qr_status, (bbox[0], status_y + 15), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return disp
    
    def get_stats(self) -> Dict:
        """Get tracking statistics"""
        return {
            'total_tracks': len(self.tracks),
            'in_zone': sum(1 for s in self.track_states.values() if s['in_zone']),
            'verified': sum(1 for s in self.track_states.values() if s['verified'])
        }


# Singleton instance
_person_tracker = None

def get_person_tracker(model_name: str = 'yolov8n.pt', **kwargs) -> PersonTracker:
    """Get or create person tracker instance"""
    global _person_tracker
    if _person_tracker is None:
        _person_tracker = PersonTracker(model_name=model_name, **kwargs)
    return _person_tracker