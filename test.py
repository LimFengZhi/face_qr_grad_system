"""
Test script for Person Tracker
Run: python test_person_tracker.py
"""

import cv2 as cv
from person_tracker import PersonTracker

def main():
    # Initialize tracker
    tracker = PersonTracker(
        model_name='yolov8n.pt',
        confidence=0.5,
        verification_zone=(0.3, 0.1, 0.7, 0.95)  # Center zone
    )
    
    # Open camera
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Press 'q' to quit")
    print("Press 'r' to reset all tracks")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get person in verification zone
        person = tracker.get_person_in_zone(frame)
        
        if person:
            print(f"Person {person['track_id']} in zone! Frames: {person['state']['frame_count_in_zone']}")
        
        # Draw visualization
        disp = tracker.draw_tracking(frame, show_zone=True, show_tracks=True)
        
        # Show stats
        stats = tracker.get_stats()
        cv.putText(disp, f"Tracks: {stats['total_tracks']} | In Zone: {stats['in_zone']}", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv.imshow('Person Tracker Test', disp)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.tracks.clear()
            tracker.track_states.clear()
            print("Tracks reset!")
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()