import cv2 as cv
import numpy as np
from typing import Tuple, Union, List

class BoxHelper:

    
    @staticmethod
    def draw_box(
        image: np.ndarray,
        coordinates: Union[Tuple[int, int, int, int], List[Tuple[int, int]]],
        text: str = "",
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        text_position: str = "bottom",  # "top" or "bottom"
        text_bg_color: Tuple[int, int, int] = None,
        font_scale: float = 0.6
    ) -> np.ndarray:
        """
        Universal function to draw box and text
        
        Args:
            image: Input image (BGR)
            coordinates: Either:
                - (x, y, w, h) for rectangle
                - (top, right, bottom, left) for face box
                - List of (x, y) points for polygon
            text: Text to display
            color: Box color (BGR)
            thickness: Line thickness
            text_position: "top" or "bottom" - where to place text
            text_bg_color: Background color for text (None = same as box color)
            font_scale: Text size
        
        Returns:
            Image with drawn box and text
        """
        output = image.copy()
        
        # Handle different coordinate formats
        if isinstance(coordinates, (list, tuple)):
            if len(coordinates) == 4 and isinstance(coordinates[0], (int, np.integer)):
                # It's a rectangle: (x, y, w, h) or (top, right, bottom, left)
                coords = coordinates
                
                # Detect format
                if coords[0] < coords[2] and coords[1] > coords[3]:
                    # Format: (top, right, bottom, left)
                    top, right, bottom, left = coords
                    x, y, w, h = left, top, right - left, bottom - top
                else:
                    # Format: (x, y, w, h)
                    x, y, w, h = coords
                
                # Draw rectangle
                cv.rectangle(output, (x, y), (x + w, y + h), color, thickness)
                
            else:
                # It's a polygon: list of (x, y) points
                pts = np.array(coordinates, dtype=np.int32)
                cv.polylines(output, [pts], True, color, thickness)
                
                # Get bounding box for text placement
                x = min(pt[0] for pt in coordinates)
                y = min(pt[1] for pt in coordinates)
                w = max(pt[0] for pt in coordinates) - x
                h = max(pt[1] for pt in coordinates) - y
        else:
            return output
        
        # Draw text if provided
        if text:
            if text_bg_color is None:
                text_bg_color = color
            
            font = cv.FONT_HERSHEY_DUPLEX
            (text_width, text_height), baseline = cv.getTextSize(text, font, font_scale, 1)
            
            padding = 6
            
            # Calculate text position
            if text_position == "bottom":
                text_y_bg = y + h
                text_y_text = y + h + text_height + padding
            else:  # top
                text_y_bg = y - text_height - padding * 2
                text_y_text = y - padding
            
            # Draw text background
            cv.rectangle(
                output,
                (x, text_y_bg),
                (x + text_width + padding * 2, text_y_bg + text_height + padding * 2),
                text_bg_color,
                -1
            )
            
            # Draw text
            cv.putText(
                output,
                text,
                (x + padding, text_y_text),
                font,
                font_scale,
                (255, 255, 255),
                1,
                cv.LINE_AA
            )
        
        return output
    
    @staticmethod
    def draw_info(
        image: np.ndarray,
        text: str,
        position: str = "top-left",
        color: Tuple[int, int, int] = (0, 255, 0),
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        font_scale: float = 0.7
    ) -> np.ndarray:
        """
        Draw info text in corner
        
        Args:
            image: Input image
            text: Text to display
            position: "top-left", "top-right", "bottom-left", "bottom-right"
            color: Text color
            bg_color: Background color
            font_scale: Text size
        
        Returns:
            Image with info text
        """
        output = image.copy()
        font = cv.FONT_HERSHEY_SIMPLEX
        
        (text_width, text_height), _ = cv.getTextSize(text, font, font_scale, 2)
        
        h, w = image.shape[:2]
        padding = 10
        
        # Calculate position
        if position == "top-left":
            x, y = padding, padding + text_height
        elif position == "top-right":
            x, y = w - text_width - padding, padding + text_height
        elif position == "bottom-left":
            x, y = padding, h - padding
        else:  # bottom-right
            x, y = w - text_width - padding, h - padding
        
        # Draw background
        cv.rectangle(output, (x - 5, y - text_height - 5), 
                    (x + text_width + 5, y + 5), bg_color, -1)
        
        # Draw text
        cv.putText(output, text, (x, y), font, font_scale, color, 2, cv.LINE_AA)
        
        return output