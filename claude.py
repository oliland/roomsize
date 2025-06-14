#!/usr/bin/env python3
"""
DXF Room Segmentation and Area Calculator
Reads DXF files, segments rooms, detects furniture, and calculates room areas.
"""

import ezdxf
import numpy as np
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.ops import unary_union, polygonize
import matplotlib.pyplot as plt
from collections import defaultdict
import math
from typing import List, Dict, Tuple, Optional

class DXFRoomAnalyzer:
    def __init__(self, dxf_file_path: str):
        """Initialize the DXF analyzer with a file path."""
        self.dxf_file_path = dxf_file_path
        self.doc = None
        self.walls = []
        self.doors = []
        self.windows = []
        self.furniture = []
        self.rooms = []
        self.room_areas = {}
        
        # Layer patterns for different elements
        self.wall_layers = ['WALL', 'WALLS', 'A-WALL', 'ARCH-WALL', '0']
        self.door_layers = ['DOOR', 'DOORS', 'A-DOOR', 'ARCH-DOOR']
        self.window_layers = ['WINDOW', 'WINDOWS', 'A-GLAZ', 'ARCH-GLAZ']
        self.furniture_layers = ['FURNITURE', 'FURN', 'A-FURN', 'ARCH-FURN', 'EQUIP']
        
        # Furniture detection patterns (block names and text content)
        self.furniture_patterns = [
            'bed', 'chair', 'table', 'desk', 'sofa', 'cabinet', 'shelf',
            'toilet', 'sink', 'bath', 'shower', 'stove', 'refrigerator',
            'washer', 'dryer', 'dishwasher', 'couch', 'dresser', 'wardrobe'
        ]
        
    def load_dxf(self) -> bool:
        """Load the DXF file."""
        try:
            self.doc = ezdxf.readfile(self.dxf_file_path)
            print(f"Successfully loaded DXF file: {self.dxf_file_path}")
            return True
        except Exception as e:
            print(f"Error loading DXF file: {e}")
            return False
    
    def extract_entities(self):
        """Extract different types of entities from the DXF file."""
        if not self.doc:
            return
        
        modelspace = self.doc.modelspace()
        
        for entity in modelspace:
            layer_name = entity.dxf.layer.upper()
            
            # Extract lines and polylines as potential walls
            if entity.dxftype() in ['LINE', 'LWPOLYLINE', 'POLYLINE']:
                if any(wall_layer in layer_name for wall_layer in self.wall_layers):
                    self.walls.append(self._entity_to_geometry(entity))
                elif any(door_layer in layer_name for door_layer in self.door_layers):
                    self.doors.append(self._entity_to_geometry(entity))
                elif any(window_layer in layer_name for window_layer in self.window_layers):
                    self.windows.append(self._entity_to_geometry(entity))
                elif any(furn_layer in layer_name for furn_layer in self.furniture_layers):
                    self.furniture.append(self._entity_to_geometry(entity))
            
            # Extract blocks (often used for furniture)
            elif entity.dxftype() == 'INSERT':
                block_name = entity.dxf.name.lower()
                if any(pattern in block_name for pattern in self.furniture_patterns):
                    self.furniture.append(self._block_to_geometry(entity))
                elif any(door_layer in layer_name for door_layer in self.door_layers):
                    self.doors.append(self._block_to_geometry(entity))
            
            # Extract text entities for furniture identification
            elif entity.dxftype() in ['TEXT', 'MTEXT']:
                text_content = entity.dxf.text.lower()
                if any(pattern in text_content for pattern in self.furniture_patterns):
                    # Create a small polygon around text location
                    x, y = entity.dxf.insert[:2]
                    self.furniture.append(Point(x, y).buffer(50))  # 50 unit buffer
        
        print(f"Extracted: {len(self.walls)} walls, {len(self.doors)} doors, "
              f"{len(self.windows)} windows, {len(self.furniture)} furniture items")
    
    def _entity_to_geometry(self, entity):
        """Convert DXF entity to Shapely geometry."""
        if entity.dxftype() == 'LINE':
            start = entity.dxf.start[:2]
            end = entity.dxf.end[:2]
            return LineString([start, end])
        
        elif entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
            points = []
            if hasattr(entity, 'get_points'):
                points = [(p[0], p[1]) for p in entity.get_points()]
            elif hasattr(entity, 'vertices'):
                points = [(v.dxf.location[0], v.dxf.location[1]) for v in entity.vertices]
            
            if len(points) > 2:
                if entity.dxf.flags & 1:  # Closed polyline
                    return Polygon(points)
                else:
                    return LineString(points)
            elif len(points) == 2:
                return LineString(points)
        
        return None
    
    def _block_to_geometry(self, insert_entity):
        """Convert block insert to geometry."""
        x, y = insert_entity.dxf.insert[:2]
        scale_x = getattr(insert_entity.dxf, 'xscale', 1.0)
        scale_y = getattr(insert_entity.dxf, 'yscale', 1.0)
        
        # Create a rectangular approximation for the block
        width = 100 * scale_x  # Default width
        height = 100 * scale_y  # Default height
        
        return Polygon([
            (x - width/2, y - height/2),
            (x + width/2, y - height/2),
            (x + width/2, y + height/2),
            (x - width/2, y + height/2)
        ])
    
    def segment_rooms(self):
        """Segment the floor plan into individual rooms."""
        if not self.walls:
            print("No walls found for room segmentation")
            return
        
        # Convert walls to lines and create a network
        wall_lines = []
        for wall in self.walls:
            if wall and hasattr(wall, 'coords'):
                wall_lines.append(wall)
        
        # Create polygons from the wall network
        try:
            # Union all wall lines
            wall_union = unary_union(wall_lines)
            
            # Get the polygons formed by the walls
            if hasattr(wall_union, 'geoms'):
                lines = list(wall_union.geoms)
            else:
                lines = [wall_union]
            
            # Use polygonize to create room polygons
            room_polygons = list(polygonize(lines))
            
            # Filter out very small polygons (likely not rooms)
            min_room_area = 1000  # Minimum area in square units
            self.rooms = [room for room in room_polygons if room.area > min_room_area]
            
            print(f"Found {len(self.rooms)} potential rooms")
            
        except Exception as e:
            print(f"Error in room segmentation: {e}")
            # Fallback: create bounding box as single room
            if wall_lines:
                all_coords = []
                for line in wall_lines:
                    all_coords.extend(list(line.coords))
                
                if all_coords:
                    xs, ys = zip(*all_coords)
                    bbox = Polygon([
                        (min(xs), min(ys)),
                        (max(xs), min(ys)),
                        (max(xs), max(ys)),
                        (min(xs), max(ys))
                    ])
                    self.rooms = [bbox]
    
    def classify_rooms(self) -> Dict[str, float]:
        """Classify rooms and calculate their areas."""
        if not self.rooms:
            return {}
        
        room_classifications = {}
        
        for i, room in enumerate(self.rooms):
            room_id = f"Room_{i+1}"
            
            # Calculate area in square meters (assuming DXF units are mm)
            area_mm2 = room.area
            area_m2 = area_mm2 / 1_000_000  # Convert mm² to m²
            
            # Simple room classification based on area and furniture
            room_type = self._classify_room_type(room, area_m2)
            
            # Remove furniture area from room area
            usable_area = self._calculate_usable_area(room)
            usable_area_m2 = usable_area / 1_000_000
            
            room_classifications[f"{room_type}_{room_id}"] = {
                'total_area_m2': round(area_m2, 2),
                'usable_area_m2': round(usable_area_m2, 2),
                'furniture_area_m2': round((area_m2 - usable_area_m2), 2),
                'geometry': room
            }
        
        self.room_areas = room_classifications
        return room_classifications
    
    def _classify_room_type(self, room_polygon: Polygon, area_m2: float) -> str:
        """Classify room type based on area and furniture."""
        furniture_in_room = []
        
        # Check which furniture items are in this room
        for furniture in self.furniture:
            if furniture and room_polygon.contains(furniture.centroid):
                furniture_in_room.append(furniture)
        
        # Classification logic
        if area_m2 < 5:
            return "Closet"
        elif area_m2 < 8:
            # Check for bathroom fixtures
            return "Bathroom"
        elif area_m2 < 15:
            if len(furniture_in_room) > 3:
                return "Bedroom"
            else:
                return "Office"
        elif area_m2 < 30:
            return "Bedroom" if len(furniture_in_room) > 2 else "Living_Room"
        else:
            return "Living_Room"
    
    def _calculate_usable_area(self, room_polygon: Polygon) -> float:
        """Calculate usable area by subtracting furniture area."""
        furniture_area = 0
        
        for furniture in self.furniture:
            if furniture and room_polygon.intersects(furniture):
                intersection = room_polygon.intersection(furniture)
                furniture_area += intersection.area
        
        return max(0, room_polygon.area - furniture_area)
    
    def generate_report(self) -> str:
        """Generate a detailed report of the analysis."""
        if not self.room_areas:
            return "No room analysis available. Please run the analysis first."
        
        report = []
        report.append("=" * 60)
        report.append("DXF ROOM ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"File: {self.dxf_file_path}")
        report.append(f"Total rooms found: {len(self.room_areas)}")
        report.append("")
        
        total_area = 0
        total_usable_area = 0
        total_furniture_area = 0
        
        for room_name, room_data in self.room_areas.items():
            report.append(f"Room: {room_name}")
            report.append(f"  Total Area: {room_data['total_area_m2']:.2f} m²")
            report.append(f"  Usable Area: {room_data['usable_area_m2']:.2f} m²")
            report.append(f"  Furniture Area: {room_data['furniture_area_m2']:.2f} m²")
            report.append(f"  Furniture Coverage: {(room_data['furniture_area_m2']/room_data['total_area_m2']*100):.1f}%")
            report.append("")
            
            total_area += room_data['total_area_m2']
            total_usable_area += room_data['usable_area_m2']
            total_furniture_area += room_data['furniture_area_m2']
        
        report.append("-" * 40)
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Floor Area: {total_area:.2f} m²")
        report.append(f"Total Usable Area: {total_usable_area:.2f} m²")
        report.append(f"Total Furniture Area: {total_furniture_area:.2f} m²")
        report.append(f"Overall Furniture Coverage: {(total_furniture_area/total_area*100):.1f}%")
        
        return "\n".join(report)
    
    def visualize_analysis(self, save_path: Optional[str] = None):
        """Create a visualization of the room analysis."""
        if not self.rooms:
            print("No rooms to visualize")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Plot walls
        for wall in self.walls:
            if wall and hasattr(wall, 'coords'):
                x, y = wall.xy
                ax.plot(x, y, 'k-', linewidth=2, label='Walls' if wall == self.walls[0] else "")
        
        # Plot rooms with different colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.rooms)))
        for i, (room, color) in enumerate(zip(self.rooms, colors)):
            x, y = room.exterior.xy
            ax.fill(x, y, color=color, alpha=0.3, label=f'Room {i+1}')
            
            # Add room label
            centroid = room.centroid
            ax.text(centroid.x, centroid.y, f'R{i+1}', ha='center', va='center', 
                   fontsize=12, fontweight='bold')
        
        # Plot furniture
        for furniture in self.furniture:
            if furniture:
                if hasattr(furniture, 'exterior'):
                    x, y = furniture.exterior.xy
                    ax.fill(x, y, color='brown', alpha=0.7)
                else:
                    ax.plot(furniture.x, furniture.y, 'ro', markersize=8)
        
        # Plot doors and windows
        for door in self.doors:
            if door and hasattr(door, 'coords'):
                x, y = door.xy
                ax.plot(x, y, 'g-', linewidth=4, label='Doors' if door == self.doors[0] else "")
        
        for window in self.windows:
            if window and hasattr(window, 'coords'):
                x, y = window.xy
                ax.plot(x, y, 'b-', linewidth=4, label='Windows' if window == self.windows[0] else "")
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_title('Room Segmentation Analysis')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def analyze(self) -> Dict[str, float]:
        """Run the complete analysis pipeline."""
        print("Starting DXF analysis...")
        
        if not self.load_dxf():
            return {}
        
        print("Extracting entities...")
        self.extract_entities()
        
        print("Segmenting rooms...")
        self.segment_rooms()
        
        print("Classifying rooms and calculating areas...")
        room_data = self.classify_rooms()
        
        print("Analysis complete!")
        return room_data


def main():
    """Example usage of the DXF Room Analyzer."""
    # Example usage
    dxf_file = "floor_plan.dxf"  # Replace with your DXF file path
    
    analyzer = DXFRoomAnalyzer(dxf_file)
    
    # Run the analysis
    room_data = analyzer.analyze()
    
    # Generate and print report
    report = analyzer.generate_report()
    print(report)
    
    # Create visualization
    analyzer.visualize_analysis("room_analysis.png")
    
    # Return room areas for further processing
    return room_data


if __name__ == "__main__":
    # Install required packages:
    # pip install ezdxf shapely matplotlib numpy
    
    main()