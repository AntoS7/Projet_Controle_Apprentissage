"""
Network generator for SUMO simulation.

This module creates the road network with three interconnected intersections
and generates the necessary SUMO network files.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
from typing import List, Tuple


class SumoNetworkGenerator:
    """Generates SUMO network files for three connected intersections."""
    
    def __init__(self, output_dir: str = "config"):
        """
        Initialize network generator.
        
        Args:
            output_dir: Directory to save generated files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_nodes_file(self) -> str:
        """Generate nodes file for the network."""
        nodes_file = os.path.join(self.output_dir, "network.nod.xml")
        
        # Create root element
        root = ET.Element("nodes")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/nodes_file.xsd")
        
        # Define node positions for three intersections in a triangle
        nodes = [
            {"id": "intersection_1", "x": "0", "y": "0", "type": "traffic_light"},
            {"id": "intersection_2", "x": "500", "y": "0", "type": "traffic_light"},
            {"id": "intersection_3", "x": "250", "y": "433", "type": "traffic_light"},
            
            # External nodes for traffic generation
            {"id": "north_1", "x": "0", "y": "200", "type": "priority"},
            {"id": "south_1", "x": "0", "y": "-200", "type": "priority"},
            {"id": "east_1", "x": "-200", "y": "0", "type": "priority"},
            {"id": "west_1", "x": "200", "y": "0", "type": "priority"},
            
            {"id": "north_2", "x": "500", "y": "200", "type": "priority"},
            {"id": "south_2", "x": "500", "y": "-200", "type": "priority"},
            {"id": "east_2", "x": "300", "y": "0", "type": "priority"},
            {"id": "west_2", "x": "700", "y": "0", "type": "priority"},
            
            {"id": "north_3", "x": "250", "y": "633", "type": "priority"},
            {"id": "south_3", "x": "250", "y": "233", "type": "priority"},
            {"id": "east_3", "x": "50", "y": "433", "type": "priority"},
            {"id": "west_3", "x": "450", "y": "433", "type": "priority"},
        ]
        
        # Add nodes to XML
        for node in nodes:
            node_elem = ET.SubElement(root, "node")
            for key, value in node.items():
                node_elem.set(key, value)
        
        # Write to file
        self._write_xml_file(root, nodes_file)
        return nodes_file
    
    def generate_edges_file(self) -> str:
        """Generate edges file for the network."""
        edges_file = os.path.join(self.output_dir, "network.edg.xml")
        
        # Create root element
        root = ET.Element("edges")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/edges_file.xsd")
        
        # Define edges (roads)
        edges = [
            # Intersection 1 connections
            {"id": "north_to_1", "from": "north_1", "to": "intersection_1", "numLanes": "2", "speed": "13.89"},
            {"id": "1_to_north", "from": "intersection_1", "to": "north_1", "numLanes": "2", "speed": "13.89"},
            {"id": "south_to_1", "from": "south_1", "to": "intersection_1", "numLanes": "2", "speed": "13.89"},
            {"id": "1_to_south", "from": "intersection_1", "to": "south_1", "numLanes": "2", "speed": "13.89"},
            {"id": "east_to_1", "from": "east_1", "to": "intersection_1", "numLanes": "2", "speed": "13.89"},
            {"id": "1_to_east", "from": "intersection_1", "to": "east_1", "numLanes": "2", "speed": "13.89"},
            
            # Intersection 2 connections  
            {"id": "north_to_2", "from": "north_2", "to": "intersection_2", "numLanes": "2", "speed": "13.89"},
            {"id": "2_to_north", "from": "intersection_2", "to": "north_2", "numLanes": "2", "speed": "13.89"},
            {"id": "south_to_2", "from": "south_2", "to": "intersection_2", "numLanes": "2", "speed": "13.89"},
            {"id": "2_to_south", "from": "intersection_2", "to": "south_2", "numLanes": "2", "speed": "13.89"},
            {"id": "west_to_2", "from": "west_2", "to": "intersection_2", "numLanes": "2", "speed": "13.89"},
            {"id": "2_to_west", "from": "intersection_2", "to": "west_2", "numLanes": "2", "speed": "13.89"},
            
            # Intersection 3 connections
            {"id": "north_to_3", "from": "north_3", "to": "intersection_3", "numLanes": "2", "speed": "13.89"},
            {"id": "3_to_north", "from": "intersection_3", "to": "north_3", "numLanes": "2", "speed": "13.89"},
            {"id": "south_to_3", "from": "south_3", "to": "intersection_3", "numLanes": "2", "speed": "13.89"},
            {"id": "3_to_south", "from": "intersection_3", "to": "south_3", "numLanes": "2", "speed": "13.89"},
            {"id": "east_to_3", "from": "east_3", "to": "intersection_3", "numLanes": "2", "speed": "13.89"},
            {"id": "3_to_east", "from": "intersection_3", "to": "east_3", "numLanes": "2", "speed": "13.89"},
            {"id": "west_to_3", "from": "west_3", "to": "intersection_3", "numLanes": "2", "speed": "13.89"},
            {"id": "3_to_west", "from": "intersection_3", "to": "west_3", "numLanes": "2", "speed": "13.89"},
            
            # Inter-intersection connections
            {"id": "1_to_2", "from": "intersection_1", "to": "intersection_2", "numLanes": "2", "speed": "13.89"},
            {"id": "2_to_1", "from": "intersection_2", "to": "intersection_1", "numLanes": "2", "speed": "13.89"},
            {"id": "1_to_3", "from": "intersection_1", "to": "intersection_3", "numLanes": "2", "speed": "13.89"},
            {"id": "3_to_1", "from": "intersection_3", "to": "intersection_1", "numLanes": "2", "speed": "13.89"},
            {"id": "2_to_3", "from": "intersection_2", "to": "intersection_3", "numLanes": "2", "speed": "13.89"},
            {"id": "3_to_2", "from": "intersection_3", "to": "intersection_2", "numLanes": "2", "speed": "13.89"},
        ]
        
        # Add edges to XML
        for edge in edges:
            edge_elem = ET.SubElement(root, "edge")
            for key, value in edge.items():
                edge_elem.set(key, value)
        
        # Write to file
        self._write_xml_file(root, edges_file)
        return edges_file
    
    def generate_traffic_lights_file(self) -> str:
        """Generate traffic lights configuration file."""
        tls_file = os.path.join(self.output_dir, "traffic_lights.add.xml")
        
        # Create root element
        root = ET.Element("additionalFile")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/additional_file.xsd")
        
        # Define traffic light programs for each intersection
        intersections = ["intersection_1", "intersection_2", "intersection_3"]
        
        for intersection_id in intersections:
            # Traffic light logic
            tl_logic = ET.SubElement(root, "tlLogic")
            tl_logic.set("id", intersection_id)
            tl_logic.set("type", "static")
            tl_logic.set("programID", "0")
            tl_logic.set("offset", "0")
            
            # Define phases
            phases = [
                {"duration": "30", "state": "GGrrGGrr"},  # North-South green
                {"duration": "5", "state": "yyrryyRr"},   # North-South yellow
                {"duration": "30", "state": "rrGGrrGG"},  # East-West green  
                {"duration": "5", "state": "rryyrryy"},   # East-West yellow
            ]
            
            for phase in phases:
                phase_elem = ET.SubElement(tl_logic, "phase")
                for key, value in phase.items():
                    phase_elem.set(key, value)
        
        # Write to file
        self._write_xml_file(root, tls_file)
        return tls_file
    
    def generate_network(self) -> str:
        """Generate complete SUMO network."""
        # Generate individual files
        nodes_file = self.generate_nodes_file()
        edges_file = self.generate_edges_file()
        tls_file = self.generate_traffic_lights_file()
        
        # Generate network using netconvert
        network_file = os.path.join(self.output_dir, "network.net.xml")
        
        netconvert_cmd = [
            "/opt/homebrew/opt/sumo/bin/netconvert",
            "--node-files", nodes_file,
            "--edge-files", edges_file,
            "--output-file", network_file,
            "--tls.guess-signals", "true",
            "--tls.cycle.time", "70"
        ]
        
        try:
            import subprocess
            # Verify netconvert exists
            if not os.path.exists("/opt/homebrew/opt/sumo/bin/netconvert"):
                raise FileNotFoundError("netconvert not found at /opt/homebrew/opt/sumo/bin/netconvert")
                
            result = subprocess.run(netconvert_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Warning: netconvert failed: {result.stderr}")
                print("You may need to install SUMO and ensure netconvert is in your PATH")
        except FileNotFoundError:
            print("Warning: netconvert not found. Please install SUMO.")
            print("You can manually run: " + " ".join(netconvert_cmd))
        
        return network_file
    
    def _write_xml_file(self, root: ET.Element, filename: str):
        """Write XML element to file with pretty formatting."""
        xml_str = ET.tostring(root, encoding='unicode')
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="    ")
        
        # Remove extra blank lines
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        pretty_xml = '\n'.join(lines)
        
        with open(filename, 'w') as f:
            f.write(pretty_xml)


def generate_sumo_network(output_dir: str = "config") -> str:
    """
    Convenience function to generate complete SUMO network.
    
    Args:
        output_dir: Directory to save network files
        
    Returns:
        Path to generated network file
    """
    generator = SumoNetworkGenerator(output_dir)
    return generator.generate_network()


if __name__ == "__main__":
    # Generate network when run as script
    network_file = generate_sumo_network()
    print(f"Generated network file: {network_file}")
