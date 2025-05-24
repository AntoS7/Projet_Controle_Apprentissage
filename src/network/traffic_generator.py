"""
Traffic generator for SUMO simulation.

This module generates traffic flows and routes for the three-intersection network.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import random
from typing import List, Dict


class TrafficGenerator:
    """Generates traffic flows and routes for SUMO simulation."""
    
    def __init__(self, output_dir: str = "config"):
        """
        Initialize traffic generator.
        
        Args:
            output_dir: Directory to save generated files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_routes_file(self, simulation_duration: int = 3600) -> str:
        """
        Generate routes and traffic flows.
        
        Args:
            simulation_duration: Duration of simulation in seconds
            
        Returns:
            Path to generated routes file
        """
        routes_file = os.path.join(self.output_dir, "routes.rou.xml")
        
        # Create root element
        root = ET.Element("routes")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")
        
        # Define vehicle types
        vtype = ET.SubElement(root, "vType")
        vtype.set("id", "car")
        vtype.set("accel", "2.6")
        vtype.set("decel", "4.5")
        vtype.set("sigma", "0.5")
        vtype.set("length", "5.0")
        vtype.set("maxSpeed", "50.0")
        
        # Define routes through the network
        routes = self._generate_routes()
        
        # Add routes to XML
        for route_id, edges in routes.items():
            route_elem = ET.SubElement(root, "route")
            route_elem.set("id", route_id)
            route_elem.set("edges", " ".join(edges))
        
        # Generate traffic flows
        flows = self._generate_traffic_flows(simulation_duration)
        
        # Add flows to XML
        for flow in flows:
            flow_elem = ET.SubElement(root, "flow")
            for key, value in flow.items():
                flow_elem.set(key, value)
        
        # Write to file
        self._write_xml_file(root, routes_file)
        return routes_file
    
    def _generate_routes(self) -> Dict[str, List[str]]:
        """Generate different routes through the network."""
        routes = {
            # Routes through intersection 1
            "route_n1_s1": ["north_to_1", "1_to_south"],
            "route_s1_n1": ["south_to_1", "1_to_north"], 
            "route_e1_w1": ["east_to_1", "1_to_east"],
            "route_n1_2": ["north_to_1", "1_to_2"],
            "route_s1_2": ["south_to_1", "1_to_2"],
            "route_e1_2": ["east_to_1", "1_to_2"],
            "route_n1_3": ["north_to_1", "1_to_3"],
            "route_s1_3": ["south_to_1", "1_to_3"],
            "route_e1_3": ["east_to_1", "1_to_3"],
            
            # Routes through intersection 2
            "route_n2_s2": ["north_to_2", "2_to_south"],
            "route_s2_n2": ["south_to_2", "2_to_north"],
            "route_w2_e2": ["west_to_2", "2_to_west"],
            "route_n2_1": ["north_to_2", "2_to_1"],
            "route_s2_1": ["south_to_2", "2_to_1"],
            "route_w2_1": ["west_to_2", "2_to_1"],
            "route_n2_3": ["north_to_2", "2_to_3"],
            "route_s2_3": ["south_to_2", "2_to_3"],
            "route_w2_3": ["west_to_2", "2_to_3"],
            
            # Routes through intersection 3
            "route_n3_s3": ["north_to_3", "3_to_south"],
            "route_s3_n3": ["south_to_3", "3_to_north"],
            "route_e3_w3": ["east_to_3", "3_to_west"],
            "route_w3_e3": ["west_to_3", "3_to_east"],
            "route_n3_1": ["north_to_3", "3_to_1"],
            "route_s3_1": ["south_to_3", "3_to_1"],
            "route_e3_1": ["east_to_3", "3_to_1"],
            "route_w3_1": ["west_to_3", "3_to_1"],
            "route_n3_2": ["north_to_3", "3_to_2"],
            "route_s3_2": ["south_to_3", "3_to_2"],
            "route_e3_2": ["east_to_3", "3_to_2"],
            "route_w3_2": ["west_to_3", "3_to_2"],
            
            # Inter-intersection routes
            "route_1_2_n": ["1_to_2", "2_to_north"],
            "route_1_2_s": ["1_to_2", "2_to_south"],
            "route_1_2_w": ["1_to_2", "2_to_west"],
            "route_2_1_n": ["2_to_1", "1_to_north"],
            "route_2_1_s": ["2_to_1", "1_to_south"],
            "route_2_1_e": ["2_to_1", "1_to_east"],
            "route_1_3_n": ["1_to_3", "3_to_north"],
            "route_1_3_s": ["1_to_3", "3_to_south"],
            "route_1_3_e": ["1_to_3", "3_to_east"],
            "route_1_3_w": ["1_to_3", "3_to_west"],
            "route_3_1_n": ["3_to_1", "1_to_north"],
            "route_3_1_s": ["3_to_1", "1_to_south"],
            "route_3_1_e": ["3_to_1", "1_to_east"],
            "route_2_3_n": ["2_to_3", "3_to_north"],
            "route_2_3_s": ["2_to_3", "3_to_south"],
            "route_2_3_e": ["2_to_3", "3_to_east"],
            "route_2_3_w": ["2_to_3", "3_to_west"],
            "route_3_2_n": ["3_to_2", "2_to_north"],
            "route_3_2_s": ["3_to_2", "2_to_south"],
            "route_3_2_w": ["3_to_2", "2_to_west"],
        }
        
        return routes
    
    def _generate_traffic_flows(self, duration: int) -> List[Dict[str, str]]:
        """Generate traffic flows with varying intensities."""
        flows = []
        
        # Base flow parameters
        base_flows = [
            # High traffic routes (main arterials)
            {"route": "route_n1_s1", "vehsPerHour": "400", "probability": "0.7"},
            {"route": "route_s1_n1", "vehsPerHour": "380", "probability": "0.7"},
            {"route": "route_n2_s2", "vehsPerHour": "420", "probability": "0.7"},
            {"route": "route_s2_n2", "vehsPerHour": "390", "probability": "0.7"},
            {"route": "route_n3_s3", "vehsPerHour": "350", "probability": "0.6"},
            {"route": "route_s3_n3", "vehsPerHour": "360", "probability": "0.6"},
            
            # Medium traffic routes (cross streets)
            {"route": "route_e1_w1", "vehsPerHour": "250", "probability": "0.5"},
            {"route": "route_w2_e2", "vehsPerHour": "270", "probability": "0.5"},
            {"route": "route_e3_w3", "vehsPerHour": "230", "probability": "0.5"},
            {"route": "route_w3_e3", "vehsPerHour": "240", "probability": "0.5"},
            
            # Inter-intersection traffic
            {"route": "route_1_2_n", "vehsPerHour": "180", "probability": "0.4"},
            {"route": "route_1_2_s", "vehsPerHour": "170", "probability": "0.4"},
            {"route": "route_2_1_n", "vehsPerHour": "190", "probability": "0.4"},
            {"route": "route_2_1_s", "vehsPerHour": "175", "probability": "0.4"},
            {"route": "route_1_3_n", "vehsPerHour": "160", "probability": "0.4"},
            {"route": "route_1_3_e", "vehsPerHour": "150", "probability": "0.4"},
            {"route": "route_3_1_n", "vehsPerHour": "165", "probability": "0.4"},
            {"route": "route_3_1_s", "vehsPerHour": "155", "probability": "0.4"},
            {"route": "route_2_3_e", "vehsPerHour": "140", "probability": "0.3"},
            {"route": "route_2_3_w", "vehsPerHour": "145", "probability": "0.3"},
            {"route": "route_3_2_n", "vehsPerHour": "148", "probability": "0.3"},
            {"route": "route_3_2_s", "vehsPerHour": "142", "probability": "0.3"},
        ]
        
        # Create time-varying flows
        time_periods = [
            {"start": 0, "end": 900, "multiplier": 0.6},      # Low traffic (0-15 min)
            {"start": 900, "end": 1800, "multiplier": 1.2},   # Morning rush (15-30 min)
            {"start": 1800, "end": 2700, "multiplier": 0.8},  # Normal traffic (30-45 min)
            {"start": 2700, "end": 3600, "multiplier": 1.0},  # Evening traffic (45-60 min)
        ]
        
        flow_id = 0
        for period in time_periods:
            for base_flow in base_flows:
                flow_id += 1
                adjusted_flow = int(int(base_flow["vehsPerHour"]) * period["multiplier"])
                
                flow = {
                    "id": f"flow_{flow_id}",
                    "route": base_flow["route"],
                    "begin": str(period["start"]),
                    "end": str(period["end"]),
                    "vehsPerHour": str(adjusted_flow),
                    "type": "car"
                }
                flows.append(flow)
        
        return flows
    
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


def generate_traffic_flows(output_dir: str = "config", duration: int = 3600) -> str:
    """
    Convenience function to generate traffic flows.
    
    Args:
        output_dir: Directory to save files
        duration: Simulation duration in seconds
        
    Returns:
        Path to generated routes file
    """
    generator = TrafficGenerator(output_dir)
    return generator.generate_routes_file(duration)


if __name__ == "__main__":
    # Generate traffic when run as script
    routes_file = generate_traffic_flows()
    print(f"Generated routes file: {routes_file}")
