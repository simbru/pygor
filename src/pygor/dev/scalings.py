import numpy as np
import math

def cycles_per_degree_to_degrees_visual_angle(cpd):
    """
    Convert cycles per degree to degrees of visual angle per cycle.
    
    Args:
        cpd (float or array): Cycles per degree
        
    Returns:
        float or array: Degrees of visual angle per cycle
    """
    return 1.0 / cpd

def degrees_visual_angle_to_cycles_per_degree(deg_vis_angle):
    """
    Convert degrees of visual angle per cycle to cycles per degree.
    
    Args:
        deg_vis_angle (float or array): Degrees of visual angle per cycle
        
    Returns:
        float or array: Cycles per degree
    """
    return 1.0 / deg_vis_angle

def microns_to_degrees_visual_angle(micron_diameter, viewing_distance_mm):
    """
    Convert micron diameter to degrees of visual angle.
    
    Args:
        micron_diameter (float or array): Diameter in microns
        viewing_distance_mm (float): Viewing distance in millimeters
        
    Returns:
        float or array: Degrees of visual angle
    """
    # Convert microns to mm
    diameter_mm = micron_diameter / 1000.0
    
    # Calculate visual angle in radians using small angle approximation
    # For small angles: tan(θ) ≈ θ, so θ = diameter / distance
    angle_radians = diameter_mm / viewing_distance_mm
    
    # Convert radians to degrees
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

def degrees_visual_angle_to_microns(deg_vis_angle, viewing_distance_mm):
    """
    Convert degrees of visual angle to micron diameter.
    
    Args:
        deg_vis_angle (float or array): Degrees of visual angle
        viewing_distance_mm (float): Viewing distance in millimeters
        
    Returns:
        float or array: Diameter in microns
    """
    # Convert degrees to radians
    angle_radians = np.radians(deg_vis_angle)
    
    # Calculate diameter in mm using small angle approximation
    diameter_mm = angle_radians * viewing_distance_mm
    
    # Convert mm to microns
    diameter_microns = diameter_mm * 1000.0
    
    return diameter_microns

def microns_to_cycles_per_degree(micron_diameter, viewing_distance_mm):
    """
    Convert micron diameter to cycles per degree.
    This assumes the micron diameter represents one cycle of a grating.
    
    Args:
        micron_diameter (float or array): Diameter in microns (one cycle)
        viewing_distance_mm (float): Viewing distance in millimeters
        
    Returns:
        float or array: Cycles per degree
    """
    # First convert to degrees visual angle
    deg_vis_angle = microns_to_degrees_visual_angle(micron_diameter, viewing_distance_mm)
    
    # Then convert to cycles per degree
    cpd = degrees_visual_angle_to_cycles_per_degree(deg_vis_angle)
    
    return cpd

def cycles_per_degree_to_microns(cpd, viewing_distance_mm):
    """
    Convert cycles per degree to micron diameter.
    This gives the diameter of one cycle in microns.
    
    Args:
        cpd (float or array): Cycles per degree
        viewing_distance_mm (float): Viewing distance in millimeters
        
    Returns:
        float or array: Diameter in microns (one cycle)
    """
    # First convert to degrees visual angle
    deg_vis_angle = cycles_per_degree_to_degrees_visual_angle(cpd)
    
    # Then convert to microns
    micron_diameter = degrees_visual_angle_to_microns(deg_vis_angle, viewing_distance_mm)
    
    return micron_diameter

def retinal_microns_to_degrees_visual_angle(retinal_microns, eye_radius_mm=None, species='zebrafish'):
    """
    Convert retinal distance in microns to degrees of visual angle.
    Uses species-specific eye parameters if not provided.
    
    Args:
        retinal_microns (float or array): Distance on retina in microns
        eye_radius_mm (float, optional): Eye radius in mm. If None, uses species default.
        species (str): Species name for default eye parameters
        
    Returns:
        float or array: Degrees of visual angle
    """
    # Default eye radii for common model organisms (approximate values)
    default_eye_radii = {
        'zebrafish': 0.075,  # larval zebrafish ~0.15mm diameter (5-7 dpf)
        'adult_zebrafish': 0.75,  # adult zebrafish ~1.5mm diameter
        'mouse': 1.6,       # ~3.2mm diameter  
        'human': 12.0,      # ~24mm diameter
        'macaque': 11.0,    # ~22mm diameter
        'rat': 2.0          # ~4mm diameter
    }
    
    if eye_radius_mm is None:
        if species.lower() in default_eye_radii:
            eye_radius_mm = default_eye_radii[species.lower()]
        else:
            raise ValueError(f"Unknown species '{species}'. Please provide eye_radius_mm manually.")
    
    # Convert microns to mm
    retinal_mm = retinal_microns / 1000.0
    
    # Calculate visual angle in radians
    # For small angles on a sphere: angle = arc_length / radius
    angle_radians = retinal_mm / eye_radius_mm
    
    # Convert to degrees
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

def degrees_visual_angle_to_retinal_microns(deg_vis_angle, eye_radius_mm=None, species='zebrafish'):
    """
    Convert degrees of visual angle to retinal distance in microns.
    Uses species-specific eye parameters if not provided.
    
    Args:
        deg_vis_angle (float or array): Degrees of visual angle
        eye_radius_mm (float, optional): Eye radius in mm. If None, uses species default.
        species (str): Species name for default eye parameters
        
    Returns:
        float or array: Distance on retina in microns
    """
    # Default eye radii for common model organisms
    default_eye_radii = {
        'zebrafish': 0.075,  # larval zebrafish ~0.15mm diameter (5-7 dpf)
        'adult_zebrafish': 0.75,  # adult zebrafish ~1.5mm diameter
        'mouse': 1.6,
        'human': 12.0,
        'macaque': 11.0,
        'rat': 2.0
    }
    
    if eye_radius_mm is None:
        if species.lower() in default_eye_radii:
            eye_radius_mm = default_eye_radii[species.lower()]
        else:
            raise ValueError(f"Unknown species '{species}'. Please provide eye_radius_mm manually.")
    
    # Convert degrees to radians
    angle_radians = np.radians(deg_vis_angle)
    
    # Calculate retinal distance in mm
    retinal_mm = angle_radians * eye_radius_mm
    
    # Convert to microns
    retinal_microns = retinal_mm * 1000.0
    
    return retinal_microns

# Example usage and validation functions
def validate_conversions():
    """
    Validate the conversion functions with known values.
    """
    print("Validation Tests:")
    print("=" * 50)
    
    # Test 1: Round trip conversion
    original_cpd = 0.116  # From the zebrafish paper
    deg_angle = cycles_per_degree_to_degrees_visual_angle(original_cpd)
    back_to_cpd = degrees_visual_angle_to_cycles_per_degree(deg_angle)
    print(f"Round trip test - Original: {original_cpd:.3f} c/d, Back: {back_to_cpd:.3f} c/d")
    
    # Test 2: Known conversion
    cpd_1 = 1.0  # 1 cycle per degree
    deg_angle_1 = cycles_per_degree_to_degrees_visual_angle(cpd_1)
    print(f"1 c/d = {deg_angle_1:.1f} degrees visual angle")
    
    # Test 3: Micron conversion (assuming 50mm viewing distance)
    viewing_dist = 50.0  # mm
    micron_size = 1000.0  # 1mm = 1000 microns
    deg_from_microns = microns_to_degrees_visual_angle(micron_size, viewing_dist)
    print(f"1000 microns at 50mm = {deg_from_microns:.2f} degrees visual angle")
    
    # Test 4: Retinal conversion for zebrafish
    retinal_distance = 100.0  # microns on retina
    visual_angle = retinal_microns_to_degrees_visual_angle(retinal_distance, species='zebrafish')
    print(f"100 microns on zebrafish retina = {visual_angle:.2f} degrees visual angle")

def compare_with_zebrafish_paper():
    """
    Compare our calculations with values from the zebrafish paper.
    """
    print("\nComparison with Zebrafish Paper:")
    print("=" * 50)
    
    # Paper reports optimal spatial frequencies
    photopic_cpd = 0.116  # c/d at photopic levels
    scotopic_cpd = 0.110  # c/d at scotopic levels
    
    # Convert to degrees visual angle
    photopic_deg = cycles_per_degree_to_degrees_visual_angle(photopic_cpd)
    scotopic_deg = cycles_per_degree_to_degrees_visual_angle(scotopic_cpd)
    
    print(f"Photopic optimal: {photopic_cpd:.3f} c/d = {photopic_deg:.2f} deg visual angle")
    print(f"Scotopic optimal: {scotopic_cpd:.3f} c/d = {scotopic_deg:.2f} deg visual angle")
    
    # Convert to retinal distances (assuming zebrafish eye)
    photopic_retinal = degrees_visual_angle_to_retinal_microns(photopic_deg, species='zebrafish')
    scotopic_retinal = degrees_visual_angle_to_retinal_microns(scotopic_deg, species='zebrafish')
    
    print(f"Photopic retinal spacing: {photopic_retinal:.0f} microns")
    print(f"Scotopic retinal spacing: {scotopic_retinal:.0f} microns")

if __name__ == "__main__":
    validate_conversions()
    compare_with_zebrafish_paper()