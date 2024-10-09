import numpy as np
from skimage import measure, filters

def find_perimeter_and_uniform_points(depth_array, num_points):
    """
    This function takes a 2D Numpy array representing a depth image, finds the perimeter
    of the object, and returns num_points uniformly distributed along this perimeter.
   
    Parameters:
    depth_array (numpy.ndarray): The input depth image as a 2D Numpy array.
    num_points (int): The number of points to distribute along the perimeter.
   
    Returns:
    numpy.ndarray: An array of points with uniform distribution along the perimeter.
    """

    # Apply edge detection using Otsu's method to binarize the depth image
    threshold_value = filters.threshold_otsu(depth_array)
    binary_image = depth_array > threshold_value

    # Find contours at a constant value
    contours = measure.find_contours(binary_image, 0.8)

    # Assuming the largest contour is our object of interest
    largest_contour = max(contours, key=lambda x: x.shape[0])

    # Function to resample the contour points to get a uniform distribution
    def resample_contour(contour, num_points):
        # Calculate the cumulative distance along the contour
        distance = np.cumsum(np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1)))
        distance = np.insert(distance, 0, 0)  # Insert 0 at the beginning
        final_distance = distance[-1]

        # Create an array of evenly spaced distance values
        uniform_distance = np.linspace(0, final_distance, num_points)

        # Interpolate the contour points to these evenly spaced values
        uniform_contour_x = np.interp(uniform_distance, distance, contour[:, 0])
        uniform_contour_y = np.interp(uniform_distance, distance, contour[:, 1])

        # Stack the coordinates to get the points
        uniform_points = np.vstack((uniform_contour_x, uniform_contour_y)).T
        return uniform_points

    # Get uniformly distributed points along the contour
    uniform_points = resample_contour(largest_contour, num_points)
   
    return uniform_points