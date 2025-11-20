def Crop_Microglia(input_image_path, output_folder, padding=20, min_size_microns=100):
    """
    Takes an input image ending in _C_0.tiff, applies a mean filter, and crops the image
    around each individual thresholded object. Only crops objects that meet a minimum size
    determined by the user. If more than 5 objects exist, randomly selects 5 to process.
    Skips objects touching the edges of the image. Saves each cropped object as a separate
    image in the specified output folder with a _#_ added before the _C_0 in the filename.
    """
    import cv2
    import os
    import numpy as np
    import random
    from skimage.measure import regionprops, label as skimage_label
    
    # Scale for converting microns² to pixels
    scale = 0.641  # microns per pixel
    min_size_pixels = int(min_size_microns / (scale ** 2))

    # Load the input _C_0 image
    image_c0 = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if image_c0 is None:
        raise ValueError(f"Error: Unable to load image at {input_image_path}")

    # Apply a mean filter (blur the _C_0 image)
    blurred_image = cv2.blur(image_c0, (5, 5))

    # Apply thresholding to the blurred image
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Label connected components in the binary image
    labeled_image = skimage_label(binary_image)

    # Get properties of labeled regions
    props = regionprops(labeled_image)

    # Filter out objects that do not meet the minimum size
    filtered_props = [region for region in props if region.area >= min_size_pixels]

    # Randomly select up to 5 objects if more than 5 exist
    if len(filtered_props) > 5:
        filtered_props = random.sample(filtered_props, 5)
        print(f"Randomly selected 5 objects out of {len(props)} total objects.")

    # Get the base filename of the input image
    base_filename_c0 = os.path.basename(input_image_path)

    # Iterate through the selected regions and crop the images
    for i, region in enumerate(filtered_props):
        min_row, min_col, max_row, max_col = region.bbox

        # Skip objects touching the edges of the image
        if min_row == 0 or min_col == 0 or max_row == image_c0.shape[0] or max_col == image_c0.shape[1]:
            print(f"Skipping object {i + 1} - touching the edge of the image.")
            continue

        # Add padding to the bounding box
        min_row = max(min_row - padding, 0)
        min_col = max(min_col - padding, 0)
        max_row = min(max_row + padding, image_c0.shape[0])
        max_col = min(max_col + padding, image_c0.shape[1])

        # Crop the _C_0 image around the padded bounding box
        cropped_image_c0 = image_c0[min_row:max_row, min_col:max_col]

        # Construct the new filename with _#_ added before _C_0
        new_filename_c0 = base_filename_c0.replace("_C_0", f"_{i + 1}_C_0")

        # Save the cropped _C_0 image in the output folder
        output_path_c0 = os.path.join(output_folder, new_filename_c0)
        cv2.imwrite(output_path_c0, cropped_image_c0)
        print(f"Saved cropped _C_0 image: {output_path_c0}")


def Isolate_Microglia(input_folder, padding=20, min_size_microns=100):
    """
    Processes all images in a folder that end with _C_0.tiff for microglia (single-channel).
    Automatically creates an 'Isolated microglia' folder inside input_folder.
    """
    import os

    output_folder = os.path.join(input_folder, "Isolated microglia")
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith("_C_0.tiff"):
            input_image_path = os.path.join(input_folder, file_name)
            Crop_Microglia(input_image_path, output_folder,
                           padding=padding, min_size_microns=min_size_microns)

def Measure_Microglia_Morphology(colorized_image_path):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import measure, io
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, disk, convex_hull_image
    from skimage.measure import regionprops, label, perimeter
    import pandas as pd  # Import pandas for Excel export
    import math  # Import math for π
    from scipy.ndimage import center_of_mass

    def calculate_inertia_tensor(binary_image):
        """
        Calculate the inertia tensor of a binary image based on its center of mass.
        """
        com = center_of_mass(binary_image)
        y_indices, x_indices = np.indices(binary_image.shape)
        x_distances = x_indices - com[1]
        y_distances = y_indices - com[0]
        Ixx = np.sum(binary_image * y_distances**2)
        Iyy = np.sum(binary_image * x_distances**2)
        Ixy = -np.sum(binary_image * x_distances * y_distances)
        return np.array([[Ixx, Ixy], [Ixy, Iyy]])

    def calculate_aspect_ratio(eigenvalues):
        """
        Calculate the aspect ratio as the square root of the ratio of the largest to the smallest eigenvalue.
        """
        return np.sqrt(np.max(eigenvalues) / np.min(eigenvalues))

    # Set scale for 20x Objective
    scale = 0.335  # microns per pixel

    # Create a "Contours" folder inside the input folder
    contours_folder = os.path.join(colorized_image_path, "Contours")
    os.makedirs(contours_folder, exist_ok=True)

    # Get a list of all TIFF files in the folder that end with _C_0
    image_files = [file for file in os.listdir(colorized_image_path) if file.endswith('_C_0.tif') or file.endswith('_C_0.tiff')]

    # Iterate over each image file to calculate measurements
    results = []
    for image_file in image_files:
        # Construct the full path to the image file
        image_path = os.path.join(colorized_image_path, image_file)

        # Open the image file and convert to grayscale
        image = io.imread(image_path, as_gray=True)

        # Apply Otsu's thresholding to segment the image
        thresh = threshold_otsu(image)
        binary_image = image > thresh

        # Perform binary closing to fill small holes
        binary_image = binary_closing(binary_image, disk(3))

        # Label connected components
        labeled_image = label(binary_image)

        # Measure properties of labeled regions
        properties = regionprops(labeled_image)

        # Find the largest connected component (largest white area)
        largest_region = max(properties, key=lambda x: x.area)

        # Calculate the convex hull of the largest region
        convex_hull = convex_hull_image(labeled_image == largest_region.label)

        # Calculate the convex hull perimeter
        convex_hull_perimeter_pixels = perimeter(convex_hull)
        convex_hull_perimeter_microns = convex_hull_perimeter_pixels * scale

        # Convert convex hull area to microns²
        convex_hull_area_pixels = np.sum(convex_hull)
        convex_hull_area_microns = convex_hull_area_pixels * (scale ** 2)

        # Calculate perimeter, area, and convex hull area for the largest region
        area_pixels = largest_region.area
        perimeter_pixels = largest_region.perimeter

        # Convert area and perimeter to microns
        area_microns = area_pixels * (scale ** 2)
        perimeter_microns = perimeter_pixels * scale

        # Calculate circularity
        if perimeter_microns > 0:  # Avoid division by zero
            circularity = (4 * math.pi * area_microns) / (perimeter_microns ** 2)
        else:
            circularity = 0  # Set circularity to 0 if perimeter is 0

        # Calculate the inertia tensor and aspect ratio
        binary_region = labeled_image == largest_region.label
        inertia_tensor = calculate_inertia_tensor(binary_region)
        eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)
        aspect_ratio = calculate_aspect_ratio(eigenvalues)

        # Calculate roughness
        if convex_hull_perimeter_microns > 0:  # Avoid division by zero
            roughness = perimeter_microns / convex_hull_perimeter_microns
        else:
            roughness = 0  # Set roughness to 0 if convex hull perimeter is 0

        # Extract the base name (first three elements separated by underscores)
        base_name = "_".join(image_file.split('_')[:3])

        # Append results
        results.append({
            "File": image_file,
            "Base Name": base_name,
            "Area (µm²)": area_microns,
            "Perimeter (µm)": perimeter_microns,
            "Convex Hull Area (µm²)": convex_hull_area_microns,
            "Convex Hull Perimeter (µm)": convex_hull_perimeter_microns,
            "Circularity": circularity,
            "Aspect Ratio": aspect_ratio,
            "Roughness": roughness
        })

        # Create a figure to display the image with contours
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image, cmap='gray')

        # Find contours for the largest region
        contours = measure.find_contours(labeled_image == largest_region.label, 0.5)

        # Plot the contours in red
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')

        # Plot the convex hull in green
        convex_hull_coords = np.argwhere(convex_hull)
        for coord in convex_hull_coords:
            ax.plot(coord[1], coord[0], 'o', markersize=1, color='green')

        # Remove axis labels and ticks
        ax.axis('off')

        # Save the image with contours and convex hull to the "Contours" folder
        output_path = os.path.join(contours_folder, f"contoured_{image_file}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Group by base name and calculate averages for numeric columns only
    numeric_columns = results_df.select_dtypes(include=[np.number]).columns
    averages_df = results_df.groupby("Base Name")[numeric_columns].mean().reset_index()

    # Merge the averages back into the original DataFrame
    results_with_averages = results_df.merge(averages_df, on="Base Name", suffixes=("", " (Avg)"))

    # Ensure averages are only populated once per base name
    results_with_averages.loc[results_with_averages.duplicated(subset=["Base Name"]), [col + " (Avg)" for col in numeric_columns]] = ""

    # Export results to an Excel file
    excel_output_path = os.path.join(colorized_image_path, "Perimeter_Area_ConvexHull_Circularity_AspectRoughness_Results.xlsx")
    results_with_averages.to_excel(excel_output_path, index=False)
    print(f"Results exported to {excel_output_path}")

    return results_with_averages

# Sholl Analysis
def Isolate_Microglia_DAPI(input_image_path, output_folder, padding=20, min_size_microns=100):
    """
    Takes an input image ending in _C_0.tiff, applies a mean filter, and crops the image
    around each individual thresholded object. Only crops objects that meet a minimum size
    determined by the user. If more than 5 objects exist, randomly selects 5 to process.
    Skips objects touching the edges of the image. Saves each cropped object as a separate
    image in the specified output folder with a _#_ added before the _C_0 in the filename.
    Also applies the same crop locations to the corresponding _C_3 image.
    """
    import cv2
    import os
    import numpy as np
    import random
    from skimage.measure import regionprops, label as skimage_label

    # Scale for converting microns² to pixels
    scale = 0.641  # microns per pixel
    min_size_pixels = int(min_size_microns / (scale ** 2))

    # Load the input _C_0 image
    image_c0 = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if image_c0 is None:
        raise ValueError(f"Error: Unable to load image at {input_image_path}")

    # Construct the path for the corresponding _C_3 image
    input_image_c3_path = input_image_path.replace("_C_0", "_C_3")
    image_c3 = cv2.imread(input_image_c3_path, cv2.IMREAD_GRAYSCALE)
    if image_c3 is None:
        raise ValueError(f"Error: Unable to load corresponding _C_3 image at {input_image_c3_path}")

    # Apply a mean filter (blur the _C_0 image)
    blurred_image = cv2.blur(image_c0, (5, 5))

    # Apply thresholding to the blurred image
    _, binary_image = cv2.threshold(
        blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Label connected components in the binary image
    labeled_image = skimage_label(binary_image)

    # Get properties of labeled regions
    props = regionprops(labeled_image)

    # Filter out objects that do not meet the minimum size
    filtered_props = [region for region in props if region.area >= min_size_pixels]

    # Randomly select up to 5 objects if more than 5 exist
    if len(filtered_props) > 5:
        filtered_props = random.sample(filtered_props, 5)
        print(f"Randomly selected 5 objects out of {len(props)} total objects.")

    # Get the base filename of the input image
    base_filename_c0 = os.path.basename(input_image_path)
    base_filename_c3 = os.path.basename(input_image_c3_path)

    # Iterate through the selected regions and crop the images
    for i, region in enumerate(filtered_props):
        # Get the bounding box of the region (min_row, min_col, max_row, max_col)
        min_row, min_col, max_row, max_col = region.bbox

        # Skip objects touching the edges of the image
        if (
            min_row == 0
            or min_col == 0
            or max_row == image_c0.shape[0]
            or max_col == image_c0.shape[1]
        ):
            print(f"Skipping object {i + 1} - touching the edge of the image.")
            continue

        # Add padding to the bounding box
        min_row = max(min_row - padding, 0)
        min_col = max(min_col - padding, 0)
        max_row = min(max_row + padding, image_c0.shape[0])
        max_col = min(max_col + padding, image_c0.shape[1])

        # Crop the _C_0 image around the padded bounding box
        cropped_image_c0 = image_c0[min_row:max_row, min_col:max_col]

        # Crop the _C_3 image around the same bounding box
        cropped_image_c3 = image_c3[min_row:max_row, min_col:max_col]

        # Construct the new filenames with _#_ added before _C_0 and _C_3
        new_filename_c0 = base_filename_c0.replace("_C_0", f"_{i + 1}_C_0")
        new_filename_c3 = base_filename_c3.replace("_C_3", f"_{i + 1}_C_3")

        # Save the cropped _C_0 image in the output folder
        output_path_c0 = os.path.join(output_folder, new_filename_c0)
        cv2.imwrite(output_path_c0, cropped_image_c0)
        print(f"Saved cropped _C_0 image: {output_path_c0}")

        # Save the cropped _C_3 image in the output folder
        output_path_c3 = os.path.join(output_folder, new_filename_c3)
        cv2.imwrite(output_path_c3, cropped_image_c3)
        print(f"Saved cropped _C_3 image: {output_path_c3}")


def Isolate_Microglia_Sholl(input_folder, padding=20, min_size_microns=100):
    """
    Processes all images in a folder that end with _C_0.tiff. For each image, applies a mean filter,
    crops the image around each individual thresholded object, and saves the cropped objects.

    The function automatically creates an 'Isolated microglia' folder inside input_folder.
    """
    import os

    output_folder = os.path.join(input_folder, "Isolated microglia Area")
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith("_C_0.tiff"):
            input_image_path = os.path.join(input_folder, file_name)
            Isolate_Microglia_DAPI(
                input_image_path,
                output_folder,
                padding=padding,
                min_size_microns=min_size_microns,
            )


def calculate_center_of_mass(c0_image_path, c3_image_path, output_base):
    """
    Use the thresholded image of the _C_0 image to apply the boundaries of the white thresholded object
    to the thresholded _C_3 image and calculate the center of mass of the enclosed _C_3 object.
    A blue crosshair is drawn on the overlay image to signify the center of mass.

    Parameters
    ----------
    c0_image_path : str
        Path to the _C_0 image (thresholded).
    c3_image_path : str
        Path to the _C_3 image.
    output_base : str
        Base path for saving the overlay image with the blue crosshair.

    Returns
    -------
    tuple or None
        (x, y) coordinates of the center of mass of the masked thresholded _C_3 object, or None if no object is found.
    """
    import cv2
    import os
    import numpy as np
    import pandas as pd
    from skimage.morphology import skeletonize
    from skimage.measure import regionprops, label as skimage_label
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from scipy.ndimage import label
    from scipy.interpolate import interp1d
    # Load the _C_0 image and threshold it
    c0_image = cv2.imread(c0_image_path, cv2.IMREAD_GRAYSCALE)
    if c0_image is None:
        raise ValueError(f"Error: Unable to load _C_0 image at {c0_image_path}")
    _, c0_binary = cv2.threshold(c0_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Load the _C_3 image and threshold it
    c3_image = cv2.imread(c3_image_path, cv2.IMREAD_GRAYSCALE)
    if c3_image is None:
        raise ValueError(f"Error: Unable to load _C_3 image at {c3_image_path}")
    _, c3_binary = cv2.threshold(c3_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply the mask from the _C_0 image to the thresholded _C_3 image
    masked_c3_image = cv2.bitwise_and(c3_binary, c3_binary, mask=c0_binary)

    # Label connected components in the masked thresholded _C_3 image
    labeled_image = skimage_label(masked_c3_image)

    # Check if any objects are found
    props = regionprops(labeled_image)
    if not props:
        print(f"Warning: No objects found in masked thresholded _C_3 image {c3_image_path}. Skipping this image.")
        return None

    # Calculate the center of mass of the largest connected component
    largest_region = max(props, key=lambda x: x.area)
    center_of_mass = largest_region.centroid  # (y, x) format
    center_x, center_y = int(center_of_mass[1]), int(center_of_mass[0])  # Convert to (x, y)

    # Create an overlay image with the blue crosshair
    overlay = cv2.cvtColor(c3_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for color overlay
    crosshair_color = (255, 0, 0)  # Blue color (BGR format)
    crosshair_thickness = 2
    crosshair_length = 10

    # Draw the horizontal and vertical lines of the crosshair
    cv2.line(overlay, (center_x - crosshair_length, center_y), (center_x + crosshair_length, center_y), crosshair_color, crosshair_thickness)
    cv2.line(overlay, (center_x, center_y - crosshair_length), (center_x, center_y + crosshair_length), crosshair_color, crosshair_thickness)

    # Save the overlay image with the blue crosshair
    overlay_output_path = f"{output_base}_C3_center_of_mass_overlay.png"
    cv2.imwrite(overlay_output_path, overlay)


    return center_x, center_y

def Initialize_Rings(image_path, output_base, soma_coordinates):
    """
    Perform Sholl analysis on a single image, save the skeletonized image with concentric rings,
    and save the intersection graph.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    output_base : str
        Base path for saving the output images and graphs.
    soma_coordinates : tuple
        (x, y) coordinates of the soma.

    Returns
    -------
    distances : list
        Distances from the cell soma (in microns).
    intersection_counts : list
        Number of intersections for each distance.
    """
    import os
    import cv2
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from skimage.morphology import skeletonize
    from scipy.ndimage import label
    from collections import defaultdict
    from scipy.interpolate import interp1d
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return [], []

    # Apply Otsu's threshold to create a binary image
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Skeletonize the binary image
    binary_image = binary_image // 255  # Normalize to 0 and 1
    skeleton = skeletonize(binary_image)

    # Overlay the skeleton on the original image
    skeleton_overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for color overlay
    skeleton_overlay[skeleton == 1] = [0, 0, 255]  # Red color for the skeleton

    # Use the provided soma coordinates
    center_x, center_y = soma_coordinates

    # Define radii parameters
    scale = 0.335  # 1 pixel = 0.335 microns
    start_radius_microns = 10.843  # Starting radius in microns
    step_radius_microns = 2.5  # Step size in microns
    max_radius_microns = 100  # Maximum radius in microns

    # Convert radii to pixels
    start_radius_pixels = int(start_radius_microns / scale)
    step_radius_pixels = int(step_radius_microns / scale)
    max_radius_pixels = int(max_radius_microns / scale)

    # Initialize lists for plotting
    distances = []  # Distances from the cell soma (in microns)
    intersection_counts = []  # Number of intersections for each ring

    # Draw concentric rings and count intersections
    for radius in range(start_radius_pixels, max_radius_pixels + 1, step_radius_pixels):
        # Draw the ring
        cv2.circle(skeleton_overlay, (center_x, center_y), radius, (0, 255, 0), 1)  # Green rings

        # Create a circular band mask for the current radius
        circle_mask = np.zeros_like(skeleton, dtype=bool)
        rr, cc = np.ogrid[:skeleton.shape[0], :skeleton.shape[1]]
        distance_from_center = np.sqrt((rr - center_y)**2 + (cc - center_x)**2)
        circle_band = (distance_from_center >= radius - 1) & (distance_from_center <= radius + 1)
        circle_mask[circle_band] = True

        # Label connected components in the skeleton
        labeled_skeleton, num_features = label(skeleton)

        # Count intersections as unique connected components that overlap with the circle
        count = 0
        for component in range(1, num_features + 1):
            component_mask = labeled_skeleton == component
            if np.any(np.logical_and(component_mask, circle_mask)):
                count += 1

        # Append the distance and intersection count
        distances.append(radius * scale)  # Convert radius to microns
        intersection_counts.append(count)

        # Add pink dots for each intersection point
        intersection_coords = np.argwhere(np.logical_and(skeleton, circle_mask))
        for coord in intersection_coords:
            y, x = coord
            cv2.circle(skeleton_overlay, (x, y), 2, (255, 105, 180), -1)  # Pink dots (BGR: 255, 105, 180)

    # Save the skeletonized image with concentric rings and pink dots
    skeleton_output_path = f"{output_base}_skeleton_overlay.png"
    cv2.imwrite(skeleton_output_path, skeleton_overlay)
 

    # Plot the number of intersections as a function of distance
    plt.figure(figsize=(8, 6))
    plt.plot(distances, intersection_counts, marker='o', color='b', label='Intersections')
    plt.title("Number of Intersections vs. Distance from Cell Soma")
    plt.xlabel("Distance from Cell Soma (microns)")
    plt.ylabel("Number of Intersections")
    plt.grid(True)
    plt.legend()

    # Save the plot
    graph_output_path = f"{output_base}_intersection_graph.png"
    plt.savefig(graph_output_path, dpi=300, bbox_inches="tight")
    plt.close()


    return distances, intersection_counts


def Process_Sholl(input_folder, output_folder_name="Sholl Analysis"):
    """
    Process only images ending with _C_0.tiff and _C_3.tiff in the input folder, perform Sholl analysis,
    and save the results (skeletonized images with concentric rings and intersection graphs) to the output folder.
    Use the center of mass from corresponding _C_3.tiff images to designate the soma.
    Average the results back to the first 2 elements of the file name.

    Parameters
    ----------
    input_folder : str
        Path to the folder containing input images.
    output_folder_name : str, optional
        Name of the folder to save the output images and graphs. Default is "Sholl Analysis".

    Returns
    -------
    None
    """
    from collections import defaultdict
    from scipy.interpolate import interp1d
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import cv2

    # Create the output folder inside the input folder
    output_folder = os.path.join(input_folder, output_folder_name)
    os.makedirs(output_folder, exist_ok=True)

    # Initialize a dictionary to store distances and intersection counts grouped by the first 2 elements of the file name
    grouped_results = defaultdict(list)

    # Initialize a dictionary to store all data for Excel output
    excel_data = {}

    # Initialize a list to store AUC results
    auc_results = []

    # Find matching pairs of _C_3.tiff and _C_0.tiff images
    c3_images = {file_name.split("_C_", 1)[0]: file_name for file_name in os.listdir(input_folder) if file_name.endswith("_C_3.tiff")}
    c0_images = {file_name.split("_C_", 1)[0]: file_name for file_name in os.listdir(input_folder) if file_name.endswith("_C_0.tiff")}

    # Process each matching pair
    for key in c3_images.keys() & c0_images.keys():
        c3_image_path = os.path.join(input_folder, c3_images[key])
        c0_image_path = os.path.join(input_folder, c0_images[key])
        output_base = os.path.join(output_folder, os.path.splitext(c0_images[key])[0])

        # Calculate the center of mass using the mask from the _C_0 image
        soma_coordinates = calculate_center_of_mass(c0_image_path, c3_image_path, output_base)

        # Skip if no objects are found
        if soma_coordinates is None:
            continue

        # Perform Sholl analysis on the corresponding _C_0.tiff image using the soma coordinates
        distances, intersection_counts = Initialize_Rings(c0_image_path, output_base, soma_coordinates)

        # Store the results in the grouped dictionary
        if distances and intersection_counts:
            # Extract the first 2 elements of the file name as the group key
            prefix = "_".join(c0_images[key].split("_")[:2])
            grouped_results[prefix].append((distances, intersection_counts))

            # Save individual image data to the Excel dictionary
            excel_data[c0_images[key]] = pd.DataFrame({
                "Distance (microns)": distances,
                "Intersections": intersection_counts
            })

            # Calculate the AUC for the current image
            auc = np.trapz(intersection_counts, distances)
            auc_results.append({"Image Name": c0_images[key], "AUC": auc})

    # Prepare a DataFrame for the averaged data
    averaged_data = {}

    # Define the resampled distances (5-micron intervals)
    resampled_distances = np.arange(10.843, 105, 2.5)  # Adjust the range as needed

    # Plot the average intersections for each group
    plt.figure(figsize=(10, 8))
    for prefix, results in grouped_results.items():
        # Align distances and compute the average intersections for the group
        all_distances = [r[0] for r in results]
        all_intersections = [r[1] for r in results]

        # Interpolate each image's data to the resampled distances
        resampled_intersections = []
        for distances, intersections in zip(all_distances, all_intersections):
            interpolation_function = interp1d(distances, intersections, kind='linear', bounds_error=False, fill_value=0)
            resampled_intersections.append(interpolation_function(resampled_distances))

        # Compute the average intersections at the resampled distances
        avg_intersections = np.mean(resampled_intersections, axis=0)

        # Save the averaged data for this group
        averaged_data[prefix] = pd.DataFrame({
            "Distance (microns)": resampled_distances,
            "Average Intersections": avg_intersections
        })

        # Plot the average intersections for this group
        plt.plot(resampled_distances, avg_intersections, marker='o', label=f"{prefix} (n={len(results)})")

    # Finalize the compiled figure
    plt.title("Average Number of Intersections vs. Distance from Cell Soma")
    plt.xlabel("Distance from Cell Soma (microns)")
    plt.ylabel("Average Number of Intersections")
    plt.grid(True)
    plt.legend()
    compiled_graph_output_path = os.path.join(output_folder, "compiled_average_intersection_graph.png")
    plt.savefig(compiled_graph_output_path, dpi=300, bbox_inches="tight")
    plt.close()


    # Save all data to an Excel file
    excel_output_path = os.path.join(output_folder, "Sholl_Analysis_Data.xlsx")
    with pd.ExcelWriter(excel_output_path) as writer:
        # Save individual image data
        for file_name, df in excel_data.items():
            df.to_excel(writer, sheet_name=file_name[:31], index=False)  # Excel sheet names are limited to 31 chars

        # Save averaged data
        for prefix, df in averaged_data.items():
            df.to_excel(writer, sheet_name=f"Avg_{prefix[:25]}", index=False)  # Truncate long prefixes for sheet names

        # Save AUC results
        auc_df = pd.DataFrame(auc_results)
        auc_df.to_excel(writer, sheet_name="AUC", index=False)



def Colorize_Microglia_DAPI(folder_path, display_results=False, save_results=True):
    """
    Colorize the yellow (C_0) and blue (C_3) channels of TIFF images and create composite images.

    """
    import os
    import glob
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.io import imread as load_image, imsave
    from skimage.color import gray2rgb

    # Create a new output path that includes the "Microglia Isolated" folder
    output_path = os.path.join(folder_path, "Colorized_Microglia_DAPI")
    os.makedirs(output_path, exist_ok=True)

    # Define the colors for the two channels
    channel_colors = {"C_0": (1, 1, 0),  # Yellow
                      "C_3": (0, 0, 1)}  # Blue

    # Get a list of base file names for all TIFF images ending in _C_0
    base_files = [os.path.splitext(os.path.basename(f))[0].rsplit('_C_0', 1)[0] for f in glob.glob(folder_path + "/*_C_0.tiff")]

    # Iterate through each base file name
    for base_name in base_files:
        channel_images = {}
        for channel, color in channel_colors.items():
            channel_path = os.path.join(folder_path, f"{base_name}_{channel}.tiff")
            if os.path.exists(channel_path):
                channel_image = load_image(channel_path).astype(np.float32)  # Use float32 for compatibility
                channel_images[channel] = channel_image
            else:
                print(f"Channel {channel} not found for {base_name}")
                break

        if len(channel_images) != 2:  # Ensure both channels are present
            print(f"Skipping {base_name} due to missing channels")
            continue

        # Normalize and colorize each channel
        merged = np.zeros_like(gray2rgb(channel_images["C_0"]), dtype=np.float32)
        for channel, color in channel_colors.items():
            channel_image = channel_images[channel]
            vmax = np.percentile(channel_image, 99.9)
            vmin = np.percentile(channel_image, 0.1)
            normalized_image = (channel_image - vmin) / (vmax - vmin)
            normalized_image = np.clip(normalized_image, 0, 1)
            colored_image = gray2rgb(normalized_image) * np.array(color, dtype=np.float32)
            merged += colored_image

        # Normalize the merged image to ensure values are within the valid range
        merged = merged / np.max(merged)

        # Save the composite image as a TIFF file
        composite_output_path = os.path.join(output_path, f'{base_name}.tiff')
        if save_results:
            imsave(composite_output_path, (merged * 255).astype(np.uint8))  # Convert to uint8 for saving


        # Optionally display the image
        if display_results:
            plt.figure(figsize=(10, 10))
            plt.axis(False)
            plt.imshow(merged)
            plt.title(f"Composite Image: {base_name}")
            plt.show()



def crop_thresholded_objects(input_image_path, output_folder, padding=20, min_size_microns=100):
    """
    Multi‑channel crop: C_0 mask, and crops C_0, C_1, C_2 for each object.
    """
    import cv2
    import os
    import numpy as np
    import random
    from skimage.measure import regionprops, label as skimage_label

    scale = 0.641  # microns per pixel
    min_size_pixels = int(min_size_microns / (scale ** 2))

    image_c0 = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if image_c0 is None:
        raise ValueError(f"Error: Unable to load image at {input_image_path}")

    input_image_c1_path = input_image_path.replace("_C_0", "_C_1")
    input_image_c2_path = input_image_path.replace("_C_0", "_C_2")
    image_c1 = cv2.imread(input_image_c1_path, cv2.IMREAD_GRAYSCALE)
    image_c2 = cv2.imread(input_image_c2_path, cv2.IMREAD_GRAYSCALE)

    if image_c1 is None:
        raise ValueError(f"Error: Unable to load corresponding _C_1 image at {input_image_c1_path}")
    if image_c2 is None:
        raise ValueError(f"Error: Unable to load corresponding _C_2 image at {input_image_c2_path}")

    blurred_image = cv2.blur(image_c0, (5, 5))
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    labeled_image = skimage_label(binary_image)
    props = regionprops(labeled_image)

    filtered_props = [region for region in props if region.area >= min_size_pixels]

    if len(filtered_props) > 5:
        filtered_props = random.sample(filtered_props, 5)
        print(f"Randomly selected 5 objects out of {len(props)} total objects.")

    base_filename_c0 = os.path.basename(input_image_path)
    base_filename_c1 = os.path.basename(input_image_c1_path)
    base_filename_c2 = os.path.basename(input_image_c2_path)

    for i, region in enumerate(filtered_props):
        min_row, min_col, max_row, max_col = region.bbox

        if min_row == 0 or min_col == 0 or max_row == image_c0.shape[0] or max_col == image_c0.shape[1]:
            print(f"Skipping object {i + 1} - touching the edge of the image.")
            continue

        min_row = max(min_row - padding, 0)
        min_col = max(min_col - padding, 0)
        max_row = min(max_row + padding, image_c0.shape[0])
        max_col = min(max_col + padding, image_c0.shape[1])

        cropped_image_c0 = image_c0[min_row:max_row, min_col:max_col]
        cropped_image_c1 = image_c1[min_row:max_row, min_col:max_col]
        cropped_image_c2 = image_c2[min_row:max_row, min_col:max_col]

        new_filename_c0 = base_filename_c0.replace("_C_0", f"_{i + 1}_C_0")
        new_filename_c1 = base_filename_c1.replace("_C_1", f"_{i + 1}_C_1")
        new_filename_c2 = base_filename_c2.replace("_C_2", f"_{i + 1}_C_2")

        output_path_c0 = os.path.join(output_folder, new_filename_c0)
        output_path_c1 = os.path.join(output_folder, new_filename_c1)
        output_path_c2 = os.path.join(output_folder, new_filename_c2)

        cv2.imwrite(output_path_c0, cropped_image_c0)
        cv2.imwrite(output_path_c1, cropped_image_c1)
        cv2.imwrite(output_path_c2, cropped_image_c2)

        print(f"Saved cropped _C_0 image: {output_path_c0}")
        print(f"Saved cropped _C_1 image: {output_path_c1}")
        print(f"Saved cropped _C_2 image: {output_path_c2}")


def Isolate_Microglia_GFAP(input_folder, padding=20, min_size_microns=100):
    """
    GFAP / bead version: processes all images in input_folder that end with _C_0.tiff,
    crops C_0/C_1/C_2 around each object, and saves them.

    Automatically creates an 'Isolated microglia' folder inside input_folder.
    """
    import os

    output_folder = os.path.join(input_folder, "Isolated microglia")
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith("_C_0.tiff"):
            input_image_path = os.path.join(input_folder, file_name)
            crop_thresholded_objects(
                input_image_path,
                output_folder,
                padding=padding,
                min_size_microns=min_size_microns,
            )


def measure_mean_gray_value_c1(input_folder):
    """
    Iterate through images ending with _C_1.tiff in input_folder, measure the mean gray
    value of the largest object, and save results to an Excel file plus outlined PNGs.

    Automatically creates:
      - 'Mean_Gray_Bead_Values_C1.xlsx' in input_folder
      - 'Outlined_C1' subfolder for outline images.
    """
    import os
    import cv2
    import pandas as pd
    from skimage.measure import regionprops, label as skimage_label
    from skimage.segmentation import find_boundaries

    output_excel_path = os.path.join(input_folder, "Mean_Gray_Bead_Values_C1.xlsx")
    outlined_folder = os.path.join(input_folder, "Outlined_C1")
    os.makedirs(outlined_folder, exist_ok=True)

    results = []

    for file_name in os.listdir(input_folder):
        if file_name.endswith("_C_1.tiff"):
            image_path = os.path.join(input_folder, file_name)

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Error: Unable to load image at {image_path}")
                results.append({"Image Name": file_name, "Mean Gray Value": None})
                continue

            _, binary_image = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            labeled_image = skimage_label(binary_image)
            props = regionprops(labeled_image, intensity_image=image)

            if props:
                largest_region = max(props, key=lambda x: x.area)
                mean_gray_value = largest_region.mean_intensity

                # Treat low signal as NA
                if mean_gray_value < 15:
                    mean_gray_value = None

                outline_mask = labeled_image == largest_region.label
                boundaries = find_boundaries(outline_mask, mode="outer")

                outlined_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                outlined_image[boundaries] = [0, 0, 255]

                outlined_image_path = os.path.join(
                    outlined_folder,
                    file_name.replace(".tiff", "_outlined.png"),
                )
                cv2.imwrite(outlined_image_path, outlined_image)
                print(f"Saved outlined image: {outlined_image_path}")
            else:
                mean_gray_value = None

            results.append({"Image Name": file_name, "Mean Gray Value": mean_gray_value})

    df = pd.DataFrame(results)

    df["Group"] = df["Image Name"].apply(lambda x: "_".join(x.split("_")[:2]))

    group_averages = df.groupby("Group")["Mean Gray Value"].mean().reset_index()
    group_averages.rename(columns={"Mean Gray Value": "Group Average"}, inplace=True)

    df["Positive"] = df["Mean Gray Value"].apply(
        lambda x: 1 if x is not None and x > 20 else 0
    )
    group_counts = df.groupby("Group")["Positive"].sum().reset_index()
    group_counts.rename(columns={"Positive": "Positive Count"}, inplace=True)

    total_counts = df.groupby("Group")["Image Name"].count().reset_index()
    total_counts.rename(columns={"Image Name": "Total Count"}, inplace=True)

    group_stats = pd.merge(group_counts, total_counts, on="Group")
    group_stats["Positive Percentage"] = (
        group_stats["Positive Count"] / group_stats["Total Count"]
    ) * 100

    df = pd.merge(df, group_averages, on="Group", how="left")
    df = pd.merge(df, group_stats, on="Group", how="left")

    df["Group Average"] = df.groupby("Group")["Group Average"].transform(
        lambda x: [x.iloc[0]] + [None] * (len(x) - 1)
    )
    df["Positive Count"] = df.groupby("Group")["Positive Count"].transform(
        lambda x: [x.iloc[0]] + [None] * (len(x) - 1)
    )
    df["Positive Percentage"] = df.groupby("Group")["Positive Percentage"].transform(
        lambda x: [x.iloc[0]] + [None] * (len(x) - 1)
    )

    df.to_excel(output_excel_path, index=False)
    print(f"Results saved to Excel file: {output_excel_path}")