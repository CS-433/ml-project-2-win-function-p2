import os
import random
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import easyocr

def visualize_predictions(image_path, results, save_dir='./predicted'):
    """this function will display an image with the predicted points

    Args:
        image_path : path to the image
        results : result of model.predict(image_path)
        save_dir : folder where the image will be saved. Defaults to './predicted'.

    Returns:
        (str) : path to the output image
    """
    # Load the image using PIL
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Get the first result
    # result = results[0]
    
    # Draw bounding boxes
    boxes = results[0].boxes
    for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Get confidence and class information
        confidence = box.conf[0]
        class_id = box.cls[0]
        class_name = results[0].names[int(class_id)]
        
        # Draw rectangle
        draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=2)
        
        # Add label
        label = f"{class_name} ({confidence:.2f})"
        draw.text((x1, y1 - 10), label, fill="green")
    
    # Draw keypoints if they exist
    if hasattr(results[0], 'keypoints'):
        keypoints = results[0].keypoints
        colors = ["red", "blue", "green", "yellow", "purple"]  # PIL color names
        
        # Convert keypoints to numpy if needed
        if hasattr(keypoints, 'data'):
            kpts = keypoints.data.cpu().numpy()
        else:
            kpts = keypoints.cpu().numpy()
        
        # Draw each keypoint
        for det_idx, det_kpts in enumerate(kpts):
            for idx, kpt in enumerate(det_kpts):
                x, y = int(kpt[0]), int(kpt[1])
                confidence = float(kpt[2]) if kpt.shape[0] > 2 else 1.0
                
                if confidence > 0.5:
                    color = colors[idx % len(colors)]
                    # Draw circle for keypoint
                    draw.ellipse(
                        [(x-4, y-4), (x+4, y+4)],
                        fill=color,
                        outline=color
                    )
                    # Add keypoint index
                    draw.text((x+5, y+5), str(idx), fill=color)
            
            # Draw connections between keypoints if defined
            # if hasattr(results[0], 'keypoint_links'):
            #     for link in results[0].keypoint_links:
            #         pt1 = tuple(map(int, det_kpts[link[0]][:2]))
            #         pt2 = tuple(map(int, det_kpts[link[1]][:2]))
            #         draw.line([pt1, pt2], fill="white", width=1)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate output filename
    image_name = os.path.basename(image_path)
    base_name, ext = os.path.splitext(image_name)
    output_path = os.path.join(save_dir, f"{base_name}_predicted{ext}")
    
    # Save the image
    image.save(output_path)
    
    return output_path


def choose_random_file(directory, to_skip=set([])):
    """chooses a random file from a directory.
    This was used to select a random image for pseudo labeling 

    Args:
        directory (str): path to the directory
        to_skip (set, optional): set of image names to skip. Defaults to set([]).

    Returns:
        (str) : chosen image
    """
    # Get a list of all files in the directory
    files = os.listdir(directory)
    
    # Filter out directories (if you want only files)
    files = [f for f in files if os.path.isfile(os.path.join(directory, f))]
    
    # Check if the directory is empty
    if not files:
        print("The directory is empty.")
        return None
    
    # Choose a random file
    random_file = random.choice(files)
    if(random_file in to_skip):
        choose_random_file(directory)
    
    return random_file

def visualize_single_pair_labels(image_path, label_path):
    """display an image with the points that were assigned to that image.
    will display either the scale bar or thorax points

    Args:
        image_path (str): image path
        label_path (str): path to a .txt file containing
                          the label of the image (example: ./thorax_dataset/labels/train/{image_path})
    """
    # Load the image
    image = Image.open(image_path)
    img_width, img_height = image.size
    draw = ImageDraw.Draw(image)

    # Read YOLO annotations
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Parse each annotation
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center, y_center, width, height, x1, y1, x2, y2 = map(float, parts[1:])

        # Denormalize YOLO bounding box coordinates
        x_center_abs = x_center * img_width
        y_center_abs = y_center * img_height
        width_abs = width * img_width
        height_abs = height * img_height

        # Calculate top-left and bottom-right corners
        x1_abs = x1 * img_width
        y1_abs = y1 * img_height
        x2_abs = x2 * img_width
        y2_abs = y2 * img_height

        # Draw bounding box
        draw.rectangle(
            [
                (x_center_abs - width_abs / 2, y_center_abs - height_abs / 2),
                (x_center_abs + width_abs / 2, y_center_abs + height_abs / 2)
            ],
            outline="green",
            width=2
        )

        # Draw keypoints (x1, y1) and (x2, y2)
        draw.ellipse(
            [(x1_abs - 5, y1_abs - 5), (x1_abs + 5, y1_abs + 5)],
            fill="red",
            outline="red"
        )
        draw.ellipse(
            [(x2_abs - 5, y2_abs - 5), (x2_abs + 5, y2_abs + 5)],
            fill="red",
            outline="red"
        )

        # Add labels near the keypoints
        draw.text((x1_abs + 5, y1_abs - 15), "P1", fill="red")
        draw.text((x2_abs + 5, y2_abs - 15), "P2", fill="red")

        # Add class label
        draw.text(
            (x_center_abs - width_abs / 2, y_center_abs - height_abs / 2 - 15),
            f"Class {class_id}",
            fill="green"
        )

    # Show the image
    # image.show()
    display(image)

def visualize_double_pair_labels(image_path, label_path):
    """display an image with the points that were assigned to that image.
    will only work with the generated folder thorax_and_scale_dataset

    Args:
        image_path (str): image path
        label_path (str): path to a .txt file containing
                          the label of the image (example: ./thorax_and_scale_dataset/labels/train/{image_path})
    """
    # Load the image
    image = Image.open(image_path)
    img_width, img_height = image.size
    draw = ImageDraw.Draw(image)

    # Read YOLO annotations
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Class names mapping
    class_names = {0: "Thorax", 1: "Bar"}

    # Parse each annotation
    for line in lines:
        parts = line.strip().split()
        # print(parts)
        class_id = int(parts[0])
        x_center, y_center, width, height, x1, y1, x2, y2,_,_,_,_ = map(float, parts[1:])

        # Denormalize YOLO bounding box coordinates
        x_center_abs = x_center * img_width
        y_center_abs = y_center * img_height
        width_abs = width * img_width
        height_abs = height * img_height

        # Calculate top-left and bottom-right corners
        x1_abs = x1 * img_width
        y1_abs = y1 * img_height
        x2_abs = x2 * img_width
        y2_abs = y2 * img_height

        # Draw bounding box
        bbox_color = "green" if class_id == 0 else "blue"
        draw.rectangle(
            [
                (x_center_abs - width_abs / 2, y_center_abs - height_abs / 2),
                (x_center_abs + width_abs / 2, y_center_abs + height_abs / 2)
            ],
            outline=bbox_color,
            width=2
        )

        # Draw keypoints (x1, y1) and (x2, y2)
        keypoint_color = "red" if class_id == 0 else "yellow"
        draw.ellipse(
            [(x1_abs - 5, y1_abs - 5), (x1_abs + 5, y1_abs + 5)],
            fill=keypoint_color,
            outline=keypoint_color
        )
        draw.ellipse(
            [(x2_abs - 5, y2_abs - 5), (x2_abs + 5, y2_abs + 5)],
            fill=keypoint_color,
            outline=keypoint_color
        )

        # Add labels near the keypoints
        draw.text((x1_abs + 5, y1_abs - 15), "P1", fill=keypoint_color)
        draw.text((x2_abs + 5, y2_abs - 15), "P2", fill=keypoint_color)

        # Add class label
        draw.text(
            (x_center_abs - width_abs / 2, y_center_abs - height_abs / 2 - 15),
            f"{class_names.get(class_id, 'Unknown')}",
            fill=bbox_color
        )

    # Show the image
    display(image)
    
    
    
def visualize_predictions(image_path, results):
    """this will plot the result of model.predict(image_path)

    Args:
        image_path (str): path to the image
        results (dict): result of model.predict(image_path)
    """
    # Load the image using PIL
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Get the first result
    # result = results[0]
    
    # Draw bounding boxes
    boxes = results[0].boxes
    for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Get confidence and class information
        confidence = box.conf[0]
        class_id = box.cls[0]
        class_name = results[0].names[int(class_id)]
        
        # Draw rectangle
        draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=2)
        
        # Add label
        label = f"{class_name} ({confidence:.2f})"
        draw.text((x1, y1 - 10), label, fill="green")
    
    # Draw keypoints if they exist
    if hasattr(results[0], 'keypoints'):
        keypoints = results[0].keypoints
        colors = ["red", "blue", "green", "yellow", "purple"]  # PIL color names
        
        # Convert keypoints to numpy if needed
        if hasattr(keypoints, 'data'):
            kpts = keypoints.data.cpu().numpy()
        else:
            kpts = keypoints.cpu().numpy()
        
        # Draw each keypoint
        for det_idx, det_kpts in enumerate(kpts):
            for idx, kpt in enumerate(det_kpts):
                x, y = int(kpt[0]), int(kpt[1])
                confidence = float(kpt[2]) if kpt.shape[0] > 2 else 1.0
                
                if confidence > 0.5:
                    color = colors[idx % len(colors)]
                    # Draw circle for keypoint
                    draw.ellipse(
                        [(x-4, y-4), (x+4, y+4)],
                        fill=color,
                        outline=color
                    )
                    # Add keypoint index
                    draw.text((x+5, y+5), str(idx), fill=color)
            
            # Draw connections between keypoints if defined
            # if hasattr(results[0], 'keypoint_links'):
            #     for link in results[0].keypoint_links:
            #         pt1 = tuple(map(int, det_kpts[link[0]][:2]))
            #         pt2 = tuple(map(int, det_kpts[link[1]][:2]))
            #         draw.line([pt1, pt2], fill="white", width=1)
    
    # Display the image
    display(image)

import cv2
import numpy as np

def detect_black_scale_bar(image_path):
    """
    Assuming the image has dimension 640x640
    The function detects the scale bar in an image, assuming it is a black, straight line,
    located at the bottom between pixel rows 550 and 640.

    Parameters:
        image_path (str): Path to the image file.
        
    Returns:
        tuple: Start and end points of the scale bar ((x1, y1), (x2, y2)) or None if not found.
    """
    # Load the image and convert to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Focus on the region between pixel rows 550 and 640 where the scale bar is located
    bottom_region = gray[550:640, :]

    # Threshold to isolate black regions
    _, binary = cv2.threshold(bottom_region, 3, 255, cv2.THRESH_BINARY_INV)

    # Detect edges to emphasize potential scale bars
    edges = cv2.Canny(binary, 3, 150)

    # Use Hough Line Transform to detect straight lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Adjust y-coordinates to match the original image's coordinate system
            y1 += 550
            y2 += 550
            
            # Check if the line is horizontal (y1 ≈ y2)
            if abs(y2 - y1) < 5:
                # Return the first valid horizontal line
                return (x1, y1), (x2, y2)

    return None  # Return None if no scale bar is found

def draw_points_and_scale(image_path, scale_bar_points, thorax_points):
    """
    Draws the scale bar and thorax start/end points on the image.

    Parameters:
        image_path (str): Path to the image file.
        scale_bar_points (tuple): Start and end points of the scale bar ((x1, y1), (x2, y2)).
        thorax_points (tuple): Start and end points of the thorax ((x3, y3), (x4, y4)).
        ALL POINTS MUST BE INTEGERS
    """
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw the scale bar
    if scale_bar_points:
        (sx1, sy1), (sx2, sy2) = scale_bar_points
        cv2.line(image_rgb, (sx1, sy1), (sx2, sy2), color=(0, 255, 0), thickness=2)
        cv2.putText(image_rgb, "Scale Bar", (sx1, sy1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 1, cv2.LINE_AA)

    # Draw the thorax start and end positions
    if thorax_points:
        (tx1, ty1), (tx2, ty2) = thorax_points
        cv2.circle(image_rgb, (tx1, ty1), radius=5, color=(255, 0, 0), thickness=-1)
        cv2.putText(image_rgb, "Thorax Start", (tx1 + 5, ty1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.circle(image_rgb, (tx2, ty2), radius=5, color=(255, 0, 0), thickness=-1)
        cv2.putText(image_rgb, "Thorax End", (tx2 + 5, ty2 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Display the image
    plt.figure(figsize=(10, 6))
    plt.imshow(image_rgb)
    plt.title("Scale Bar and Thorax Positions")
    plt.axis("off")
    plt.show()

def draw_points_and_scale(image_path, scale_bar_points, thorax_points):
    """
    Draws the scale bar and thorax start/end points on the image.

    Parameters:
        image_path (str): Path to the image file.
        scale_bar_points (tuple): Start and end points of the scale bar ((x1, y1), (x2, y2)).
        thorax_points (tuple): Start and end points of the thorax ((x3, y3), (x4, y4)).
        ALL POINTS MUST BE INTEGERS
    """
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw the scale bar
    if scale_bar_points:
        (sx1, sy1), (sx2, sy2) = scale_bar_points
        cv2.line(image_rgb, (sx1, sy1), (sx2, sy2), color=(0, 255, 0), thickness=2)
        cv2.putText(image_rgb, "Scale Bar", (sx1, sy1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 1, cv2.LINE_AA)

    # Draw the thorax start and end positions
    if thorax_points:
        (tx1, ty1), (tx2, ty2) = thorax_points
        cv2.circle(image_rgb, (tx1, ty1), radius=5, color=(255, 0, 0), thickness=-1)
        cv2.putText(image_rgb, "Thorax Start", (tx1 + 5, ty1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.circle(image_rgb, (tx2, ty2), radius=5, color=(255, 0, 0), thickness=-1)
        cv2.putText(image_rgb, "Thorax End", (tx2 + 5, ty2 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Display the image
    plt.figure(figsize=(10, 6))
    plt.imshow(image_rgb)
    plt.title("Scale Bar and Thorax Positions")
    plt.axis("on")
    plt.show()
    
def resize_image(img_path, img_name):
    """resizes the image inside the folder containing all the image (image_path)
    and places it inside resized_images

    Args:
        img_path (str): path to the folder containing all the image (ex: /images/ant.jpg)
        img_name (str): name of the image file (ex: ant.jpg)

    Returns:
        str: path to the output image
    """
    # Open the image
    img = Image.open(img_path)
    # Resize the image
    image_dim = 640
    resized_img = img.resize((image_dim, image_dim), Image.Resampling.LANCZOS)
    # Save the resized image
    resized_img_path = os.path.join('./resized_images', img_name)
    resized_img.save(resized_img_path)
    return resized_img_path

def process_images_from_csv(input_csv, output_csv, images_folder):
    """
    Reads a CSV file with image paths, detects scale bar positions, and saves results to a new CSV file.
    
    Parameters:
        input_csv (str): Path to the input CSV file containing image paths.
        output_csv (str): Path to save the updated CSV with scale bar positions.
        images_folder (str): Path to the images
    """
    # Load the CSV
    df = pd.read_csv(input_csv)
    
    # Ensure the CSV has an 'image_path' column
    if 'ant' not in df.columns:
        raise ValueError("CSV must contain an 'ant' column to find the images.")
    
    # Add new columns for scale bar positions
    df['x1_bar'] = None
    df['y1_bar'] = None
    df['x2_bar'] = None
    df['y2_bar'] = None

    count = 0

    # Process each image
    for idx, row in df.iterrows():
        image_path = images_folder + row['ant']
        image_path = resize_image(image_path, row["ant"])
        result = detect_black_scale_bar(image_path)

        if result:
            start_point, end_point = result
            df.at[idx, 'x1_bar'] = start_point[0]
            df.at[idx, 'y1_bar'] = start_point[1]
            df.at[idx, 'x2_bar'] = end_point[0]
            df.at[idx, 'y2_bar'] = end_point[1]
        else:
            print(f"Scale bar not detected for image: {image_path}")
            count += 1
    
    # Save the updated DataFrame to a new CSV
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv} \n Number of images with undetected scale bars: {count}")


def produce_heatmap(model, img_path, file_name):
    """this will display a heatmap of the predicted image

    Args:
        model (YOLO): yolo model
        img_path (str): path to the image to predict (ex: ./original/ant.jpg)
        file_name (str): name of the image (ex: ant.jpg)
    """
    from ultralytics.solutions import heatmap

    im0 = cv2.imread(img_path)   # path to image file

    # Heatmap Init
    heatmap_obj = heatmap.Heatmap(colormap=cv2.COLORMAP_JET,
                        imw=im0.shape[0],  # should same as im0 width
                        imh=im0.shape[1],  # should same as im0 height
                        view_img=True)

    results = model.track(im0, persist=True)
    im0 = heatmap_obj.generate_heatmap(im0)
    base_dir = os.getcwd()
    os.makedirs(os.path.join(base_dir, 'heatmaps'), exist_ok=True)
    cv2.imwrite(f"{file_name}.jpg", im0)
    print("File saved inside heatmaps folder")

def detect_text(path_folder_image, file_name, path_directory_save, n_split, overlap, scaling_factor):
    """ Find text in the image using EasyOCR pre-tained model.
        The image is split into smaller patches to increase detection speed
    
    Args:
        path_folder_images (str): path of the directory containing the image 
        file_name (str): name of the image file
        path_directory_save (str): path of the directory where to save the images
        n_split (int): Define the number of patches, which is n_split*n_split
        overlap (int): this is the percentage of overlap between neighboring patches. This number belongs to [0.,0.5]
        scaling_factor (int): Images must be upsaled to improve detection performance. 

    Returns:
        text_ (list): list that contains the text bounding box, text detected string, and confidance score (belongs to [0.,1.0])
        concat_text (str): concatenated detected text
    """

    path = path_folder_image + file_name
    im_original = cv2.imread(path)
    s = np.asarray(im_original.shape)
    print(s)
    noTextDetected = True
    size_patch = np.floor(s/n_split).astype(int)

    print(size_patch)
    
    v = np.flip(np.arange(n_split))
    
    noTextDetected = True
    for x in v:
        for y in v:
            #stops when noTextDetected is equal to False
            if noTextDetected:
                
                x1 = max(0, np.floor(size_patch[0]*(x-overlap)).astype(int))
                x2 = min(s[0], np.ceil(size_patch[0]*(x+1+overlap)).astype(int))
                   
                y1 = max(0,np.floor(size_patch[1]*(y-overlap)).astype(int))
                y2 = min(s[1], np.ceil(size_patch[1]*(y+1+overlap)).astype(int))

                im = im_original[x1:x2,y1:y2]
                
                scale_factor = scaling_factor
                im = cv2.resize(im, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
                blur_factor = 5
                im = cv2.blur(im, (blur_factor, blur_factor))
                
                # instance text detector
                reader = easyocr.Reader(['en'], gpu=False)

                # detect text on image
                text_ = reader.readtext(im)

                threshold = 0.25
                # draw bbox and text

                concat_text = ""
                for t_, t in enumerate(text_):

                    bbox, text, score = t
                    concat_text = concat_text + " " + text

                if len(concat_text) >= 4:
                    #sets noTextDetected equal to false if the last 2 characters are 'cm', 'mm', 'pm', or 'um' to stop analyse images further
                    # the mu letter of micro meter is either detected as p or u
                    noTextDetected = not (concat_text[-2:] == 'cm' or concat_text[-2:] == 'mm' or concat_text[-2:] == 'pm' or concat_text[-2:] == 'um')
                    print(concat_text)
                    print(concat_text[0].isnumeric)
                    
                    cv2.rectangle(im, bbox[0], bbox[2], (0, 255, 0), 5)
                    cv2.putText(im, concat_text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)
                    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                    plt.savefig( path_directory_save + "detected_text_" + file_name)
                    plt.show()
                    
    return text_, concat_text
