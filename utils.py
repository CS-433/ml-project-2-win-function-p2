import os
import random
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display


def choose_random_file(directory, to_skip=set([])):
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