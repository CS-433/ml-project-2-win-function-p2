import os
from ultralytics import YOLO
from PIL import Image, ImageDraw

def visualize_predictions(image_path, results, save_dir='./predicted'):
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
    
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate output filename
    image_name = os.path.basename(image_path)
    base_name, ext = os.path.splitext(image_name)
    output_path = os.path.join(save_dir, f"{base_name}_predicted{ext}")
    
    # Save the image
    image.save(output_path)
    
    return output_path

if __name__ == '__main__':
    model = YOLO('./best.pt')
    image_path = './original/antweb1008008_p_1.jpg'
    results = model.predict(source=image_path)
    output_path = visualize_predictions(image_path, results)
    print(f"Saved predicted image to: {output_path}")