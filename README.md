# 3D Object Detection and Vehicle Classification

This project implements a 3D object detection and vehicle classification system. It reads LiDAR point cloud data and object labels, processes them to generate a Bird's Eye View (BEV) image, and draws 3D bounding boxes around detected objects, classifying them into various vehicle types.

## Features

- Load and parse LiDAR point cloud data.
- Read and parse object labels in the format used by the KITTI dataset.
- Generate BEV images from LiDAR data.
- Draw 3D bounding boxes on BEV images, accounting for object orientation.
- Classify detected objects into vehicle types such as Car, Truck, Motorcycle, etc.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/3DObjectDetection-VehicleClassification.git
    cd 3DObjectDetection-VehicleClassification
    ```
2. Install the required dependencies:
    ```bash
    pip install numpy opencv-python
    ```

## Usage

1. **Prepare your dataset:**
   - Organize your dataset directory structure similar to the KITTI dataset:
     ```
     dataset/
     ├── kitti/
         ├── training/
             ├── velodyne/
             ├── label_2/
     ```
   - Place your LiDAR `.bin` files in the `velodyne` folder and label `.txt` files in the `label_2` folder.

2. **Run the main script:**
    ```bash
    python main.py
    ```

3. **Adjust the parameters:**
   - Modify the `dataset_dir` and `sample_id` in the `main` function to point to your dataset and the sample you want to process.
   - Adjust the `boundary` dictionary to specify the area of interest for the BEV image.

## Code Overview

- `Object3D`: Class representing a 3D object with methods to parse label data, classify object types, and calculate object properties.
- `load_labels(label_filename)`: Function to load object labels from a file.
- `draw_3d_box(bev_image, obj, boundary, discretization)`: Function to draw a 3D bounding box on a BEV image.
- `load_velo_scan(velo_filename)`: Function to load and parse LiDAR data from a binary file.
- `create_bev_image(lidar_data, boundary, discretization)`: Function to generate a BEV image from LiDAR data.
- `add_labels_to_bev(bev_image, labels, boundary, discretization)`: Function to add labeled bounding boxes to a BEV image.
- `get_corners(x, y, w, l, yaw)`: Function to compute the corners of a bounding box given its center and dimensions.
- `drawRotatedBox(img, x, y, width, length, yaw, color, thickness)`: Function to draw a rotated bounding box on a BEV image.

## Example

Below is a simplified example of how to use the provided functions to generate a BEV image and draw bounding boxes:

```python
import cv2
import numpy as np
import os

def main():
    dataset_dir = "path/to/your/dataset"
    sample_id = "000110"
    lidar_file = os.path.join(dataset_dir, "velodyne", f"{sample_id}.bin")
    label_file = os.path.join(dataset_dir, "label_2", f"{sample_id}.txt")

    lidar_data = load_velo_scan(lidar_file)
    boundary = {'minX': -20, 'maxX': 40, 'minY': -20, 'maxY': 20, 'minZ': -2, 'maxZ': 2}
    bev_image = create_bev_image(lidar_data, boundary)
    
    objects = load_labels(label_file)
    labels = [[obj.xctr, obj.yctr, obj.zctr, obj.l, obj.w, obj.h, obj.ry] for obj in objects]
    bev_image_with_labels = add_labels_to_bev(bev_image, labels, boundary)

    cv2.imwrite(f"bev_image_with_labels_{sample_id}.png", bev_image_with_labels)
    cv2.imshow("BEV Image with Labels", bev_image_with_labels)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to adjust any details or add any additional sections you think are necessary for your project.
