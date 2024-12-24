import numpy as np

def calculate_road_width(label_path):
    widths = []
    with open(label_path, 'r') as file:
        for line in file:
            coords = list(map(float, line.strip().split()))
            if len(coords) == 4:
                x1_left, y1_left, x1_right, y1_right = coords
                width = np.sqrt((x1_right - x1_left) ** 2 + (y1_right - y1_left) ** 2)
                widths.append(width)
    return widths

# Example usage
label_path = 'D:/Code/Robust-Lane-Detection/data/testset/truth/1_13.jpg'
road_widths = calculate_road_width(label_path)
average_width = np.mean(road_widths)
print(f'Average road width: {average_width}')