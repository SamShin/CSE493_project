from PIL import Image, ImageDraw, ImageFont
import os
import sys # For sys.exit

def draw_bounding_box_on_image(image_input_path, normalized_bbox, label=None, color="red", thickness=3, output_path=None):
    """
    Draws a single bounding box on an image and saves it.

    Args:
        image_input_path (str): Path to the input image.
        normalized_bbox (list or tuple): A list/tuple [xmin, ymin, width, height]
                                         with normalized coordinates (0.0 to 1.0).
        label (str, optional): Text label to draw near the box. Defaults to None.
        color (str, optional): Color of the bounding box and label. Defaults to "red".
        thickness (int, optional): Thickness of the bounding box lines. Defaults to 3.
        output_path (str): Path where the image with the box will be saved.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        image = Image.open(image_input_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_input_path}")
        return False
    except Exception as e:
        print(f"Error opening image {image_input_path}: {e}")
        return False

    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size

    if not (normalized_bbox and len(normalized_bbox) == 4 and all(isinstance(n, (float, int)) for n in normalized_bbox)):
        print(f"Error: Invalid bounding box format. Expected [xmin, ymin, width, height]. Got: {normalized_bbox}")
        return False

    xmin_norm, ymin_norm, w_norm, h_norm = normalized_bbox
    abs_xmin = xmin_norm * img_width
    abs_ymin = ymin_norm * img_height
    abs_width = w_norm * img_width
    abs_height = h_norm * img_height

    x0 = int(abs_xmin)
    y0 = int(abs_ymin)
    x1 = int(abs_xmin + abs_width)
    y1 = int(abs_ymin + abs_height)

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(img_width, x1)
    y1 = min(img_height, y1)

    draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=thickness)

    if label:
        try:
            font_size = max(15, int(img_height * 0.025))
            try:
                font = ImageFont.truetype("arial.ttf", size=font_size)
            except IOError:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", size=font_size)
                except IOError:
                    font = ImageFont.load_default()
        except Exception as e:
            print(f"Warning: Could not load custom font ({e}). Using default.")
            font = ImageFont.load_default()

        if hasattr(draw, 'textbbox'):
            text_box_dims = draw.textbbox((0,0), label, font=font)
            text_width = text_box_dims[2] - text_box_dims[0]
            text_height = text_box_dims[3] - text_box_dims[1]
        else:
            text_width, text_height = draw.textsize(label, font=font)

        text_margin = thickness
        text_x = x0 + text_margin
        text_y = y0 - text_height - text_margin * 2

        if text_y < 0:
            text_y = y0 + text_margin * 2
        if text_x + text_width > img_width:
            text_x = x1 - text_width - text_margin

        draw.rectangle(
            [(text_x - 2, text_y - 2), (text_x + text_width + 2, text_y + text_height + 2)],
            fill=color
        )
        draw.text((text_x, text_y), label, fill="white", font=font)

    if not output_path:
        print("Error: Output path must be provided.")
        return False

    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir: # Check if output_dir is not empty (e.g. for saving in current dir)
             os.makedirs(output_dir, exist_ok=True)
        image.save(output_path)
        print(f"Image with bounding box saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")
        return False

# No argparse needed if we are not using command-line arguments
# def main_with_args(): ... (original main function using argparse)

def run_drawing_directly():
    """
    Sets parameters directly in the script and calls the drawing function.
    """
    # --- Define your parameters here ---
    # Get the directory of the currently running script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(script_dir) # Assuming script is in a 'scripts' subdir

    input_image_path = os.path.join(project_root_dir, "data", "video_002", "frames", "frame_0000.jpg")
    # Bounding box as a list of floats: [xmin, ymin, width, height]
    bbox_coords = [
            0.22888875007629395,
            0.34391942620277405,
            0.32378149032592773,
            0.6560805439949036
          ]
    output_image_path = os.path.join(script_dir, "output_from_script.jpg") # Save in the same dir as the script

    custom_label = "Detected Object"
    custom_color = "blue"
    custom_thickness = 4
    # --- End of parameter definition ---

    print(f"Input image: {input_image_path}")
    print(f"Bounding box: {bbox_coords}")
    print(f"Output image: {output_image_path}")
    print(f"Label: {custom_label}, Color: {custom_color}, Thickness: {custom_thickness}")

    success = draw_bounding_box_on_image(
        image_input_path=input_image_path,
        normalized_bbox=bbox_coords,
        label=custom_label,
        color=custom_color,
        thickness=custom_thickness,
        output_path=output_image_path
    )

    if success:
        print("Drawing successful!")
    else:
        print("Drawing failed.")
        sys.exit(1)


if __name__ == "__main__":
    # Before running, ensure Pillow is installed:
    # pip install Pillow
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("CRITICAL ERROR: Pillow (PIL) library is not installed.")
        print("This script requires Pillow for drawing bounding boxes on images.")
        print("Please install it by running: pip install Pillow")
        sys.exit(1)

    # To run with parameters defined in the script:
    run_drawing_directly()

    # If you still want the option to run with command-line arguments,
    # you would keep the original main() function that uses argparse
    # and then decide which one to call, e.g., based on a flag or just comment one out.
    # For now, I've removed the argparse-based main() to keep it simple.