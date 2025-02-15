from pathlib import Path
from util import load_image, show_image, show_comparison, measure_time
from filter_imple import Sobel

def test_sobel(input_dir = "test_images"):
    # Test image files
    input_dir = Path(input_dir)
    input_files= list(input_dir.glob("*.jpg"))

    # Sobel Filter
    sobel = Sobel()

    # Do test
    for input_file in input_files:
        img = load_image(str(input_file))

        # Sobel
        manual_time, filtered_manual = measure_time(sobel.filter, img, method="manual")
        conv2d_time, filtered_conv2d = measure_time(sobel.filter, img, method="conv2d")

        title1 = f"Manual ({manual_time:.4f}s)"
        title2 = f"Conv2D ({conv2d_time:.4f}s)"
        show_comparison(filtered_manual, filtered_conv2d, title1=title1, title2=title2)

if __name__ == "__main__":
    input_dir = "test_images"
    test_sobel(input_dir)
