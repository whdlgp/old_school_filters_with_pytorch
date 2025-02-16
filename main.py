from pathlib import Path
from util import load_image, save_comparison_grid, measure_time
from filter_imple import Sobel, Prewitt, Laplacian, GaussianBlur, LaplacianOfGaussian

def test_filter(filter, input_dir="test_images"):
    #Tests a given filter on all images in input_dir

    input_dir = Path(input_dir)
    input_files = list(input_dir.glob("*.jpg"))
    input_size = (300, 300)

    results = []  # Store all filter results

    print(f"\n[INFO] Running {filter.__class__.__name__} filter on {len(input_files)} images...")

    for input_file in input_files:
        img = load_image(str(input_file), input_size)

        # Apply filter and measure time
        manual_time, filtered_manual = measure_time(filter.filter, img, method="manual")
        conv2d_time, filtered_conv2d = measure_time(filter.filter, img, method="conv2d")

        # Store result with labels
        results.append((filtered_manual, filtered_conv2d, f"{filter.__class__.__name__} (Manual: {manual_time:.4f}s)", f"{filter.__class__.__name__} (Conv2D: {conv2d_time:.4f}s)"))

        print(f"[INFO] Processed {input_file.name}: Manual={manual_time:.4f}s, Conv2D={conv2d_time:.4f}s")

    return results  # Return all results for this filter


if __name__ == "__main__":
    input_dir = "test_images"
    
    filters = [Sobel(), Prewitt(), Laplacian(), GaussianBlur(), LaplacianOfGaussian()]

    for filter_obj in filters:
        results = test_filter(filter_obj, input_dir)
        save_comparison_grid(results, f"filter_{filter_obj.__class__.__name__}")
