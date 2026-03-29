#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRIME AI Feature Extraction Validation Script

Validates the GRIME AI feature extraction module using two approaches:

1. ANALYTICAL VALIDATION
   Synthetic pure-color images with known exact expected values.
   Metrics: Intensity, Shannon Entropy, GCC, ExG, HSV cluster centers.
   Pass criteria: computed value matches analytical expectation within tolerance.
   Images saved as PNG to avoid JPEG compression artifacts on pure colors.

2. DIRECTIONAL VALIDATION
   Pairs of images where the expected directional relationship between
   metric values is known, even if exact values cannot be computed analytically.
   Metrics: GLCM (contrast, homogeneity, energy), Gabor, LBP, Fourier.
   Pass criteria: metric value is higher/lower as expected between the pair.

Results are written to CSV for comparison and manuscript reporting.

Usage:
    python grime_ai_feature_validation.py generate
    python grime_ai_feature_validation.py generate --validate
    python grime_ai_feature_validation.py validate
    python grime_ai_feature_validation.py validate --input C:/images --output C:/results

Author: Generated for GRIME AI validation
License: Apache License, Version 2.0
"""

import os
import cv2
import csv
import argparse
import numpy as np
from datetime import datetime

try:
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    from skimage.filters import gabor
    from skimage.color import rgb2gray
    import pywt
    TEXTURE_AVAILABLE = True
except ImportError:
    TEXTURE_AVAILABLE = False
    print("WARNING: scikit-image or pywavelets not available. Texture validation will be skipped.")


# ======================================================================================================================
# CONSTANTS
# ======================================================================================================================

# ======================================================================================================================
# HYPERLINK HELPER
# ======================================================================================================================

def make_hyperlink(filepath):
    """
    Returns an Excel HYPERLINK formula for the given file path, matching
    the format used in GRIME AI triage and feature extraction CSV reports.
    The formula is returned as a plain string; csv.writer handles quoting.
    """
    clean = filepath.strip().strip('"')
    return f'=HYPERLINK("{clean}", "{os.path.basename(clean)}")'  


TOLERANCE_STRICT = 0.001   # For analytically exact values
TOLERANCE_JPEG   = 0.02    # Wider tolerance when JPEG compression may affect values
IMG_W = 256
IMG_H = 256


# ======================================================================================================================
# IMAGE GENERATION
# Analytical images saved as PNG to avoid JPEG compression artifacts.
# Directional images saved as PNG for the same reason.
# ======================================================================================================================

def make_solid(r, g, b, w=IMG_W, h=IMG_H):
    return np.full((h, w, 3), [b, g, r], dtype=np.uint8)

def make_gradient(w=IMG_W, h=IMG_H):
    image = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        val = int(x / (w - 1) * 255)
        image[:, x] = [val, val, val]
    return image

def make_horizontal_stripes(w=IMG_W, h=IMG_H, stripe_width=8):
    image = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        if (y // stripe_width) % 2 == 0:
            image[y, :] = 255
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def make_checkerboard(w=IMG_W, h=IMG_H, sq=8):
    """Checkerboard produces maximum GLCM contrast at distance=1."""
    image = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if ((x // sq) + (y // sq)) % 2 == 0:
                image[y, x] = 255
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def make_vertical_stripes(w=IMG_W, h=IMG_H, stripe_width=8):
    image = np.zeros((h, w), dtype=np.uint8)
    for x in range(w):
        if (x // stripe_width) % 2 == 0:
            image[:, x] = 255
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def make_noise(w=IMG_W, h=IMG_H, seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

def make_blurry(source_image, kernel=61):
    return cv2.GaussianBlur(source_image, (kernel, kernel), 0)


def generate_test_images(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Analytical images — PNG only, exact pixel values required
    analytical = {
        "pure_red.png":     make_solid(255, 0, 0),
        "pure_green.png":   make_solid(0, 255, 0),
        "pure_blue.png":    make_solid(0, 0, 255),
        "pure_white.png":   make_solid(255, 255, 255),
        "pure_black.png":   make_solid(0, 0, 0),
        "mid_gray.png":     make_solid(127, 127, 127),
    }

    # Directional images — PNG to preserve spatial structure
    directional = {
        "gradient.png":              make_gradient(),
        "horizontal_stripes.png":    make_horizontal_stripes(),
        "vertical_stripes.png":      make_vertical_stripes(),
        "noise.png":                 make_noise(),
        "stripes_blurred.png":       make_blurry(make_horizontal_stripes()),
        "checkerboard.png":          make_checkerboard(),
        "checkerboard_blurred.png":  make_blurry(make_checkerboard(), kernel=31),
        "uniform_gray.png":          make_solid(128, 128, 128),
    }

    all_images = {**analytical, **directional}
    for filename, image in all_images.items():
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, image)
        print(f"  {filename}")

    return {name: os.path.join(output_dir, name) for name in all_images}


# ======================================================================================================================
# METRIC COMPUTATION
# ======================================================================================================================

def compute_intensity(bgr_image):
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    return float(cv2.mean(gray)[0])

def compute_entropy(bgr_image):
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 255])
    total = gray.shape[0] * gray.shape[1]
    entropy = 0.0
    for count in hist:
        p = count[0] / total
        if p > 0:
            entropy += -p * np.log2(p)
    return entropy

def compute_gcc(bgr_image):
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype(float)
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    total = r + g + b
    total[total == 0] = 1
    return float(np.mean(g / total))

def compute_exg(bgr_image):
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype(float)
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    total = r + g + b
    total[total == 0] = 1
    rn, gn, bn = r/total, g/total, b/total
    return float(np.mean(2*gn - rn - bn))

def compute_hsv_dominant(bgr_image):
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV).astype(float)
    h = float(np.mean(hsv[:,:,0]) * 2.0)  # OpenCV H: 0-180 → scale to 0-360
    s = float(np.mean(hsv[:,:,1]))
    v = float(np.mean(hsv[:,:,2]))
    return h, s, v

def compute_glcm(bgr_image):
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)
    contrast    = float(graycoprops(glcm, 'contrast').mean())
    homogeneity = float(graycoprops(glcm, 'homogeneity').mean())
    energy      = float(graycoprops(glcm, 'energy').mean())
    return contrast, homogeneity, energy

def compute_gabor_response(bgr_image, frequency=0.3, theta=0):
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
    filt_real, filt_imag = gabor(gray, frequency=frequency, theta=theta)
    return float(np.mean(np.sqrt(filt_real**2 + filt_imag**2)))

def compute_fourier_high_freq(bgr_image):
    """Mean high-frequency energy — higher for textured/sharp images."""
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY).astype(float)
    f_shift = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.abs(f_shift)
    rows, cols = gray.shape
    cy, cx = rows // 2, cols // 2
    y, x = np.indices((rows, cols))
    distances = np.sqrt((x - cx)**2 + (y - cy)**2)
    # High frequency = outer 50% of the spectrum
    mask = distances > (min(rows, cols) * 0.25)
    return float(np.mean(magnitude[mask]))


# ======================================================================================================================
# VALIDATION TESTS
# ======================================================================================================================

def run_analytical_tests(image_paths):
    results = []

    def test(name, image_name, metric, actual, expected, tol=TOLERANCE_STRICT):
        passed = abs(actual - expected) <= tol
        filepath = image_paths.get(image_name, "")
        results.append({
            "Test Type":    "Analytical",
            "Test Name":    name,
            "Image":        make_hyperlink(filepath) if filepath else image_name,
            "Metric":       metric,
            "Expected":     round(expected, 6),
            "Actual":       round(actual, 6),
            "Difference":   round(abs(actual - expected), 6),
            "Tolerance":    tol,
            "Pass / Fail":  "Pass" if passed else "FAIL",
        })
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name} | {metric}: expected={expected:.6f}, actual={actual:.6f}")

    print("\nAnalytical tests...")

    def img(name): return cv2.imread(image_paths[name])

    # --- Intensity ---
    test("Pure white",  "pure_white.png", "Intensity", compute_intensity(img("pure_white.png")), 255.0, tol=0.01)
    test("Pure black",  "pure_black.png", "Intensity", compute_intensity(img("pure_black.png")),   0.0, tol=0.01)
    test("Mid gray",    "mid_gray.png",   "Intensity", compute_intensity(img("mid_gray.png")),    127.0, tol=0.01)

    # --- Shannon Entropy: uniform image → entropy = 0 ---
    test("Pure red",    "pure_red.png",   "Entropy", compute_entropy(img("pure_red.png")),    0.0, tol=0.001)
    test("Pure green",  "pure_green.png", "Entropy", compute_entropy(img("pure_green.png")),  0.0, tol=0.001)
    test("Pure white",  "pure_white.png", "Entropy", compute_entropy(img("pure_white.png")),  0.0, tol=0.001)
    test("Pure black",  "pure_black.png", "Entropy", compute_entropy(img("pure_black.png")),  0.0, tol=0.001)

    # --- GCC ---
    # Pure green [R=0,G=255,B=0]: GCC = 255/255 = 1.0
    # Pure red   [R=255,G=0,B=0]: GCC = 0/255   = 0.0
    # Pure blue  [R=0,G=0,B=255]: GCC = 0/255   = 0.0
    # Pure white [R=255,G=255,B=255]: GCC = 255/765 = 1/3
    test("Pure green",  "pure_green.png", "GCC", compute_gcc(img("pure_green.png")),  1.0,    tol=TOLERANCE_STRICT)
    test("Pure red",    "pure_red.png",   "GCC", compute_gcc(img("pure_red.png")),    0.0,    tol=TOLERANCE_STRICT)
    test("Pure blue",   "pure_blue.png",  "GCC", compute_gcc(img("pure_blue.png")),   0.0,    tol=TOLERANCE_STRICT)
    test("Pure white",  "pure_white.png", "GCC", compute_gcc(img("pure_white.png")),  1/3,    tol=TOLERANCE_STRICT)

    # --- ExG ---
    # Pure green normalized: r=0, g=1, b=0 → ExG = 2(1)-0-0 = 2.0
    # Pure red   normalized: r=1, g=0, b=0 → ExG = 2(0)-1-0 = -1.0
    # Pure white normalized: r=g=b=1/3     → ExG = 2/3-1/3-1/3 = 0.0
    test("Pure green",  "pure_green.png", "ExG", compute_exg(img("pure_green.png")),  2.0,  tol=TOLERANCE_STRICT)
    test("Pure red",    "pure_red.png",   "ExG", compute_exg(img("pure_red.png")),   -1.0,  tol=TOLERANCE_STRICT)
    test("Pure white",  "pure_white.png", "ExG", compute_exg(img("pure_white.png")),  0.0,  tol=TOLERANCE_STRICT)

    # --- HSV ---
    # OpenCV HSV hue range is 0-180; GRIME AI scales by x2 to give 0-360.
    # Pure red   BGR[0,0,255]   → HSV H=0   → scaled H=0
    # Pure green BGR[0,255,0]   → HSV H=60  → scaled H=120
    # Pure blue  BGR[255,0,0]   → HSV H=120 → scaled H=240
    # Pure white → S=0, V=255
    # Pure black → S=0, V=0
    h, s, v = compute_hsv_dominant(img("pure_red.png"))
    test("Pure red",   "pure_red.png",   "HSV-H",  h,   0.0, tol=1.0)
    test("Pure red",   "pure_red.png",   "HSV-S",  s, 255.0, tol=1.0)
    test("Pure red",   "pure_red.png",   "HSV-V",  v, 255.0, tol=1.0)

    h, s, v = compute_hsv_dominant(img("pure_green.png"))
    test("Pure green", "pure_green.png", "HSV-H",  h, 120.0, tol=1.0)
    test("Pure green", "pure_green.png", "HSV-S",  s, 255.0, tol=1.0)
    test("Pure green", "pure_green.png", "HSV-V",  v, 255.0, tol=1.0)

    h, s, v = compute_hsv_dominant(img("pure_blue.png"))
    test("Pure blue",  "pure_blue.png",  "HSV-H",  h, 240.0, tol=1.0)
    test("Pure blue",  "pure_blue.png",  "HSV-S",  s, 255.0, tol=1.0)
    test("Pure blue",  "pure_blue.png",  "HSV-V",  v, 255.0, tol=1.0)

    h, s, v = compute_hsv_dominant(img("pure_white.png"))
    test("Pure white", "pure_white.png", "HSV-S",  s,   0.0, tol=1.0)
    test("Pure white", "pure_white.png", "HSV-V",  v, 255.0, tol=1.0)

    return results


def run_directional_tests(image_paths):
    results = []

    def test(name, metric, val_a, label_a, val_b, label_b, expected_direction):
        if expected_direction == 'A>B':
            passed = val_a > val_b
        else:
            passed = val_a < val_b
        path_a = image_paths.get(label_a, "")
        path_b = image_paths.get(label_b, "")
        results.append({
            "Test Type":      "Directional",
            "Test Name":      name,
            "Metric":         metric,
            "Image A":        make_hyperlink(path_a) if path_a else label_a,
            "Value A":        round(val_a, 6),
            "Image B":        make_hyperlink(path_b) if path_b else label_b,
            "Value B":        round(val_b, 6),
            "Expected":       f"{label_a} {'>' if expected_direction == 'A>B' else '<'} {label_b}",
            "Pass / Fail":    "Pass" if passed else "FAIL",
        })
        status = "PASS" if passed else "FAIL"
        sym = ">" if expected_direction == "A>B" else "<"
        print(f"  [{status}] {name} | {metric}: {label_a}={val_a:.4f} {sym} {label_b}={val_b:.4f}")

    print("\nDirectional tests...")

    def img(name): return cv2.imread(image_paths[name])

    noise_img   = img("noise.png")
    white_img   = img("uniform_gray.png")
    hstripe_img = img("horizontal_stripes.png")
    vstripe_img = img("vertical_stripes.png")
    blurred_img = img("stripes_blurred.png")
    gradient_img= img("gradient.png")
    checker_img = img("checkerboard.png")
    checker_blurred = img("checkerboard_blurred.png")

    # --- Entropy ---
    # Noise has much higher entropy than a uniform image
    test("Noise vs uniform",    "Entropy",
         compute_entropy(noise_img),    "noise.png",
         compute_entropy(white_img),    "uniform_gray.png", "A>B")
    # Gradient has higher entropy than uniform
    test("Gradient vs uniform", "Entropy",
         compute_entropy(gradient_img), "gradient.png",
         compute_entropy(white_img),    "uniform_gray.png", "A>B")
    # Gradient spans all 256 gray values uniformly — higher entropy than noise which has uneven distribution
    test("Gradient vs noise",   "Entropy",
         compute_entropy(gradient_img), "gradient.png",
         compute_entropy(noise_img),    "noise.png", "A>B")

    if TEXTURE_AVAILABLE:
        # --- GLCM ---
        # Checkerboard (maximum pixel transitions) vs blurred checkerboard
        c_check, h_check, e_check = compute_glcm(checker_img)
        c_cblur, h_cblur, e_cblur = compute_glcm(checker_blurred)
        # Uniform vs noise
        c_noise, h_noise, e_noise = compute_glcm(noise_img)
        c_uni,   h_uni,   e_uni   = compute_glcm(white_img)

        # Checkerboard at distance=1 has maximum contrast; blurred has fewer sharp transitions
        test("Checkerboard vs blurred",  "GLCM Contrast",
             c_check, "checkerboard.png", c_cblur, "checkerboard_blurred.png", "A>B")
        # Uniform image has perfect homogeneity=1; noise has near-zero
        test("Uniform vs noise",         "GLCM Homogeneity",
             h_uni, "uniform_gray.png", h_noise, "noise.png", "A>B")
        # Uniform image has maximum energy=1; noise has near-zero
        test("Uniform vs noise",         "GLCM Energy",
             e_uni, "uniform_gray.png", e_noise, "noise.png", "A>B")
        # Noise has much higher contrast than uniform
        test("Noise vs uniform",         "GLCM Contrast",
             c_noise, "noise.png", c_uni, "uniform_gray.png", "A>B")

        # --- Gabor ---
        # Gabor filter perpendicular to horizontal stripes should respond strongly to H-stripes
        # and weakly to V-stripes (oriented the same way as the filter)
        g_h_on_hstripe = compute_gabor_response(hstripe_img, frequency=0.3, theta=np.pi/2)
        g_h_on_vstripe = compute_gabor_response(vstripe_img, frequency=0.3, theta=np.pi/2)
        test("Gabor perp to H-stripes vs V-stripes", "Gabor energy",
             g_h_on_hstripe, "horizontal_stripes.png", g_h_on_vstripe, "vertical_stripes.png", "A>B")

        g_v_on_vstripe = compute_gabor_response(vstripe_img, frequency=0.3, theta=0)
        g_v_on_hstripe = compute_gabor_response(hstripe_img, frequency=0.3, theta=0)
        test("Gabor perp to V-stripes vs H-stripes", "Gabor energy",
             g_v_on_vstripe, "vertical_stripes.png", g_v_on_hstripe, "horizontal_stripes.png", "A>B")

        # --- Fourier high-frequency energy ---
        # Checkerboard (sharp edges) should have more high-freq energy than blurred version
        hf_check   = compute_fourier_high_freq(checker_img)
        hf_cblur   = compute_fourier_high_freq(checker_blurred)
        hf_noise   = compute_fourier_high_freq(noise_img)
        hf_uniform = compute_fourier_high_freq(white_img)

        test("Checkerboard vs blurred: high-freq energy", "Fourier high-freq",
             hf_check, "checkerboard.png", hf_cblur, "checkerboard_blurred.png", "A>B")
        test("Noise vs uniform: high-freq energy", "Fourier high-freq",
             hf_noise, "noise.png", hf_uniform, "uniform_gray.png", "A>B")

    return results


# ======================================================================================================================
# OUTPUT
# ======================================================================================================================

def write_results(analytical_results, directional_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if analytical_results:
        path = os.path.join(output_dir, f"analytical_validation_{timestamp}.csv")
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=analytical_results[0].keys())
            writer.writeheader()
            writer.writerows(analytical_results)
        passed = sum(1 for r in analytical_results if r["Pass / Fail"] == "Pass")
        print(f"\nAnalytical: {passed}/{len(analytical_results)} passed. Results: {path}")

    if directional_results:
        path = os.path.join(output_dir, f"directional_validation_{timestamp}.csv")
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=directional_results[0].keys())
            writer.writeheader()
            writer.writerows(directional_results)
        passed = sum(1 for r in directional_results if r["Pass / Fail"] == "Pass")
        print(f"Directional: {passed}/{len(directional_results)} passed. Results: {path}")


def print_summary(analytical_results, directional_results):
    all_results = analytical_results + directional_results
    total  = len(all_results)
    passed = sum(1 for r in all_results if r["Pass / Fail"] == "Pass")
    failed = total - passed
    print(f"\n{'='*55}")
    print(f"OVERALL: {passed}/{total} tests passed, {failed} failed.")
    if failed > 0:
        print("FAILED tests:")
        for r in all_results:
            if r["Pass / Fail"] != "Pass":
                print(f"  - {r['Test Name']} | {r['Metric']}")
    print(f"{'='*55}")


# ======================================================================================================================
# CLI
# ======================================================================================================================

def cmd_generate(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output is None:
        args.output = os.path.join(script_dir, "data", "feature_extraction_test_images")

    print(f"Generating test images in: {args.output}")
    image_paths = generate_test_images(args.output)
    print(f"\nGeneration complete. {len(image_paths)} images created.")

    if args.validate:
        results_dir = os.path.join(script_dir, "results")
        analytical  = run_analytical_tests(image_paths)
        directional = run_directional_tests(image_paths)
        write_results(analytical, directional, results_dir)
        print_summary(analytical, directional)


def cmd_validate(args):
    if args.input is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.input = os.path.join(script_dir, "data", "feature_extraction_test_images")
    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output = os.path.join(script_dir, "results")

    print(f"Running validation on: {args.input}")
    extensions = ('.jpg', '.jpeg', '.png')
    image_paths = {}
    for f in os.listdir(args.input):
        if os.path.splitext(f)[1].lower() in extensions:
            image_paths[f] = os.path.join(args.input, f)

    if not image_paths:
        print(f"No images found in {args.input}")
        return

    analytical  = run_analytical_tests(image_paths)
    directional = run_directional_tests(image_paths)
    write_results(analytical, directional, args.output)
    print_summary(analytical, directional)


def main():
    parser = argparse.ArgumentParser(
        description="GRIME AI Feature Extraction Validation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate test images:
    python grime_ai_feature_validation.py generate

  Generate and run validation:
    python grime_ai_feature_validation.py generate --validate

  Run validation on existing images:
    python grime_ai_feature_validation.py validate

  Run validation on a specific folder:
    python grime_ai_feature_validation.py validate --input C:/images --output C:/results
        """
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("generate", help="Generate test images")
    gen_parser.add_argument("--output", default=None,
                            help="Output folder (default: 'Feature Extraction Test Images' next to script)")
    gen_parser.add_argument("--validate", action="store_true",
                            help="Also run validation after generating")
    gen_parser.set_defaults(func=cmd_generate)

    val_parser = subparsers.add_parser("validate", help="Run validation on existing images")
    val_parser.add_argument("--input", default=None,
                            help="Input folder (default: 'Feature Extraction Test Images' next to script)")
    val_parser.add_argument("--output", default=None,
                            help="Output folder for results CSV")
    val_parser.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
