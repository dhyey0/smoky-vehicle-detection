import json

ORIGINAL_ANNOTATIONS_FILE = "./annotations/annotations.json"
FIXED_ANNOTATIONS_FILE = "./annotations/annotations_fixed.json"

def fix_filenames_in_coco():
    """
    Reads a COCO annotation file, removes the UUID-like prefix from
    each 'file_name', and saves a new, corrected annotation file.
    """
    print(f"Loading annotations from: {ORIGINAL_ANNOTATIONS_FILE}")
    
    try:
        with open(ORIGINAL_ANNOTATIONS_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Original annotation file not found!")
        return

    images_info = data.get('images', [])
    
    if not images_info:
        print("No images found in the annotation file.")
        return
        
    print("Fixing filenames...")
    fixed_count = 0
    for image_info in images_info:
        original_filename = image_info.get('file_name', '')
        
        # Check if the filename contains a hyphen, which is part of the pattern
        if '-' in original_filename:
            # Split the string by the hyphen and take the last part
            # e.g., "bc057e7e-image1.jpg" becomes "image1.jpg"
            corrected_filename = original_filename.split('-')[-1]
            image_info['file_name'] = corrected_filename
            fixed_count += 1

    print(f"Fixed {fixed_count} filenames.")

    # Save the corrected data to a new file
    with open(FIXED_ANNOTATIONS_FILE, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Successfully saved corrected annotations to: {FIXED_ANNOTATIONS_FILE}")


if __name__ == "__main__":
    fix_filenames_in_coco()