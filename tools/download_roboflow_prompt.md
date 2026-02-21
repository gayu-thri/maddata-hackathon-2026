Act as a Senior ML Engineer. Write a Python script for "RoadScan AI" that downloads and prepares a computer vision dataset from Roboflow Universe.

### 1. Requirements & Classes
The dataset must include the following classes optimized for US road conditions:
- Potholes
- Cracked Pavement (Alligator/Longitudinal)
- Debris Obstruction in Roads
- Faded Lane Markings
- Damaged/Obscured/Broken Road Signs
- Normal Road (Negative Class)

### 2. Data Integrity & Balance
- The script must pull from Roboflow
- Implement a balancing logic: If one class has 500 images and another has 2000, the script should undersample or oversample to ensure a 1:1 ratio across all classes.
- Ensure a strict 80/10/10 split (Train/Val/Test).
- I need atleast 1000 clean samples after filtering under each image class or whatever you think is ideal.
- Ensure the data is "clean": Add a function to filter out images that are too small or have corrupted headers
- To remove any false positives, maybe use a AI-Powered Cleaning (CLIP Filter), after downloading, use openai/clip-vit-base-patch32 to verify images. Create a CLIP_CLASS_PROMPTS dictionary for the classes above. Filter out images that don't meet a 0.60 confidence threshold for their respective class to ensure the dataset is "hackathon-clean" and low-noise.
- Format the final output into a standard Directory Tree: `dataset/{split}/{class_name}/{image}.jpg` ({split} are train, test, val)
- Make sure it's usable for fine-tuning `EfficientNet_V2_S` 

### 3. Hackathon Specifics
- The script should output a `metadata.csv` that maps filenames, labels to a dummy GPS coordinate (latitude, longitude) so we can test our Google Maps integration logic later.
- Use `roboflow` and `shutil` libraries. 

Provide the full Python code with comments explaining how it handles the class imbalance.