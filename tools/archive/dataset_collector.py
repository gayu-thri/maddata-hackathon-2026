import os

# ============ CONFIG ============

IMAGES_PER_CLASS = 500
SAVE_DIR = "road_dataset_from_ddgs"

CLASS_SEARCH_QUERIES = {
    "potholes": [
        "pothole in asphalt road close up",
        "large pothole on city street",
        "deep pothole highway damage",
        "car hitting pothole road",
        "pothole from dashcam perspective",
        "water filled pothole after rain",
        "urban pothole road surface",
        "rural road pothole damage",
    ],
    "cracked_pavement": [
        "alligator cracking asphalt road",
        "longitudinal crack pavement",
        "transverse crack asphalt",
        "road surface crack close up",
        "cracked road from top view",
        "cracked pavement drone view",
        "severe pavement cracking highway",
    ],
    "road_debris_obstruction": [
        "fallen tree blocking road",
        "rock on highway road hazard",
        "debris on roadway dashcam",
        "construction debris road",
        "object blocking lane road",
        "road obstruction accident scene",
        "storm debris on road",
    ],
    "broken_road_signs": [
        "damaged traffic sign pole",
        "bent stop sign roadside",
        "fallen road sign on street",
        "vandalized traffic sign",
        "broken street sign urban",
        "rusted damaged road sign",
    ],
    "faded_lane_markings": [
        "faded lane lines highway",
        "worn road markings asphalt",
        "barely visible lane markings",
        "old pavement paint faded",
        "night road faded lane markings",
        "rainy road faded lines",
    ],
    "normal_road": [
        "smooth asphalt road daytime",
        "clean highway with clear lane markings",
        "well maintained city road",
        "newly paved road surface",
        "clear road dashcam view",
        "urban road no damage",
    ]
}

# =================================

from icrawler.builtin import BingImageCrawler # Bing is often less "ban-heavy" than Google

def collect_dataset():
    for class_name, queries in CLASS_SEARCH_QUERIES.items():
        print(f"\n--- Scraping Class: {class_name} ---")
        
        # Create a specific folder for the class
        class_dir = os.path.join(SAVE_DIR, class_name)
        
        # Initialize the crawler (Bing is generally more permissive)
        # 'threads=4' makes it much faster
        crawler = BingImageCrawler(
            downloader_threads=4,
            storage={'root_dir': class_dir}
        )

        for query in queries:
            print(f"Searching for: {query}")
            # icrawler handles the 403 errors and retries internally
            crawler.crawl(
                keyword=query, 
                max_num=IMAGES_PER_CLASS // len(queries), # Distribute count across queries
                overwrite=False
            )

if __name__ == "__main__":
    collect_dataset()
