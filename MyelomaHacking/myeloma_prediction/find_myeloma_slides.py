import os
from LLRunner.read.SST import sst

slide_source_dir = "/pesgisipth/NDPI"

# find all the NDPI files in the slide_source_dir that start with H
slides = [f for f in os.listdir(slide_source_dir) if f.startswith("H") and f.endswith(".NDPI")]

print(f"Number of slides found: {len(slides)}")
