from LLRunner.SST import sst

slide_source_dir = "/pesgisipth/NDPI"

# find all the NDPI files in the slide_source_dir that start with H
slides = sst.find_files(slide_source_dir, "H*.ndpi")

print(f"Number of slides found: {len(slides)}")
