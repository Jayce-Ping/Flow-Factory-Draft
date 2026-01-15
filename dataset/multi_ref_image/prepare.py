# This file preprocess the downloaded `sharegpt4o_image_mini` first
# and convert to a trivia dataset with varying number of reference images.
# ```
# bash download.sh
# 
# python prepare.py
# ```
import json
import os
import random

trivia_prompt = "Combine these images together."
for split in ['train', 'test']:
    file = f"{split}.jsonl"
    new_file = f"{split}_ori.jsonl"
    os.rename(file, new_file)
    with open(new_file, 'r') as f:
        data = [json.loads(line) for line in f]

    all_images = [item['image'] for item in data]    
    seen_combinations = set() # Track unique sets of images
    new_data = []    
    i = 0
    data_num = len(data)
    while len(new_data) < data_num:
        random.seed(42 + i)
        ref_image_num = random.randint(2, 3)
        ref_images = random.sample(all_images, ref_image_num)
        
        # Create a unique signature (sorted so order doesn't matter)
        signature = tuple(sorted(ref_images))
        
        if signature not in seen_combinations:
            new_item = {
                'prompt': trivia_prompt,
                'images': ref_images
            }
            new_data.append(new_item)
            seen_combinations.add(signature)
        
        i += 1 # Increment seed/counter to try again if duplicate found
    
    # Write to `train.jsonl`/`test.jsonl`
    with open(file, 'w') as f:
        for item in new_data:
            f.write(json.dumps(item) + '\n')


    # Remove old files
    if os.path.exists(new_file):
        os.remove(new_file)