import json

# Label mapping: `cat_to_name.json` mapping encoded categories to flower names
def cat_labels(path='cat_to_name.json'):
    with open(path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
