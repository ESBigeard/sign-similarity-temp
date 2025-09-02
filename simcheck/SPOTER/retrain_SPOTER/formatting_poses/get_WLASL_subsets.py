if __name__ == "__main__":
    import argparse
    import os
    import json
    from tqdm import tqdm
    from collections import defaultdict

    complete_file = "../../../../data/WLASL/WLASL_v0.3.json"
    subset_file = "../../../../data/WLASL/WLASL300/nslt_300.json"

    data_complete = json.load(open(complete_file, 'r'))
    data_subset = json.load(open(subset_file, 'r'))
    print(data_subset.keys())

    labels = {"Label": [], "Video": [], "Gloss": []}

    for idx, class_data in enumerate(data_complete):
        gloss = class_data["gloss"]
        videos_info = class_data["instances"]
        for video_info in videos_info:
            labels["Gloss"].append(gloss)
            labels["Label"].append(idx)
            labels["Video"].append(video_info["video_id"]) 
            
    videos = labels["Video"]
    classes = labels["Label"]
    glosses = labels["Gloss"]
    labels = dict(zip(videos, zip(classes, glosses)))

    dict_final = defaultdict(list)
    instances_dict = defaultdict(list)

    for video in data_subset.keys():
        if video not in labels.keys():
            print(f"{video} not in the global file.")
            continue
        Label = labels[video][0]
        Gloss = labels[video][1]
        dict_final[Label].append({
            "video_id": video,
            "gloss": Gloss,
            "split": data_subset[video]["subset"]
        })

    dict_dict_final =  defaultdict(list)
    final_list = []

    for label, instances in dict_final.items():
        classs = {
            "gloss": instances[0]["gloss"],  # Assuming all instances have the same gloss
            "instances": []  # Initialize as an empty list
        }
        
        for idx, instance in enumerate(instances):
            video_id = instance["video_id"]
            split = instance["split"]
            classs["instances"].append({
                "video_id": video_id,
                "split": split
            })
        
        final_list.append(classs)

    # for label, instances in dict_dict_final.items():
    #     print(f"Label: {label}, Instances: {instances}")
    
    # save  the final list to a JSON file
    json.dump(final_list, open("WLASL300.json", 'w'), indent=4)
            


        