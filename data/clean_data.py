import json
from data.loader import Loader

def main():
    old_file = open("data/start_kit/WLASL_v0.3.json", "r")
    old_data = json.load(old_file)
    old_file.close()

    new_data = {}
    valid = 0
    total = 0

    for gloss in old_data:
        new_data[gloss["gloss"]] = []
        print(f"Cleaning gloss: {gloss["gloss"]}")

        video_data = None
        for video in gloss["instances"]:
            print(total)
            video_data = Loader.get_video_data(video)
            total += 1
            if video_data is not None:
                valid += 1
                new_data[gloss["gloss"]].append(video)

    print("Writing cleaned data to: cleaned_data.json")
    with open("data/cleaned_data.json", "w") as fp:
        json.dump(new_data)

    print(f"Results: {valid} / {total} videos kept ({(100 * valid/total):.2f} %)")
    
if __name__ == "__main__":
    main()