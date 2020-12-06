import os, shutil
import csv
class Prep(object):
    def __init__(self):
        self.labels = os.listdir('data/')

    def generate_flat_img_folder(self):
        os.mkdir("data/flat")
        for l in self.labels:
            for f in os.listdir("data/" + l):
                shutil.copyfile("data/" + l + "/" + f, "data/flat")
        print("Flat folder generated")

    def generate_csv_from_flat(self):
        with open("flat_data.csv", "x", newline='') as csvfile:
            flat = os.listdir("data/flat")
            writer = csv.DictWriter(csvfile, fieldnames=["image", "target"])
            for f in flat:
                if ".png" in f:
                    print(os.path.exists('./data/Parasitized/' + f))
                    writer.writerow({
                        'image': 'data/flat/' + f,
                        'target': 1 if os.path.exists('./data/Parasitized/' + f) else 0
                    })
            print("flat_csv generated")

    def generate_csv(self):
        with open("data.csv", 'rw', newline='') as csvfile:
            fieldnames = os.listdir('data/')
            writer = csv.DictWriter(csvfile, fieldnames=["image", "target"])
            writer.writeheader()
            target = 0
            for l in fieldnames:
                for r, d, f in os.walk("data/" + l):
                    for file in f:
                        if file.endswith(".png"):
                            writer.writerow({
                                'image': file,
                                'target': target
                            })
                    target += 1
            print("CSV Generated")