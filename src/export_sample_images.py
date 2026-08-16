import os
from PIL import Image
from torchvision import datasets

CLASS_NAMES = [
    "tshirt", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "ankle_boot"
]

def main():
    print("==== Exporting Real Sample PNG images for the tool")
    output_dir = "data/sample_images"
    os.makedirs(output_dir, exist_ok=True)

    #Load raw dataset without normalization/resizing to save true PNGs
    test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True)

    #Target class indices to export: [03: sneaker, 01: trouser, 00:tshirt, 08:bag, 09: ankle_boot]
    target_classes = {
        7: "03_sneaker.png",
        1: "01_trouser.png",
        0: "00_tshirt.png",
        8: "08_bag.png",
        9: "09_ankle_boot.png"
    }

    exported_count = 0
    saved_files = []

    for img, label in test_dataset:
        if label in target_classes:
            filename = target_classes.pop(label)
            filepath = os.path.join(output_dir, filename)

            #Save PIL image directly PNG file
            img.save(filepath)
            saved_files.append(filepath)
            exported_count += 1

        if not target_classes:
            break

    print(f"Successfully exported {exported_count} PNG sample images to '{output_dir}/':")
    for path in saved_files:
        print(f"  -{path}")
    print("\nPart 3 tool 'classify_product_image' is ready to evaluate these exact PNG files!")

if __name__ == "__main__":
    main()