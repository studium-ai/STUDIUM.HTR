import os, argparse

def join_file(parts_folder, output_file):
    # Find all parts in the folder
    parts = sorted([p for p in os.listdir(parts_folder) if ".part" in p])

    with open(output_file, 'wb') as output:
        for part in parts:
            part_path = os.path.join(parts_folder, part)
            with open(part_path, 'rb') as f:
                output.write(f.read())
            print(f"Merged: {part_path}")

    print(f"✅ File reconstructed: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_chunks", type=str) # The path to XML files.
    parser.add_argument("--path_merge", type=str) # The path to the images.
    args = parser.parse_args()
    join_file(
        args.path_chunks,
        args.path_merge
    )
