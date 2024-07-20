from Image import load, load_glob
from Collage import Collage

target_path = input("Target path: ")
input_path = input("Input path: ")
lock_aspect_ratio = input("Lock input aspect ratio (y/n): ") == "y"
only_improvements = (
    input("Discard iteration if it doesn't improve the output (y/n): ") == "y"
)

Collage(
    target=load(target_path),
    inputs=load_glob(input_path),
    lock_aspect_ratio=lock_aspect_ratio,
    only_improvements=only_improvements,
).run()
