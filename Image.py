from typing import Generator
from PIL import Image, ImageFile
from glob import iglob
import numpy as np
import os.path


def load(path: str) -> ImageFile.ImageFile | None:
    if os.path.isfile(path):
        return Image.open(path)
    else:
        return None


def load_glob(glob_path: str) -> Generator[ImageFile.ImageFile, None, None]:
    for path in iglob(glob_path, recursive=True):
        file = load(path)
        if file is not None:
            yield file


def rotate(image: Image.Image, angle: float) -> Image.Image:
    return image.rotate(angle, expand=True)


def scale(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.resize(size, Image.Resampling.NEAREST)


def change_opacity(image: Image.Image, opacity: float) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    return Image.blend(image, overlay, opacity)


def average_color(image: Image.Image, mask: Image.Image):
    return np.average(np.array(image), (0, 1), np.array(mask))


def set_color(image: Image.Image, average_color: tuple[int, int, int]) -> Image.Image:
    gray_arr = np.array(image.convert("L"))

    avg_normalised = np.array(average_color, dtype=np.float32) / 255.0
    new_img_arr = (gray_arr[..., np.newaxis] * avg_normalised).astype(np.uint8)

    new_img = Image.fromarray(new_img_arr, "RGB")
    new_img.putalpha(image.getchannel("A"))
    return new_img


def transform(
    image: Image.Image,
    angle: float,
    size: tuple[int, int],
) -> Image.Image:
    return rotate(scale(image, size), angle)
