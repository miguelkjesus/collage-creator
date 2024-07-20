from __future__ import annotations

from PIL import Image
import numpy as np
import random

from Image import transform, average_color, set_color


def lerp(a, b, t):
    return np.interp(t, [0, 1], [a, b])


class Mutation:
    def __init__(
        self,
        image: Image.Image,
        normalised_position: tuple[float, float],
        angle: float,
        size: tuple[int, int],
        pos_factor: float = 1.1,
        angle_factor: float = 1.1,
        size_factor: float = 1.1,
        factor_factor: float = 0.8,
    ) -> None:
        self.image = image
        self.normalised_position = tuple(np.clip(normalised_position, 0, 1))
        self.angle = angle % 360
        self.size = tuple(np.maximum(1, size))

        self.pos_factor = pos_factor
        self.angle_factor = angle_factor
        self.size_factor = size_factor
        self.factor_factor = factor_factor

        self.rendered = None

    def __repr__(self) -> str:
        return f"Mutation(pos=({float(self.normalised_position[0]):.3f}, {float(self.normalised_position[1]):.3f}), angle={float(self.angle):.1f}, size=({int(self.size[0])}, {int(self.size[1])}))"

    def _mutate_value(self, value, factor, value_type=float):
        return value_type(random.uniform(value / factor, value * factor))

    def _mutate_values(self, values, factor, value_type=float, array_type=tuple):
        return array_type(
            self._mutate_value(value, factor, value_type) for value in values
        )

    def _get_position(
        self, background: Image.Image, image: Image.Image
    ) -> tuple[int, int]:
        return (
            int(lerp(-image.size[0], background.size[0], self.normalised_position[0])),
            int(lerp(-image.size[1], background.size[1], self.normalised_position[1])),
        )

    def _get_color(
        self, target: Image.Image, image: Image.Image, position: tuple[int, int]
    ) -> tuple[int, int, int] | None:
        pos = np.array(position)
        img_size = np.array(image.size)
        target_size = np.array(target.size)

        roi_box = (
            *np.clip(pos, 0, target_size),
            *np.clip(img_size + pos, 0, target_size),
        )
        roi = target.crop(roi_box)

        mask_box = (
            *np.clip(-pos, 0, img_size),
            *np.clip(target_size - pos, 0, img_size),
        )
        mask = image.getchannel("A").crop(mask_box)

        try:
            return average_color(roi, mask)
        except ZeroDivisionError:
            return None

    def mutate(self) -> Mutation:
        return Mutation(
            self.image,
            self._mutate_values(self.normalised_position, self.pos_factor, float),
            self._mutate_value(self.angle, self.angle_factor, float),
            self._mutate_values(self.size, self.size_factor, int),
            self.pos_factor * self.factor_factor,
            self.angle_factor * self.factor_factor,
            self.size_factor * self.factor_factor,
        )

    def render(
        self, background: Image.Image, target: Image.Image
    ) -> Image.Image | None:
        if self.rendered is not None:
            return self.rendered

        transformed = transform(self.image, self.angle, self.size)

        position = self._get_position(background, transformed)
        color = self._get_color(target, transformed, position)
        if color is None:
            return None

        colored = set_color(transformed, color)
        self.rendered = background.copy()
        self.rendered.paste(colored, position, colored)
        return self.rendered
