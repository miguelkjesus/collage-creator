from PIL import Image, ImageFile
import numpy as np
import random

from Mutation import Mutation


class Collage:
    def __init__(
        self,
        target: ImageFile.ImageFile,
        inputs: list[ImageFile.ImageFile],
        output: ImageFile.ImageFile | None = None,
        lock_aspect_ratio: bool = False,
        only_improvements: bool = False,
    ) -> None:
        self.target = target.convert("RGB")
        self.inputs = [input.convert("RGBA") for input in inputs]
        self.output = (
            output.convert("RGB")
            if output is not None
            else Image.new("RGB", self.target.size)
        )
        self.output_score = 0

        self.lock_aspect_ratio = lock_aspect_ratio
        self.only_improvements = only_improvements

        self.iterations = 100000
        self.population = 200  # number of mutations to create
        self.selection_num = 10  # number of mutations to keep every evolution
        self.evolutions = 1  # how many times mutations should evolve

        self.max_scale = 2  # relative to the longest edge of the target

        self.best_mutations: list[tuple[float, Mutation]] = []

    def register_mutation(self, mut: Mutation, score: float) -> None:
        self.best_mutations.append((score, mut))
        self.best_mutations.sort(key=lambda x: x[0], reverse=True)
        if len(self.best_mutations) > self.selection_num:
            self.best_mutations = self.best_mutations[: self.selection_num]

    def run(self) -> None:
        self.best_mutations = []

        for it in range(self.iterations):
            print(f"Iteration {it + 1}:")
            print("\tCreating initial population")
            self.get_initial_population()
            for ev in range(self.evolutions):
                print(f"\tStarting evolution {ev + 1}/{self.evolutions}")
                self.evolve()

            score, best_mut = self.best_mutations[0]
            print(f"\tScore difference: {score - self.output_score:.4g}")
            if self.only_improvements and score < self.output_score:
                print("\tNo improvement; discarding iteration.")
                continue

            print(f"\tBest: {best_mut}")
            self.output_score = score
            self.output = best_mut.render(self.output, self.target)

            print("\tSaving output\n")
            self.output.save("output.png")

    def get_score(self, mut: Mutation) -> float | None:
        image = mut.render(self.output, self.target)
        if image is None:
            return None

        image_arr = np.array(image)
        target_arr = np.array(self.target)
        mse = np.sum((target_arr - image_arr) ** 2)
        mse /= float(self.target.size[0] * self.target.size[1])
        return 1 - mse / (255**2)

    def get_initial_population(self) -> None:
        i = 0
        while i < self.population:
            mut = self.random_mutation()
            score = self.get_score(mut)
            if score is None:
                continue
            self.register_mutation(mut, score)
            i += 1

    def evolve(self) -> None:
        num_children = self.population // len(self.best_mutations) - 1
        for _, mut in self.best_mutations:
            i = 0
            while i < num_children:
                child = mut.mutate()
                score = self.get_score(child)
                if score is None:
                    continue

                self.register_mutation(child, score)
                i += 1

    def random_mutation(self) -> Mutation:
        image = random.choice(self.inputs)
        if self.lock_aspect_ratio:
            size = (
                np.array((1, image.size[1] / image.size[0]))
                * random.randint(1, np.array(np.max(self.target.size)) * self.max_scale)
            ).astype(int)
        else:
            size = tuple(
                np.random.randint(
                    1, np.array(np.max(self.target.size)) * self.max_scale, size=(2)
                )
            )

        return Mutation(
            image=image,
            angle=random.uniform(0, 360),
            normalised_position=tuple(np.random.uniform(0, 1, size=(2))),
            size=size,
        )
