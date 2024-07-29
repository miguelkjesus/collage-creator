from typing import Iterable, Generator, Any
from multiprocessing import Pool, cpu_count
from functools import partial
from PIL import Image
import numpy as np
import imageio
import random
import time

from Mutation import Mutation


class Collage:
    def __init__(
        self,
        *,
        target: Image.Image,
        inputs: Iterable[Image.Image],
        output: Image.Image | None = None,
        output_path: str,
        video_path: str | None = None,
        fps: int | None = None,
        iterations: int | None = None,  # how many images to stop add
        evolutions: int,  # how many times mutations should evolve
        population: int,  # number of mutations to create
        fittest_num: int,  # number of mutations to keep every evolution
        lock_aspect_ratio: bool | None = None,
        only_improvements: bool | None = None,
        num_processes: int | None = None,
        chunk_size: int | None = None,
    ) -> None:
        self.target = target.convert("RGB")
        self.inputs = [input.convert("RGBA") for input in inputs]

        self.output = (
            output.convert("RGB")
            if output is not None
            else Image.new("RGB", self.target.size)
        )
        self.output_path = output_path
        self.output_score = 0

        self.video = None
        self.video_path = video_path
        self.fps = fps if fps is not None else 30

        self.lock_aspect_ratio = (
            lock_aspect_ratio if lock_aspect_ratio is not None else False
        )
        self.only_improvements = (
            only_improvements if only_improvements is not None else False
        )

        self.iterations = iterations if iterations is not None else -1
        self.population = population
        self.fittest_num = fittest_num
        self.evolutions = evolutions

        self.max_scale = 1.1  # relative to the longest edge of the target

        self.num_processes = num_processes if num_processes is not None else cpu_count()
        self.chunk_size = (
            chunk_size
            if chunk_size is not None
            else max(population // (self.num_processes * 16), 1)  # XXX arbritrary
        )

        self.best_mutations: list[tuple[float, Mutation]] = []

    def __enter__(self):
        if self.video_path is not None:
            self.video = imageio.get_writer(self.video_path, fps=self.fps)  # type: ignore

        return self

    def __exit__(self, *_: Any) -> None:
        if self.video is not None:
            self.video.close()

    def register_mutation(self, mut: Mutation, score: float) -> None:
        self.best_mutations.append((score, mut))
        self.best_mutations.sort(key=lambda x: x[0], reverse=True)
        if len(self.best_mutations) > self.fittest_num:
            self.best_mutations = self.best_mutations[: self.fittest_num]

    def run(self) -> None:
        self.best_mutations.clear()

        it = 0
        while it != self.iterations:
            print(f"Iteration {it + 1}:")
            print("\tCreating initial population")
            self.create_initial_population()
            for ev in range(self.evolutions):
                print(f"\tStarting evolution {ev + 1}/{self.evolutions}")
                self.evolve()

            score, best_mut = self.best_mutations[0]
            print(f"\tScore: {score:.4g}")
            print(f"\tScore difference: {score - self.output_score:.4g}")
            if self.only_improvements and score < self.output_score:
                print("\tNo improvement; discarding iteration.")
                continue

            print(f"\tBest: {best_mut}")
            self.output_score = score
            output = best_mut.render(self.output, self.target)
            if output is None:
                continue
            self.output = output

            print(f"\tSaving output to '{self.output_path}'")
            self.output.save(self.output_path)

            if self.video is not None:
                print(f"\tAdding frame to '{self.video_path}'")
                self.append_to_video(self.output)

            print()

            it += 1

    def append_to_video(self, image: Image.Image) -> None:
        if self.video is None:
            return

        self.video.append_data(np.array(image))

    def create_initial_population(self) -> None:
        mutations = (self.random_mutation() for _ in range(self.population))
        self.process_mutations(mutations)

    def evolve(self) -> None:
        mutations = self.get_evolutions()
        self.process_mutations(mutations)

    def process_mutations(self, mutations: Iterable[Mutation]) -> None:
        start = time.perf_counter()

        with Pool(self.num_processes) as pool:
            process_mutation_with_state = partial(
                process_mutation, self.output, self.target
            )
            for mut, score in pool.imap_unordered(
                process_mutation_with_state, mutations, chunksize=self.chunk_size
            ):
                if score is not None:
                    self.register_mutation(mut, score)

        print(f"\t\tProcess time: {time.perf_counter() - start:.4g}s")

    def get_evolutions(self) -> Generator[Mutation, None, None]:
        for _, mut in self.best_mutations:
            for _ in range(self.population // len(self.best_mutations)):
                yield mut.mutate()

    def random_mutation(self) -> Mutation:
        high = lambda: int(np.array(np.max(self.target.size)) * self.max_scale)

        image = random.choice(self.inputs)
        if self.lock_aspect_ratio:
            size: tuple[int, int] = tuple(
                (
                    np.array((1, image.size[1] / image.size[0]))
                    * random.randint(1, high())
                ).astype(int)
            )
        else:
            size: tuple[int, int] = tuple(np.random.randint(1, high(), size=(2)))

        return Mutation(
            image=image,
            angle=random.uniform(0, 360),
            normalised_position=tuple(np.random.uniform(0, 1, size=(2))),
            size=size,
        )


def get_score(target: Image.Image, image: Image.Image) -> float:
    image_arr = np.array(image)
    target_arr = np.array(target)
    mse = np.sum((target_arr - image_arr) ** 2)
    mse /= float(target.size[0] * target.size[1])
    return float(1 - mse / (255**2))


def process_mutation(
    output: Image.Image, target: Image.Image, mut: Mutation
) -> tuple[Mutation, float | None]:
    image = mut.render(output, target)
    if image is None:
        return mut, None
    return mut, get_score(target, image)
