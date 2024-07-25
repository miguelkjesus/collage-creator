from itertools import chain
from argparse import ArgumentParser
from Image import load, load_glob
from Collage import Collage


def main():
    parser = ArgumentParser()
    parser.add_argument("-t", "--target", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-i", "--inputs", required=True, nargs="+")
    parser.add_argument("-p", "--population", required=True, type=int)
    parser.add_argument("-e", "--evolutions", required=True, type=int)
    parser.add_argument("-f", "--fittest-num", required=True, type=int)
    parser.add_argument("-I", "--iterations", type=int)
    parser.add_argument("-v", "--video")  # TODO
    parser.add_argument("--lock-aspect-ratio", action="store_true")
    parser.add_argument("--only-improvements", action="store_true")
    parser.add_argument("--num-processes", type=int)
    parser.add_argument("--chunk-size", type=int)
    args = parser.parse_args()

    target = load(args.target)
    if target is None:
        return

    Collage(
        target=target,
        output=load(args.output),
        output_path=args.output,
        inputs=chain.from_iterable(load_glob(glob) for glob in args.inputs),
        population=args.population,
        evolutions=args.evolutions,
        fittest_num=args.fittest_num,
        iterations=args.iterations,
        lock_aspect_ratio=args.lock_aspect_ratio,
        only_improvements=args.only_improvements,
    ).run()


if __name__ == "__main__":
    main()
