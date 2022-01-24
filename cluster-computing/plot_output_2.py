import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def main():
    path = Path(sys.argv[1])

    with open(path) as fp:
        lines = fp.readlines()

    lines = lines[2:]  # Remove header
    data = [           # Parse data 
        (int(lines[i].split(" ")[-1]), float(lines[i+1].split(" ")[-2]))
        for i in range(0, len(lines) - 1, 3)
    ]
    
    # plt.rcParams["figure.figsize"] = (10, 5)
    plt.title("Execution time with OpenMP on a 16-core CPU with different precisions")
    plt.plot(
        [d[0] for d in data],
        [d[1] for d in data],
        "-o"
    )
    plt.xlabel("Number of Summands")
    plt.ylabel("Execution time [s]")
    # plt.xticks([d[0] for d in data])

    out_path = path.with_suffix('.png')
    while out_path.exists():
        replace = ""
        while replace != "y" and replace != "n":
            replace = input(f"{out_path} already exists. Want to replace? (y/n)").strip().lower()
        if replace == "n":
            out_path = Path(input("Enter new filename: "))
        else:
            break

    plt.savefig(out_path)



if __name__ == "__main__":
    main()