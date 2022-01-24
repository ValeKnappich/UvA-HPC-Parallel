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
    
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.title("Execution time with OpenMP on a 16-core CPU")
    plt.plot(
        [d[0] for d in data],
        [d[1] for d in data],
        "-o"
    )
    plt.xlabel("Number of Threads")
    plt.ylabel("Execution time [s]")
    plt.xticks([d[0] for d in data])
    # Add annotations for 16th and 32nd point
    for i in (15, 16, 31):
        y_offset = 5 if i != 16 else 6 
        plt.annotate(f"{data[i][1]:.2f}s", data[i], (data[i][0] - 1, data[i][1] + y_offset), arrowprops=dict(arrowstyle="simple", color="black"))

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