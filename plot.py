# Matplotlib for plotting graphs
import matplotlib.pyplot as plt
# Numpy for loading data to plot
import numpy as np
# Converts a function to a commandline interface
import clize


# Plots sorted values from the flattened input and output file
def plot(inp: str, out: str, plot_file: str = "plot.png"):
    # Load and flatten the input and output data
    inp = np.load(inp).flatten()
    out = np.load(out).flatten()
    # Sort the data according to the input and plot as line plot
    plt.plot(inp[np.argsort(inp)], out[np.argsort(inp)])
    # Save the plotted graph to file
    plt.savefig(plot_file)
    # Show the plot as a window
    plt.show()


# Script entrypoint
if __name__ == "__main__":
    # Run the plotting function as commandline
    clize.run(plot)
