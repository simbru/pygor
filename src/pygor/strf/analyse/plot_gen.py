import sys
import pygor 
import pygor.strf.analyse.plot


def plot_regiment():
        

if __name__ == "__main__":
    try: 
        experiment_filepath = sys.argv[1]
        output_path = sys.argv[2]
    except IndexError:    
        print("Hello, traveler! You have envoked the plot_gen function, without input. You will now be prompted for \n", 
            "input. You should pass the following: [Experiment-class pickle path] [output path]. Good luck!")
        experiment_filepath = input("Experiment pickle path: ")
        output_path = input("Output path: ")
    print("You said to... Look at experiemnt:", experiment_filepath, "and save outputs to:", output_path)
    
    experiment = pygor.classes.experiment.Experiment(experiment_filepath)
    