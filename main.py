from argparse import ArgumentParser
from pipeline import Pipeline

def main():
    """" Initializes a pipeline class object and runs pipeline.main """

    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to config file")
    args = parser.parse_args()
    config_path = args.config_path                      #get command line argument

    pipeline = Pipeline(config_path=config_path)        #initialize pipeline object
    pipeline.main()                                     #run pipeline.main


if __name__ == '__main__':
    main()




