import argparse

DEBUG_LEVEL = 0

def main():
    Parser = argparse.ArgumentParser()
    """
    Image path, number of pairs / dataset size, Output Path, Debug Level
    """
    Parser.add_argument(
        "--ImagePath",
        default="",  # TODO: Add default when dataset is added to repo.
        help="Base path of images, Default: TODO",
    )
    Parser.add_argument(
        "--OutputPath",
        default="/Outputs/Data/", 
        help="Base path of images, Default: /Outputs/Data",
    )

    Parser.add_argument(
        "--DebugLevel",
        type=int,
        default=0,
        help="Increase debug verbosity with higher debug level"
    )

    Args = Parser.parse_args()
    ImagePath = Args.ImagePath
    OutputPath = Args.OutputPath
    global DEBUG_LEVEL
    DEBUG_LEVEL = Args.DebugLevel
    
    return


if __name__ == '__main__':
    main()