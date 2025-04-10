import os
import glob
from argparse import ArgumentParser

from utils import DataLoader, Preprocessor, Converter, Resolver
from utils import TempFileManager

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--replays', type=str, nargs='+', help="Input file or directory containing .rep files")
    parser.add_argument('--include_components', '--components', type=str, nargs='+',
                        default=['worker', 'ground', 'air', 'building', 'vision'],
                        help='List of channel component names to include')
    args = parser.parse_args()
    return args


def resolve_input_path(input_path):
    source_data_root = os.path.join(os.getcwd(), 'data', 'input', 'src')

    if os.path.isdir(os.path.join(source_data_root, input_path)) and input_path.endswith('.rep'):
        rep_dir = source_data_root + input_path
    elif os.path.isdir(os.path.join(source_data_root, input_path + '.rep')):
        rep_dir = os.path.join(source_data_root, input_path + '.rep')
        print(f"[INFO] Interpreting input as directory with '.rep' suffix: {rep_dir}")
    else:
        raise ValueError("Invalid input path: must be a directory ending in '.rep' or a base directory with a corresponding '.rep' subdirectory")

    return rep_dir


def main():
    args = parse_arguments()
    
    print(f"[INFO] Using the following components: {', '.join(args.include_components)}")

    for rep in args.replays:
        print(f"Current replay data: {rep}")
        rep_dir = resolve_input_path(rep)
        
        loader = DataLoader(rep_dir, args)
        
        temp_manager = TempFileManager(loader.data['temp_dir'])

        with Preprocessor(loader.data, args) as preprocessor:
            preprocessor.run()

        with Converter(loader.data, args) as converter:
            converter.run()
            
        with Resolver(loader.data['temp_dir'], loader.data['resolution_frame'], args.output, args.include_components) as resolver:
            resolver.run()
        
        temp_manager.cleanup()
            

if __name__ == '__main__':
    main()