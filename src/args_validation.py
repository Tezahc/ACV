import argparse

def args_validation(parser):
    
    parser.add_argument('-i', '--input_video_file', type=str, default=0)
    parser.add_argument('-o', '--output_video_file', type=str, required=False)

    return parser.parse_args()