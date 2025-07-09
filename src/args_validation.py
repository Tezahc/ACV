import argparse

def args_validation(parser):
    
    parser.add_argument('-i', '--input_video_file', type=str, default=0)
    parser.add_argument('-o', '--output_video_file', type=str, required=False)
    parser.add_argument("-p", "--show_plots", action='store_true', help="Enable matplotlib plots")

    return parser.parse_args()