import sys

from logparser import Drain

input_dir = "project_raw/"  # The input directory of log file
output_dir = "project_parsed/"  # The output directory of parsing results
log_file_all = "BGL.log"  # The input log file name
log_file_train = (
    "BGL_train.log"  # The input log file name containing only the training data
)
log_format = "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>"  # BGL log format
# Regular expression list for optional preprocessing (default: [])
regex = [r"core\.\d+"]
st = 0.5  # Similarity threshold
depth = 4  # Depth of all leaf nodes

# run on training dataset
parser = Drain.LogParser(
    log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex
)
parser.parse(log_file_all)

# run on complete dataset
parser = Drain.LogParser(
    log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex
)
parser.parse(log_file_train)
