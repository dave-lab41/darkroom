#!/bin/sh

# Data preparation scripts
./data/VOC0712/create_list.sh
./data/VOC0712/create_data.sh

# Begin training inside jupyter by opening a terminal and executing 'python examples/ssd/ssd_pascal.py'
/usr/local/bin/jupyter notebook --allow-root --port=8888 --ip=0.0.0.0 --no-browser

# Instead of going through jupyter, this script begins SSD training on the GPUs with default 300x300 for 120k iterations
# See examples/ssd/ssd_pascal_xxx.py for an example of 450x450 for 120 iterations. GPU memory requirements will be higher. 
#python examples/ssd/ssd_pascal.py

