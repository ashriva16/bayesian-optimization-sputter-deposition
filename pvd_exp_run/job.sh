#!/bin/bash

source /home/ashriva/anaconda3/bin/activate mlenv2
python --version
which python

# Trial runs
# nohup python -u run_exp.py -opt 0
# nohup python -u run_exp.py -opt 1
# nohup python -u run_exp.py -opt 2
# nohup python -u run_exp.py -opt 3

# nohup python -u run_exp.py -man .15 .3 0
# nohup python -u run_exp.py -man .15 .3 1
# nohup python -u run_exp.py -man .15 .3 2
# nohup python -u run_exp.py -man .15 .3 3


# Final runs
# python -u run.py -man .15 .3 0
# python -u run.py -man .15 .3 1
# python -u run.py -man .15 .3 2
# python -u run.py -opt 3
# python -u run.py -opt 4
# python -u run.py -opt 5
# python -u run.py -opt 6
# python -u run.py -opt 7
# python -u run.py -opt 8
# python -u run.py -opt 9
# python -u run.py -opt 10

# ----------------------------------------------------------------
# python -u plots_exp.py 0
# python -u plots_exp.py 1
# python -u plots_exp.py 2
# python -u plots_exp.py 3
# python -u plots_exp.py 4
# python -u plots_exp.py 5
# python -u plots_exp.py 6
# python -u plots_exp.py 7
# python -u plots_exp.py 8
# python -u plots_exp.py 9
# python -u plots_exp.py 10