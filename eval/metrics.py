import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.structures import *
from build_prooftree import *


def length_metric(thm):
    return len(thm.proof)

def modularity_metric(thm):
    _,_,_,depth = getProofTree(thm)
    return depth


def percent_change(old,new):
    return ((new-old)/old) * 100
