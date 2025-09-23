# axolotl/integrations/weightedSFT/args.py
from typing import Optional
from pydantic import BaseModel

class WeightedSFTArgs(BaseModel):
    """
    Input args for Weighted SFT plugin.
    """
    normalize_batch_weights: Optional[bool] = True