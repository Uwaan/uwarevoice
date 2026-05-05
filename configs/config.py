"""Wrapper class for RVC object, which expects these parameters

"RVC for realtime" uses this class to do a whole bunch of stuff, might look
into implementation, but for now we are tracking these things ourself so
just put them in a format that RVC understands
"""


class Config:
    def __init__(
        self,
        device: str = "cuda:0",
        is_half: bool = True,
        use_jit: bool = False,
        dml: bool = False,
    ) -> None:
        self.device = device
        self.is_half = is_half
        self.use_jit = use_jit
        self.dml = dml
