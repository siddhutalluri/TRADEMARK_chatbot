from pydantic import BaseModel
# 2. Class which describes Bank Notes measurements
class inp_txt(BaseModel):
    text : str