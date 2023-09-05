from pydantic import BaseModel

class banknotes(BaseModel):
    variance: float 
    skewness: float 
    curtosis: float 
    entropy: float