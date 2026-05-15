from pydantic import BaseModel


class CustomerRequest(BaseModel):

    Recency: int

    Frequency: int

    Monetary: float

    Age: int

    CustAccountBalance: float