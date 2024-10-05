
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from typing import Optional


# You need this to be able to turn classes into JSONs and return
from fastapi. encoders import jsonable_encoder 
from fastapi.responses import JSONResponse

fakeInvoiceTable = dict()

class Customer(BaseModel):
    customer_id : str
    country : str

class URLlink():
    url :  str

class Invoice(BaseModel):
    invoice_no : int
    invoice_date : str
    customer : Optional[URLlink] = None


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello"}

@app.post("/customer")
async def create_customer(item : Customer):  
    json_compatible_item_data = jsonable_encoder(item)
    return JSONResponse(content=json_compatible_item_data, status_code=201)



@app.get("/customer/{customer_id}")
async def get_customer(customer_id : str):
    if customer_id == "1234":
        item = Customer(customer_id="1234",country="UK")
        json_compatible_item_data = jsonable_encoder(item)
        return JSONResponse(content=json_compatible_item_data)
    else:
        raise HTTPException(status_code=404, detail="Item not found")


# Create a new invoice for a customer
@app.post("/customer/{customer_id}/invoice")
async def create_invoice(customer_id: str, invoice: Invoice):
    
    # Add the customer link to the invoice
    invoice.customer.url = "/customer/" + customer_id
    
    # Turn the invoice instance into a JSON string and store it
    jsonInvoice = jsonable_encoder(invoice)
    fakeInvoiceTable[invoice.invoice_no] = jsonInvoice

    # Read it from the store and return the stored item
    ex_invoice = fakeInvoiceTable[invoice.invoice_no]
    
    return JSONResponse(content=ex_invoice)
