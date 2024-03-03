from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
import ast

load_dotenv()

db = SQLDatabase.from_uri("sqlite:///Chinook.db")

llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
chain = create_sql_query_chain(llm, db)
query = "資料庫中共有幾位藝術家? (Artist)"
print(query)
response = chain.invoke({"question": query})

response = response.replace('"', '') + ";"
print(response)

result = ast.literal_eval(db.run(response))[0][0]
print(result)
