from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from uuid import uuid4
import os
from dotenv import load_dotenv
import numpy as np

import pymongo
import certifi

# Load MongoDB URI from .env
MONGO_URI = os.getenv("MONGO_URI")

# Establish connection with SSL certificate handling
client = pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where())

# Check if connection is successful
try:
    client.server_info()
    print("✅ MongoDB Connection Successful")
except Exception as e:
    print(f"❌ MongoDB Connection Failed: {e}")


load_dotenv()

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Setup
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["group_chat"]
students_collection = db["students"]
groups_collection = db["groups"]

# Pinecone Setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("groupchat")

# Model Setup
model = SentenceTransformer('all-MiniLM-L6-v2')

class Student(BaseModel):
    name: str
    email: str
    college: str
    interests: list[str]
    skills: list[str]
    preferred_group: str
    future_goals: str

class Group(BaseModel):
    name: str
    description: str
    tags: list[str]
    max_members: int = 10

def generate_embedding(text: str) -> list[float]:
    return model.encode(text).tolist()

def create_group_embedding(members: list[str]) -> list[float]:
    member_embeddings = []
    for member_id in members:
        student = students_collection.find_one({"_id": member_id})
        if student and "embedding" in student:
            member_embeddings.append(student["embedding"])
    return np.mean(member_embeddings, axis=0).tolist() if member_embeddings else None

@app.post("/add_student")
async def add_student(student: Student):
    try:
        text = f"{student.interests} {student.skills} {student.preferred_group} {student.future_goals}"
        embedding = generate_embedding(text)
        student_id = str(uuid4())
        student_dict = student.model_dump()
        student_dict["_id"] = student_id
        student_dict["embedding"] = embedding
        students_collection.insert_one(student_dict)
        index.upsert([(student_id, embedding)])
        return {"message": "Student added successfully", "student_id": student_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/find_groups/{student_id}")
async def find_groups(student_id: str):
    try:
        student = students_collection.find_one({"_id": student_id})
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")
        
        results = index.query(vector=student["embedding"], top_k=5, include_metadata=True)
        matched_groups = []
        
        for match in results["matches"]:
            group = groups_collection.find_one({"_id": match["id"]})
            if group:
                matched_groups.append({
                    "group_id": group["_id"],
                    "name": group["name"],
                    "description": group["description"],
                    "tags": group.get("tags", []),
                    "current_members": len(group.get("members", [])),
                    "max_members": group["max_members"],
                    "status": "Available" if len(group.get("members", [])) < group["max_members"] else "Full"
                })
        return {"matched_groups": matched_groups}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/join_group/{student_id}/{group_id}")
async def join_group(student_id: str, group_id: str):
    try:
        student = students_collection.find_one({"_id": student_id})
        group = groups_collection.find_one({"_id": group_id})
        
        if not student or not group:
            raise HTTPException(status_code=404, detail="Student or Group not found")
        
        if len(group.get("members", [])) >= group["max_members"]:
            return {"message": "Group is full"}
        
        groups_collection.update_one({"_id": group_id}, {"$push": {"members": student_id}})
        new_embedding = create_group_embedding(group.get("members", []) + [student_id])
        
        if new_embedding:
            index.upsert([(group_id, new_embedding)])
        
        return {"message": f"{student['name']} joined {group['name']}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/create_group/{student_id}")
async def create_group(student_id: str, group: Group):
    try:
        student = students_collection.find_one({"_id": student_id})
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")
        
        group_id = str(uuid4())
        group_dict = group.model_dump()
        group_dict["_id"] = group_id
        group_dict["members"] = [student_id]
        group_dict["embedding"] = student["embedding"]
        groups_collection.insert_one(group_dict)
        index.upsert([(group_id, group_dict["embedding"])])
        
        return {"message": "Group created", "group_id": group_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")