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
import certifi
import logging

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB Connection
try:
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    db = client["group_chat"]
    students_collection = db["students"]
    groups_collection = db["groups"]
    logger.info("✅ MongoDB Connection Successful")
except Exception as e:
    logger.error(f"❌ MongoDB Connection Failed: {e}")

# Pinecone Setup
try:
    pc = Pinecone(api_key=PINECONE_API_KEY, environment="us-east-1")  # Adjust based on your Pinecone index region

    index_name = "groupchat"  # Make sure this matches your Pinecone index name
    index = pc.Index(index_name)
    logger.info("✅ Pinecone Connection Successful")
except Exception as e:
    logger.error(f"❌ Pinecone Connection Failed: {e}")

# Sentence Transformer Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# FastAPI App
app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    """Generate sentence embeddings using SentenceTransformer."""
    embedding = model.encode(text).tolist()
    logger.info(f"Generated embedding for text: {text}")
    return embedding

def create_group_embedding(members: list[str]) -> list[float]:
    """Calculate average embedding for a group."""
    member_embeddings = []
    for member_id in members:
        student = students_collection.find_one({"_id": member_id})
        if student and "embedding" in student:
            member_embeddings.append(student["embedding"])
    if member_embeddings:
        group_embedding = np.mean(member_embeddings, axis=0).tolist()
        logger.info(f"Generated group embedding: {group_embedding}")
        return group_embedding
    else:
        logger.warning("No member embeddings found for group")
        return None

@app.post("/add_student")
async def add_student(student: Student):
    try:
        # Generate embedding from student data
        text = f"{student.interests} {student.skills} {student.preferred_group} {student.future_goals}"
        embedding = generate_embedding(text)
        
        # Create student document
        student_id = str(uuid4())
        student_dict = student.model_dump()
        student_dict["_id"] = student_id
        student_dict["embedding"] = embedding
        
        # Insert student into MongoDB
        students_collection.insert_one(student_dict)
        logger.info(f"Student added to MongoDB: {student_id}")
        
        # Upsert student embedding into Pinecone
        index.upsert([(student_id, embedding)])
        logger.info(f"Student upserted to Pinecone: {student_id}")
        
        return {"message": "Student added successfully", "student_id": student_id}
    except Exception as e:
        logger.error(f"Error adding student: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/find_groups/{student_id}")  # Change POST to GET
async def find_groups(student_id: str):
    try:
        # Fetch student from MongoDB
        student = students_collection.find_one({"_id": student_id})
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")
        
        # Query Pinecone for similar groups
        results = index.query(vector=[student["embedding"]], top_k=5, include_metadata=True, namespace="")
        logger.info(f"Pinecone query results: {results}")
        
        # Fetch matched groups from MongoDB
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
        
        logger.info(f"Matched groups: {matched_groups}")
        return {"matched_groups": matched_groups}
    except Exception as e:
        logger.error(f"Error finding groups: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
@app.post("/join_group/{student_id}/{group_id}")
async def join_group(student_id: str, group_id: str):
    try:
        # Fetch student and group from MongoDB
        student = students_collection.find_one({"_id": student_id})
        group = groups_collection.find_one({"_id": group_id})
        
        if not student or not group:
            raise HTTPException(status_code=404, detail="Student or Group not found")
        
        # Check if group is full
        if len(group.get("members", [])) >= group["max_members"]:
            return {"message": "Group is full"}
        
        # Add student to group
        groups_collection.update_one({"_id": group_id}, {"$push": {"members": student_id}})
        logger.info(f"Student {student_id} joined group {group_id}")
        
        # Update group embedding in Pinecone
        new_embedding = create_group_embedding(group.get("members", []) + [student_id])
        if new_embedding:
            index.upsert([(group_id, new_embedding)])
            logger.info(f"Updated group embedding in Pinecone: {group_id}")
        
        return {"message": f"{student['name']} joined {group['name']}"}
    except Exception as e:
        logger.error(f"Error joining group: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/create_group/{student_id}")
async def create_group(student_id: str, group: Group):
    try:
        # Fetch student from MongoDB
        student = students_collection.find_one({"_id": student_id})
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")
        
        # Create group document
        group_id = str(uuid4())
        group_dict = group.model_dump()
        group_dict["_id"] = group_id
        group_dict["members"] = [student_id]
        group_dict["embedding"] = student["embedding"]
        
        # Insert group into MongoDB
        groups_collection.insert_one(group_dict)
        logger.info(f"Group created in MongoDB: {group_id}")
        
        # Upsert group embedding into Pinecone
        index.upsert([(group_id, group_dict["embedding"])])
        logger.info(f"Group upserted to Pinecone: {group_id}")
        
        return {"message": "Group created", "group_id": group_id}
    except Exception as e:
        logger.error(f"Error creating group: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")