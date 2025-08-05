import glob
from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from auth import ACCESS_TOKEN_EXPIRE_MINUTES, create_access_token, get_current_user, get_current_role
from Split_Mod import  PDFProcessor
import hashlib
import psycopg2
import os
from datetime import datetime, timedelta
from typing import Optional
import asyncpg
import bcrypt
import torch
import base64
from PIL import Image, ImageEnhance
from io import BytesIO
import io
from byaldi import RAGMultiModalModel
import aiofiles
import aiohttp
from aiohttp import ClientTimeout
import ollama
import uvicorn
import asyncio
import traceback
from pydantic import BaseModel
import logging
import time
import aiofiles.os as aios
import shutil, uuid
import csv
import time
from typing import List, Dict
from pathlib import Path

Questions=[
    "When looking onto the output shaft, what is the standard direction of rotation?",
    "What is an example case to contact SEW-EURODRIVE?",
    "What is the description for right-angle gear unit with the designation 'WHF..'?",
    "What is the View of output at end B of a 'W' Series gear unit at stage 3?",
    "What is the Flange diameter of the gear unit SF67p?",
    "What is the tightening torque in Nm for gear unit, SF87p with flange diameter of 350 and using screw/nut M16?",
    "What are the steps to mount a shaft-mounted gear unit with splined hollow shaft?",
    "What are the steps to activating the breather valve?",
    "What are the steps to mounting the cover?",
]


Pages=[
    r"C:\Users\sgdrig01\Desktop\AI App Internship project\Testing Data\SPRIOPLAN_gearmotors_pics\page_33.png",
    r"C:\Users\sgdrig01\Desktop\AI App Internship project\Testing Data\SPRIOPLAN_gearmotors_pics\page_36.png",
    r"C:\Users\sgdrig01\Desktop\AI App Internship project\Testing Data\SPRIOPLAN_gearmotors_pics\page_29.png",
    r"C:\Users\sgdrig01\Desktop\AI App Internship project\Testing Data\SPRIOPLAN_gearmotors_pics\page_34.png",
    r"C:\Users\sgdrig01\Desktop\AI App Internship project\Testing Data\SPRIOPLAN_gearmotors_pics\page_37.png",
    r"C:\Users\sgdrig01\Desktop\AI App Internship project\Testing Data\SPRIOPLAN_gearmotors_pics\page_39.png",
    r"C:\Users\sgdrig01\Desktop\AI App Internship project\Testing Data\SPRIOPLAN_gearmotors_pics\page_55.png",
    r"C:\Users\sgdrig01\Desktop\AI App Internship project\Testing Data\SPRIOPLAN_gearmotors_pics\page_45.png",
    r"C:\Users\sgdrig01\Desktop\AI App Internship project\Testing Data\SPRIOPLAN_gearmotors_pics\page_82.png",
]

Parameters=[
    {
        "num_ctx": 512,
        "temperature": 1.35,
        "top_k":50,
        "repeat_penalty": 1.0,
        "mirostat_mode": 2.0,
        "mirostat_tau": 3.0,
        "seed": 560
    },
    {
        "num_ctx": 512,
        "temperature": 1.3,
        "top_k":50,
        "repeat_penalty": 1.0,
        "mirostat_mode": 2.0,
        "mirostat_tau": 3.0,
        "seed": 561
    },
    {
        "num_ctx": 512,
        "temperature": 1.25,
        "top_k":50,
        "repeat_penalty": 1.0,
        "mirostat_mode": 2.0,
        "mirostat_tau": 3.0,
        "seed": 562
    },
]


input={
    "Question": Questions,
    "Page": Pages,
}

################################################################################################
question=[]
answer1=[]
time_taken1=[]
answer2=[]
time_taken2=[]
answer3=[]
time_taken3=[]

output={
    "question": question,
    "answer1": answer1,
    "time_taken1": time_taken1,
    "answer2": answer2,
    "time_taken2": time_taken2,
    "answer3": answer3,
    "time_taken3": time_taken3,
}

################################################################################################

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Completely disable CUDA
torch.cuda.is_available = lambda: False  # Override CUDA check
device = torch.device("cpu")

async def analyze_image(
    image: UploadFile = File(...),
    query: str = Form(...)
):
    # Validate file type
    if not image.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Only image files (JPEG, PNG) are allowed"
        )

    try:
        start_time = time.time()
        print("\nStarting image analysis...")

        # Read and encode the image file
        image_data = await image.read()
        image_b64 = base64.b64encode(image_data).decode('utf-8')

        # Call the existing function with the converted base64
        result = await get_multimodal_res(image_b64, query)

        elapsed = time.time() - start_time
        print(f"Image analysis completed in {elapsed:.2f} seconds")

        return result
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Failed to process image: {str(e)}"
            }
        )
##########################################################################################
 
def enhance_image_b64(original_b64: str, debug: bool = False) -> str:
    """
    Enhance image quality (contrast + sharpness) while keeping base64 format.
    
    Args:
        original_b64: Base64 encoded image string
        
    Returns:
        Enhanced base64 image string (PNG format)
    """
    try:
        # Decode base64
        image_data = base64.b64decode(original_b64)
        img = Image.open(io.BytesIO(image_data))
        
        if debug:  # Show original
            img.show(title="ORIGINAL INPUT")

        # Convert to grayscale if needed (better for text/number clarity)
        if img.mode != 'L':
           img = img.convert('L')  # 'L' = 8-bit grayscale
        
        saturation = ImageEnhance.Color(img)
        contrast = ImageEnhance.Contrast(img)
        #brightness = ImageEnhance.Brightness(img)
        #sharpness = ImageEnhance.Sharpness(img)

        img = saturation.enhance(0.5)
        #img = contrast.enhance(0.9)  
        
        
        # Sharpen edges (helps distinguish similar numbers)
        # img = img.filter(ImageFilter.SHARPEN)
        
        if debug:  # Show enhanced
            img.show(title="ENHANCED OUTPUT (What model sees)")

        # Optional: Uncomment to binarize (black/white only)
        # img = img.point(lambda x: 0 if x < 180 else 255, '1')  # Adjust threshold
        
        # Re-encode to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")  # PNG preserves quality
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    except Exception as e:
        # Return original if enhancement fails
        print(f"Enhancement failed (using original): {str(e)}")
        return original_b64


async def get_multimodal_res(image_b64: str, query: str, params: dict = None) -> dict:
    prompt = f"""<|im_start|>system
You are a form data extractor. Your response must contain ONLY these 2 lines:

Field: [exact field name from image]
Requirement: [visible text OR "Not specified"]

RULES:
1. NEVER add:
   - Prefixes like "Answer:"
   - Explanations
   - Unrequested analysis
2. If field exists but text is unclear: "Requirement: Not clearly visible"
3. Maximum 40 words total<|im_end|>
<|im_start|>user
Locate and extract: {query}<|im_end|>
<|im_start|>assistant
Field:  """

    if not image_b64 or not query:
        return {
            "status": "error",
            "message": "Missing required parameters: image_b64 or query."
        }

    model_name = "llama3.2-vision:11b-instruct-q8_0"
    base_url = "http://localhost:11434/api"
    
    # Default parameters (will be overridden by params argument)
    default_params = {
        "num_ctx": 512,
        "temperature": 0.8,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "mirostat_mode": 0,
        "mirostat_tau": 0,
        "seed": 0
    }
    
    # Update defaults with provided params
    if params:
        default_params.update(params)

    # Increased timeouts
    model_check_timeout = ClientTimeout(total=60)
    model_pull_timeout = ClientTimeout(total=1800)
    generation_timeout = ClientTimeout(total=600)
    
    try:
        #enhanced_b64 = enhance_image_b64(image_b64)
        async with aiohttp.ClientSession() as session:
            # Step 1: Check if model exists
            try:
                async with session.get(
                    f"{base_url}/tags",
                    timeout=model_check_timeout
                ) as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to fetch model tags: {await resp.text()}")

                    models_json = await resp.json()
                    model_names = [m.get("name") for m in models_json.get("models", [])]
                    
                    if model_name not in model_names:
                        print(f"Model not found. Pulling {model_name}...")
                        async with session.post(
                            f"{base_url}/pull",
                            json={"name": model_name},
                            timeout=model_pull_timeout
                        ) as pull_resp:
                            if pull_resp.status != 200:
                                raise Exception(f"Failed to pull model: {await pull_resp.text()}")
                            
                            async for line in pull_resp.content:
                                if b'"status":"success"' in line:
                                    break
                        print("Model pulled successfully.")
            except asyncio.TimeoutError:
                return {
                    "status": "error",
                    "message": "Timeout while checking/pulling the model."
                }
            except Exception as model_err:
                return {
                    "status": "error",
                    "message": f"Model check/pull failed: {str(model_err)}"
                }

            # Step 2: Generate response with the provided parameters
            payload = {
                "model": model_name,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "num_ctx": default_params.get("num_ctx"),
                    "temperature": default_params.get("temperature"),
                    "top_k": default_params.get("top_k"),
                    "repeat_penalty": default_params.get("repeat_penalty"),
                    "mirostat_mode": default_params.get("mirostat_mode"),
                    "mirostat_tau": default_params.get("mirostat_tau"),
                    "seed": default_params.get("seed")
                }
            }

            try:
                async with session.post(
                    f"{base_url}/generate",
                    json=payload,
                    timeout=generation_timeout
                ) as resp:
                    if resp.status != 200:
                        error_detail = await resp.text()
                        return {
                            "status": "error",
                            "message": f"Model response failed: {error_detail}"
                        }

                    result = await resp.json()
                    answer = result.get("response", "").strip()
                    if not answer:
                        return {
                            "status": "error",
                            "message": "Empty response from the model."
                        }

                    return {
                        "status": "success",
                        "answer": answer,
                        "image_b64": image_b64,
                        "query_used": query,
                        "parameters_used": default_params
                    }
            except asyncio.TimeoutError:
                return {
                    "status": "error",
                    "message": "The model took too long to respond."
                }

    except Exception as e:
        tb = traceback.format_exc()
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}",
            "traceback": tb
        }

async def run_automated_testing():
    """Main function to run the automated parameter testing"""
    results_dir = Path(__file__).parent / "test_results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"test_results_{timestamp}.csv"
    
    fieldnames = [
        "parameter_variant", "question", "image_path",
        "answer1", "time_taken1", "answer2", "time_taken2", "answer3", "time_taken3",
        "parameters_used"
    ]
    
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for param_idx, params in enumerate(Parameters, 1):
            print(f"\n{'='*50}")
            print(f"Testing Parameter Variant {param_idx}/{len(Parameters)}")
            print(f"Parameters: {params}")
            
            for question, image_path in zip(Questions, Pages):
                print(f"\nProcessing Question: {question[:50]}...")
                
                question_results = {
                    "parameter_variant": f"Variant_{param_idx}",
                    "question": question,
                    "image_path": image_path,
                    "parameters_used": str(params)
                }
                
                for attempt in range(1, 4):
                    try:
                        async with aiofiles.open(image_path, mode='rb') as f:
                            image_data = await f.read()
                        
                        # Encode image directly instead of using UploadFile
                        image_b64 = base64.b64encode(image_data).decode('utf-8')
                        
                        start_time = time.time()
                        result = await get_multimodal_res(
                            image_b64=image_b64,
                            query=question,
                            params=params
                        )
                        elapsed = time.time() - start_time
                        
                        if result["status"] == "success":
                            question_results[f"answer{attempt}"] = result["answer"]
                            question_results[f"time_taken{attempt}"] = elapsed
                        else:
                            question_results[f"answer{attempt}"] = f"ERROR: {result['message']}"
                            question_results[f"time_taken{attempt}"] = elapsed
                            
                    except Exception as e:
                        question_results[f"answer{attempt}"] = f"EXCEPTION: {str(e)}"
                        question_results[f"time_taken{attempt}"] = 0
                        print(f"Attempt {attempt} failed: {str(e)}")
                
                writer.writerow(question_results)
                csvfile.flush()
    
    print(f"\nTesting complete! Results saved to: {csv_path}")

# Add this to run when the script is executed directly
if __name__ == "__main__":
    asyncio.run(run_automated_testing())