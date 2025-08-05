import asyncio
import os
import shutil
from pathlib import Path
from typing import Set
from concurrent.futures import ThreadPoolExecutor
import PyPDF2
from pdf2image import convert_from_path
import aiofiles
import aiofiles.os as aios

VENV_BASE = r"C:\Users\sgdrig01\Desktop\AI App Internship project\AI server\venv"
POPPLER_PATH = os.path.join(VENV_BASE, "poppler-24.08.0", "Library", "bin")
os.environ["PATH"] += os.pathsep + POPPLER_PATH
print(f"Poppler: {POPPLER_PATH} ({'Exists' if os.path.exists(POPPLER_PATH) else 'Missing'})")

class PDFProcessor:
    def __init__(self, input_dir: str, output_dir: str, max_workers: int = 4):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def is_already_processed(self, pdf_file: str) -> bool:
        """Check if PDF has already been fully processed"""
        pdf_name = os.path.splitext(pdf_file)[0]
        pdf_output_dir = os.path.join(self.output_dir, pdf_name)
        
        # Check if both the PDF and images directory exist
        pdf_exists = await aios.path.exists(os.path.join(pdf_output_dir, pdf_file))
        images_dir_exists = await aios.path.exists(os.path.join(pdf_output_dir, 'images'))
        
        if not (pdf_exists and images_dir_exists):
            return False
            
        # Verify at least one image exists
        try:
            images = await aios.listdir(os.path.join(pdf_output_dir, 'images'))
            return len(images) > 0
        except FileNotFoundError:
            return False

    async def process_single_pdf(self, pdf_file: str):
        """Process a single PDF file"""
        if await self.is_already_processed(pdf_file):
            print(f"Skipping {pdf_file} (already processed)")
            return True

        pdf_path = os.path.join(self.input_dir, pdf_file)
        pdf_name = os.path.splitext(pdf_file)[0]
        pdf_output_dir = os.path.join(self.output_dir, pdf_name)
        images_dir = os.path.join(pdf_output_dir, 'images')
        
        try:
            # Create directories
            if await aios.path.exists(pdf_output_dir):
                await aios.rmtree(pdf_output_dir)
            await aios.makedirs(images_dir, exist_ok=True)
            
            print(f"Starting processing {pdf_file}...")
            
            # Convert PDF to images (run in executor)
            def convert_pdf():
                return convert_from_path(pdf_path)
            
            loop = asyncio.get_running_loop()
            images = await loop.run_in_executor(self.executor, convert_pdf)
            
            # Save images
            for i, image in enumerate(images):
                output_path = os.path.join(images_dir, f"page_{i+1}.png")
                image.save(output_path, 'PNG')
            
            # Move PDF to output directory
            dest_pdf_path = os.path.join(pdf_output_dir, pdf_file)
            await self.move_file(pdf_path, dest_pdf_path)
            
            print(f"Completed processing {pdf_file}")
            return True
            
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            # Clean up if failed
            if await aios.path.exists(pdf_output_dir):
                await aios.rmtree(pdf_output_dir)
            return False
    
    async def move_file(self, src: str, dst: str):
        """Async file move with progress for large files"""
        if await aios.path.exists(dst):
            await aios.remove(dst)
            
        # For large files, show progress
        file_size = (await aios.stat(src)).st_size
        if file_size > 10 * 1024 * 1024:  # >10MB
            print(f"Moving {os.path.basename(src)} ({file_size/1024/1024:.1f}MB)...")
            
        async with aiofiles.open(src, 'rb') as f_src:
            async with aiofiles.open(dst, 'wb') as f_dst:
                while True:
                    chunk = await f_src.read(64 * 1024)  # 64KB chunks
                    if not chunk:
                        break
                    await f_dst.write(chunk)
        
        await aios.remove(src)
    
    async def process_all_pdfs(self):
        """Process all PDFs in input directory"""
        await aios.makedirs(self.output_dir, exist_ok=True)
        
        pdf_files = [
            f for f in await aios.listdir(self.input_dir) 
            if f.lower().endswith('.pdf')
        ]
        
        if not pdf_files:
            print("No new PDF files to process")
            return
        
        print(f"Found {len(pdf_files)} new PDF file(s) to process")
        
        # Process in parallel
        tasks = [self.process_single_pdf(pdf_file) for pdf_file in pdf_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if r is True)
        print(f"Processing complete. {success_count}/{len(pdf_files)} succeeded")

async def main():
    input_directory = r"C:\Users\sgdrig01\Desktop\AI App Internship project\AI server\pdf_files"
    output_directory = r"C:\Users\sgdrig01\Desktop\AI App Internship project\Processed_PDFs(DO NOT TOUCH)"
    
    processor = PDFProcessor(input_directory, output_directory, max_workers=4)
    await processor.process_all_pdfs()

if __name__ == "__main__":
    asyncio.run(main())