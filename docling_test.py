import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from docling_core.types.doc.document import DocTagsDocument
from docling_core.types.doc import DoclingDocument
from PIL import Image
from pathlib import Path
import json

# 🔧 Config
MODEL = r"C:\Users\sgdrig01\Desktop\AI App Internship project\AI server\smoling\SmolDocling-256M-preview"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROMPT = "Convert page to Docling."

def load_image(path):
    return Image.open(path).convert("RGB")

def process_image(image):
    processor = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForVision2Seq.from_pretrained(MODEL).to(DEVICE)

    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
        add_generation_prompt=True
    )
    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt",
        truncation=True
    ).to(DEVICE)

    outputs = model.generate(**inputs, max_new_tokens=2048)
    gen = outputs[0, inputs.input_ids.shape[1]:]
    return processor.batch_decode(gen.unsqueeze(0), skip_special_tokens=False)[0].lstrip()

def main(img_path):
    img = load_image(img_path)
    tags = process_image(img)

    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([tags], [img])
    doc = DoclingDocument(name="ParsedDoc")
    doc.load_from_doctags(doctags_doc)

    # 📝 1. Save Markdown directly:
    doc.save_as_markdown("parsed_page.md")
    print("✅ Markdown saved to parsed_page.md")

    # 2. Save structured JSON directly:
    doc.save_as_json("parsed_page.json", image_mode=None)
    print("✅ JSON saved to parsed_page.json")

    # 3. Or, if you still want to use export_to_dict():
    data = doc.export_to_dict()
    with open("parsed_page_dict.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("✅ Dict JSON saved to parsed_page_dict.json")

if __name__ == "__main__":
    main(r"C:\Users\sgdrig01\Downloads\page 8.png")