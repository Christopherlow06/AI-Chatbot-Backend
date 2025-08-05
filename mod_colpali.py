from pathlib import Path
from typing import Union, List, Optional, Dict
from byaldi import RAGMultiModalModel
from byaldi.objects import Result
from byaldi.colpali import ColPaliModel
import torch
import os
from colpali_engine.models import ColPali, ColPaliProcessor

class PersistentColPali(ColPaliModel):
    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, Path],
        n_gpu: int = -1,
        verbose: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        index_root: str = r"C:\Users\sgdrig01\Desktop\indexed_data",  # Default to your desktop
        **kwargs,
    ):
        # Create directory if it doesn't exist
        self._index_root = str(Path(index_root).absolute())
        os.makedirs(self._index_root, exist_ok=True)

        self.index_name = None
        self.current_index_path = None
        self.full_document_collection = False
        self.highest_doc_id = -1
        self.collection = {}
        self.indexed_embeddings = []
        self.embed_id_to_doc_id = {}
        self.doc_id_to_metadata = {}
        self.doc_ids_to_file_names = {}
        self.doc_ids = set()
        
        # Try different initialization approaches
        try:
            # First try standard initialization
            super().__init__(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                n_gpu=n_gpu,
                verbose=verbose,
                load_from_index=False,
                index_root=self._index_root,
                device=device,
                **kwargs,
            )
        except TypeError:
            # If that fails, try minimal initialization
            super().__init__()
            # Manually set attributes
            self.pretrained_model_name_or_path = pretrained_model_name_or_path
            self.n_gpu = n_gpu
            self.verbose = verbose
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.index_root = index_root
            self.kwargs = kwargs
        
        self.current_index_path = None
        self.index_name = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        n_gpu: int = -1,
        verbose: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        index_root: str = r"C:\Users\sgdrig01\Desktop\indexed_data",
        **kwargs,
    ):
        
        index_root = str(Path(index_root).absolute())
        os.makedirs(index_root, exist_ok=True)
        """Alternative factory method that bypasses parent class issues"""
        instance = cls.__new__(cls)
        
        # Direct attribute setting
        instance.pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        instance.n_gpu = n_gpu
        instance.verbose = verbose
        instance.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        instance.index_root = index_root
        instance.kwargs = kwargs
        instance.current_index_path = None
        
        # Initialize model components
        if "colpali" in instance.pretrained_model_name_or_path.lower():
            instance.model = ColPali.from_pretrained(
                instance.pretrained_model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
            )
            instance.processor = ColPaliProcessor.from_pretrained(
                instance.pretrained_model_name_or_path,
                token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
            )
        else:
            raise ValueError("Unsupported model type")
            
        instance.model = instance.model.eval()
        return instance

    def create_index(
        self,
        input_path: Union[str, Path],
        store_collection_with_index: bool = False,
        metadata: Optional[List[Dict[str, Union[str, int]]]] = None,
        max_image_width: Optional[int] = None,
        max_image_height: Optional[int] = None,
    ) -> Path:
        """Create and store index with automatic naming"""
        input_path = Path(input_path)
        pdf_name = input_path.stem
        index_name = f"{pdf_name}_index"
        index_path = Path(self.index_root) / index_name
        
        # Force clean state for new index
        self.index_name = None
        self.current_index_path = None
        
        # Call parent's index method with overwrite=True
        try:
            result = super().index(
                input_path=input_path,
                index_name=index_name,
                store_collection_with_index=store_collection_with_index,
                metadata=metadata,
                max_image_width=max_image_width,
                max_image_height=max_image_height,
                overwrite=True  # Force overwrite if exists
            )
        except ValueError as e:
            if "already loaded" in str(e):
                # Clear existing index and try again
                self.index_name = None
                result = super().index(
                    input_path=input_path,
                    index_name=index_name,
                    store_collection_with_index=store_collection_with_index,
                    metadata=metadata,
                    max_image_width=max_image_width,
                    max_image_height=max_image_height,
                    overwrite=True
                )
            else:
                raise
        
        self.index_name = index_name
        self.current_index_path = index_path
        
        # Verify index was created on disk
        if not index_path.exists():
            raise RuntimeError(f"Index directory was not created at {index_path}")
            
        print(f"Index successfully created at: {index_path}")
        print(f"Directory contents: {os.listdir(index_path)}")
        
        return index_path

    def load_index(self, pdf_path: Union[str, Path]):
        """Load index using parent's from_index mechanism"""
        pdf_path = Path(pdf_path)
        pdf_name = pdf_path.stem
        index_name = f"{pdf_name}_index"
        index_dir = Path(self._index_root) / index_name
        
        if not index_dir.exists():
            raise FileNotFoundError(f"Index directory {index_dir} not found")
        
        # Use parent's from_index
        loaded = self.__class__.from_index(
            index_path=index_dir,
            n_gpu=self.n_gpu,
            verbose=self.verbose,
            device=self.device,
            index_root=str(index_dir.parent),
        )
        
        # Transfer state
        self.__dict__.update(loaded.__dict__)
        self.current_index_path = index_dir
        self.index_name = index_name

    def list_indexes(self):
        """List all available indexes in the index_root"""
        return [d.name for d in Path(self.index_root).glob("*_index") if d.is_dir()]
    
    def delete_index(self, pdf_path: Union[str, Path]):
        """Delete index associated with a PDF"""
        pdf_name = Path(pdf_path).stem
        index_dir = Path(self.index_root) / f"{pdf_name}_index"
        if index_dir.exists():
            import shutil
            shutil.rmtree(index_dir)

    def search_using_pdf(
        self,
        query: Union[str, List[str]],
        pdf_path: Optional[Union[str, Path]] = None,
        k: int = 10,
        filter_metadata: Optional[Dict[str, str]] = None,
        return_base64_results: Optional[bool] = None,
    ) -> Union[List[Result], List[List[Result]]]:
        """
        Search using index associated with a PDF file
        """
        if pdf_path:
            self.load_index(pdf_path)
        elif not self.current_index_path:
            raise RuntimeError("No index loaded and no PDF path provided")
        
        return self.search(
            query=query,
            k=k,
            filter_metadata=filter_metadata,
            return_base64_results=return_base64_results
        )