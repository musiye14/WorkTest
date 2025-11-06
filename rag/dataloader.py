from langchain_community.document_loaders import PyPDFLoader 
from enum import Enum
from chunkbase import ChunkerBase
import os


class FileType(Enum):
    TXT="txt"
    PDF="pdf"
    CSV="csv"

    @classmethod
    def from_extension(cls,ext:str):
        try:
            return cls(ext.lower())
        except Exception as e:
            print(f"不支持的文件类型:{ext}")

class DataLoader():
    def __init__(self,filepath:str, chunker:ChunkerBase) -> None:
        self.file_type = self.get_file_type(filepath)
        self.filepath = filepath
        self.chunker = chunker


    def get_file_type(self,filepath:str)->str:

        _,ext_with_dot = os.path.splitext(filepath)

        if not ext_with_dot:
            print(f"找不到后缀:{filepath}")
        ext = ext_with_dot[1:].lower()

        return FileType.from_extension(ext)

    def chunke(self):
        self.chunker.chunker
        pass
    


try:
    loader1 = DataLoader("document.pdf")
    loader2 = DataLoader("/path/to/my_report.txt")
    
    # 尝试不支持的文件类型或没有扩展名的文件
    loader4 = DataLoader("image.jpg") 
    loader3 = DataLoader("spreadsheet.csv")

    
    print(f"\n文件 'document.pdf' 的枚举类型是: {loader1.file_type}")
    print(f"这个枚举类型的名称是: {loader1.file_type.name}") # TXT, PDF, etc.
    print(f"这个枚举类型的值是: {loader1.file_type.value}") # txt, pdf, etc.
    # 运行时断言
    assert loader1.file_type == FileType.PDF
    assert loader2.file_type.value == "txt"
except ValueError as e:
    print(f"处理文件时发生错误: {e}")


