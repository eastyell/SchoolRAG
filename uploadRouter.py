'''
    创建时间：2023-07-20
    功能：上传文件并处理成知识库

'''

import os
import shutil
from typing import List
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import HTMLResponse
from langchain_zhipu import ZhipuAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# 全局配置（和主文件统一，也可抽单独config）
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
PERSIST_DIRECTORY = "./chroma_db_school"
COLLECTION_NAME = "school_knowledge"
UPLOAD_DIR = "./temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

print("正在初始化 Embedding 模型...")
embeddings = ZhipuAIEmbeddings(model="embedding-3", api_key=ZHIPU_API_KEY)
print("✅ 模型初始化完成！")

# 创建路由分组，统一前缀 /app_upload
router = APIRouter(
    prefix="/app_upload",
    tags=["文档上传知识库模块"]
)

# 接口地址变为：POST /app_upload/api/upload
@router.post("/api/upload")
async def upload_and_process_files(files: List[UploadFile] = File(...)):
    results = []
    all_chunks = []
    success_count = 0
    fail_count = 0
    
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    for file in files:
        filename = file.filename
        if not filename:
            continue
        if not filename.lower().endswith(('.pdf', '.docx')):
            results.append({"file": filename, "status": "error", "message": "格式不支持"})
            fail_count += 1
            continue

        file_path = os.path.join(UPLOAD_DIR, filename)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            if filename.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                loader = Docx2txtLoader(file_path)
            documents = loader.load()
            for doc in documents:
                doc.metadata["source"] = filename
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n\n", "\n", "。", "！", "？", "；", ".", " "]
            )
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)
            results.append({"file": filename, "status": "success", "chunks": len(chunks)})
            success_count += 1
        except Exception as e:
            results.append({"file": filename, "status": "error", "message": str(e)})
            fail_count += 1
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    if all_chunks:
        BATCH_SIZE = 32
        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[i: i + BATCH_SIZE]
            db.add_documents(batch)

    total_docs = db._collection.count()
    return {
        "message": f"批量处理完成！成功 {success_count} 个，失败 {fail_count} 个。",
        "details": results,
        "total_chunks_added": len(all_chunks),
        "total_docs_in_db": total_docs
    }

# 页面地址变为 GET /app_upload
@router.get("/", response_class=HTMLResponse)
def get_upload_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>学校知识库 - 批量文档上传</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0f2f5; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; padding: 20px; }
            .container { background: #fff; padding: 40px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); width: 100%; max-width: 600px; text-align: center; }
            h2 { color: #333; margin-bottom: 20px; }
            .upload-area { border: 2px dashed #ccc; border-radius: 8px; padding: 40px 20px; cursor: pointer; transition: border-color 0.3s; margin-bottom: 20px; }
            .upload-area:hover { border-color: #007bff; }
            .upload-area.dragover { border-color: #007bff; background-color: #f0f8ff; }
            input[type="file"] { display: none; }
            .file-name { font-size: 14px; color: #666; margin-top: 10px; word-break: break-all; }
            button { background: #007bff; color: white; border: none; padding: 12px 24px; border-radius: 5px; font-size: 16px; cursor: pointer; transition: background 0.3s; }
            button:disabled { background: #aaa; cursor: not-allowed; }
            .status { margin-top: 20px; padding: 15px; border-radius: 5px; display: none; font-size: 14px; text-align: left; line-height: 1.6; }
            .success { background: #d4edda; color: #155724; display: block; }
            .error { background: #f8d7da; color: #721c24; display: block; }
            .loading { background: #fff3cd; color: #856404; display: block; text-align: center; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>📚 学校知识库批量上传</h2>
            <p style="color: #666; font-size: 14px;">支持同时选择多个 PDF 和 Word (.docx) 文件</p>
            
            <div class="upload-area" id="dropArea" onclick="document.getElementById('fileInput').click()">
                <p>📁 点击选择文件 或 将多个文件拖拽到此处</p>
                <input type="file" id="fileInput" accept=".pdf,.docx" multiple>
                <div class="file-name" id="fileName">未选择任何文件</div>
            </div>
            
            <button id="uploadBtn" onclick="uploadFile()" disabled>批量上传并处理</button>
            
            <div class="status" id="statusMsg"></div>
        </div>

        <script>
            const dropArea = document.getElementById('dropArea');
            const fileInput = document.getElementById('fileInput');
            const fileName = document.getElementById('fileName');
            const uploadBtn = document.getElementById('uploadBtn');
            const statusMsg = document.getElementById('statusMsg');
            
            let currentFiles = []; 

            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, preventDefaults, false);
            });
            function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

            ['dragenter', 'dragover'].forEach(eventName => {
                dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
            });
            ['dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
            });

            dropArea.addEventListener('drop', (e) => handleFiles(e.dataTransfer.files));
            fileInput.addEventListener('change', () => handleFiles(fileInput.files));

            function handleFiles(fileList) {
                currentFiles = []; 
                let validFiles = [];
                
                for (let i = 0; i < fileList.length; i++) {
                    let file = fileList[i];
                    let ext = file.name.split('.').pop().toLowerCase();
                    if (ext === 'pdf' || ext === 'docx') {
                        validFiles.push(file);
                    }
                }
                
                if (validFiles.length > 0) {
                    currentFiles = validFiles;
                    let names = currentFiles.map(f => f.name).join(', ');
                    fileName.textContent = `已选择 ${currentFiles.length} 个文件: ${names.length > 80 ? names.substring(0, 80) + '...' : names}`;
                    uploadBtn.disabled = false;
                    statusMsg.className = 'status'; 
                } else {
                    currentFiles = [];
                    fileName.textContent = '❌ 未检测到支持的格式 (仅支持 PDF/Word)';
                    uploadBtn.disabled = true;
                }
            }

            async function uploadFile() {
                if (currentFiles.length === 0) return;

                const formData = new FormData();
                currentFiles.forEach(file => {
                    formData.append('files', file); 
                });

                uploadBtn.disabled = true;
                statusMsg.className = 'status loading';
                statusMsg.textContent = '⏳ 正在批量上传、解析并向量化文档，文件较多时请耐心等待...';

                try {
                    const response = await fetch('/app_upload/api/upload', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();

                    if (response.ok) {
                        statusMsg.className = 'status success';
                        
                        let detailsHtml = result.details.map(d => {
                            if(d.status === 'success') return `✅ ${d.file} <span style="color:#666">(${d.chunks}块)</span>`;
                            return `❌ ${d.file} <span style="color:red">(${d.message})</span>`;
                        }).join('<br>');
                        
                        statusMsg.innerHTML = `
                            <b>${result.message}</b><br><br>
                            ${detailsHtml}<br><br>
                            📊 本次共新增 <b>${result.total_chunks_added}</b> 个文本块。<br>
                            🗄️ 知识库当前总文本块数: <b>${result.total_docs_in_db}</b>
                        `;
                        
                        currentFiles = [];
                        fileInput.value = ''; 
                        fileName.textContent = '未选择任何文件';
                    } else {
                        throw new Error(result.detail || '上传失败');
                    }
                } catch (error) {
                    statusMsg.className = 'status error';
                    statusMsg.textContent = `❌ 错误: ${error.message}`;
                } finally {
                    uploadBtn.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)