'''
    创建时间：2023-07-21
    功能：管理文档的增删改查
'''

import os
import chromadb
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ⚠️ 确保这里的路径与你的 main.py 完全一致！建议使用绝对路径
PERSIST_DIRECTORY = "./chroma_db_school"
COLLECTION_NAME = "school_knowledge"

# 创建路由器，并指定统一前缀
router = APIRouter(prefix="/manage_doc", tags=["文档管理"])

class DocumentUpdate(BaseModel):
    content: str
    metadata: Optional[dict] = None

# ==================== API 接口 ====================

@router.get("/api/documents")
async def get_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    source: Optional[str] = None
):
    try:
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        collection = client.get_collection(COLLECTION_NAME)
        all_docs = collection.get(include=['documents', 'metadatas'])

        if source:
            filtered = [(all_docs['ids'][i], all_docs['documents'][i], all_docs['metadatas'][i])
                        for i, meta in enumerate(all_docs['metadatas']) if meta.get('source') == source]
        else:
            filtered = list(zip(all_docs['ids'], all_docs['documents'], all_docs['metadatas']))

        total = len(filtered)
        start = (page - 1) * page_size
        end = start + page_size
        documents = [{"id": f[0], "content": f[1], "metadata": f[2]} for f in filtered[start:end]]

        return {"documents": documents, "total": total, "page": page,
                "page_size": page_size, "total_pages": (total + page_size - 1) // page_size}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    try:
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        collection = client.get_collection(COLLECTION_NAME)
        doc = collection.get(ids=[doc_id], include=['documents', 'metadatas'])
        if not doc['ids']:
            raise HTTPException(status_code=404, detail="文档不存在")
        return {"id": doc['ids'][0], "content": doc['documents'][0], "metadata": doc['metadatas'][0]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/api/documents/{doc_id}")
async def update_document(doc_id: str, update: DocumentUpdate):
    try:
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        collection = client.get_collection(COLLECTION_NAME)
        old_doc = collection.get(ids=[doc_id], include=['metadatas'])
        if not old_doc['ids']:
            raise HTTPException(status_code=404, detail="文档不存在")

        collection.delete(ids=[doc_id])
        metadata = update.metadata if update.metadata else old_doc['metadatas'][0]
        collection.add(ids=[doc_id], documents=[update.content], metadatas=[metadata])
        return {"message": "更新成功", "id": doc_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    try:
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        collection = client.get_collection(COLLECTION_NAME)
        doc = collection.get(ids=[doc_id])
        if not doc['ids']:
            raise HTTPException(status_code=404, detail="文档不存在")
        collection.delete(ids=[doc_id])
        return {"message": "删除成功", "id": doc_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/sources")
async def get_sources():
    try:
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        collection = client.get_collection(COLLECTION_NAME)
        all_docs = collection.get(include=['metadatas'])
        sources = set(meta.get('source') for meta in all_docs['metadatas'] if 'source' in meta)
        return {"sources": sorted(list(sources))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== 前端页面 ====================
@router.get("/", response_class=HTMLResponse)
async def get_management_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>向量文档管理</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', sans-serif; background: #f5f7fa; padding: 20px; }
            .container { max-width: 1400px; margin: 0 auto; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }
            .header h1 { font-size: 28px; margin-bottom: 10px; }
            .stats { display: flex; gap: 20px; margin-top: 15px; }
            .stat-item { background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 5px; }
            .controls { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; display: flex; gap: 15px; align-items: center; }
            .controls select { padding: 8px 12px; border: 1px solid #ddd; border-radius: 5px; min-width: 200px; }
            /* 修复1：表格固定布局，强制列宽生效，不会被内容挤压变窄 */
            table {
                width: 100%;
                background: white;
                border-radius: 10px;
                overflow: hidden;
                table-layout: fixed;
            }
            th { background: #f8f9fa; padding: 15px; text-align: left; border-bottom: 2px solid #dee2e6; }
            td { padding: 15px; border-bottom: 1px solid #dee2e6; vertical-align: top; }

            /* 修复2：外层div做多行省略，不直接作用td，解决宽度压缩、末尾漏字 */
            .content-wrap {
                width: 100%;
                line-height: 1.5;
                max-height: 7.5em; /* 1.5*5=5行高度，精准截断 */
                display: -webkit-box;
                -webkit-line-clamp: 5;
                -webkit-box-orient: vertical;
                overflow: hidden;
                text-overflow: ellipsis;
                word-break: break-all; /* 长文本强制换行，避免溢出漏字 */
            }

            .actions-cell { white-space: nowrap; width: 120px; }
            .actions-wrapper { display: flex; gap: 8px; align-items: center; }

            .btn { padding: 6px 12px; border: none; border-radius: 5px; cursor: pointer; font-size: 14px; }
            .btn-edit { background: #ffc107; color: #000; }
            .btn-edit:hover { background: #e0a800; }
            .btn-delete { background: #dc3545; color: white; }
            .btn-delete:hover { background: #c82333; }

            .pagination { display: flex; justify-content: center; gap: 10px; margin-top: 20px; }
            .pagination button { padding: 8px 16px; border: 1px solid #ddd; background: white; border-radius: 5px; cursor: pointer; }
            .pagination button:hover:not(:disabled) { background: #667eea; color: white; }
            .pagination button:disabled { opacity: 0.5; cursor: not-allowed; }
            .pagination .current { background: #667eea; color: white; }

            .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); }
            .modal-content { background: white; margin: 5% auto; padding: 30px; border-radius: 10px; width: 80%; max-width: 800px; }
            .modal-header { display: flex; justify-content: space-between; margin-bottom: 20px; }
            .close { font-size: 28px; cursor: pointer; }
            textarea { width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 5px; min-height: 200px; font-family: inherit; }
            input[readonly] { background: #f8f9fa; padding: 10px; border: 1px solid #ddd; border-radius: 5px; width: 100%; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📚 向量文档管理系统</h1>
                <p>向量数据库统计信息：</p>
                <div class="stats">
                    <div class="stat-item">总文档数：<span id="totalDocs">0</span></div>
                    <div class="stat-item">当前页：<span id="currentPage">1</span> / <span id="totalPages">1</span></div>
                </div>
            </div>
            <div class="controls">
                <label>来源筛选：</label>
                <select id="sourceFilter" onchange="loadDocuments()"><option value="">全部</option></select>
                <label>每页：</label>
                <select id="pageSize" onchange="loadDocuments()">
                    <option value="10">10</option><option value="20" selected>20</option>
                    <option value="50">50</option><option value="100">100</option>
                </select>
            </div>
            <table>
                <thead>
                    <tr>
                        <th style="width: 60px;">序号</th>
                        <!-- 修复3：固定内容列宽度，恢复原来500px展示空间 -->
                        <th style="width: 500px;">内容</th>
                        <th style="width: 200px;">元数据</th>
                        <th style="width: 120px;">操作</th>
                    </tr>
                </thead>
                <tbody id="docTable"><tr><td colspan="4" style="text-align:center;padding:40px;">加载中...</td></tr></tbody>
            </table>
            <div class="pagination" id="pagination"></div>
        </div>

        <!-- 编辑模态框 -->
        <div id="editModal" class="modal">
            <div class="modal-content">
                <div class="modal-header"><h2>编辑文档</h2><span class="close" onclick="closeModal()">&times;</span></div>
                <div style="margin-bottom:20px;"><label>文档ID：</label><input type="text" id="editId" readonly></div>
                <div style="margin-bottom:20px;"><label>内容：</label><textarea id="editContent"></textarea></div>
                <div style="display:flex;justify-content:flex-end;gap:10px;">
                    <button class="btn" style="background:#6c757d;color:white;padding:10px 20px;" onclick="closeModal()">取消</button>
                    <button class="btn" style="background:#28a745;color:white;padding:10px 20px;" onclick="saveDoc()">保存</button>
                </div>
            </div>
        </div>

        <script>
            let currentPage = 1, currentDocId = null;

            async function loadSources() {
                const res = await fetch('/manage_doc/api/sources');
                const data = await res.json();
                const sel = document.getElementById('sourceFilter');
                data.sources.forEach(s => {
                    const opt = document.createElement('option');
                    opt.value = s; opt.textContent = s;
                    sel.appendChild(opt);
                });
            }

            async function loadDocuments() {
                const pageSize = document.getElementById('pageSize').value;
                const source = document.getElementById('sourceFilter').value;
                const url = `/manage_doc/api/documents?page=${currentPage}&page_size=${pageSize}${source ? `&source=${encodeURIComponent(source)}` : ''}`;
                const res = await fetch(url);
                const data = await res.json();
                document.getElementById('totalDocs').textContent = data.total;
                document.getElementById('currentPage').textContent = data.page;
                document.getElementById('totalPages').textContent = data.total_pages;
                renderTable(data.documents);
                renderPagination(data.page, data.total_pages);
            }

            function renderTable(docs) {
                const tbody = document.getElementById('docTable');
                if (!docs.length) {
                    tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;padding:40px;">暂无文档</td></tr>';
                    return;
                }
                const pageSize = parseInt(document.getElementById('pageSize').value);
                const start = (currentPage - 1) * pageSize;

                tbody.innerHTML = docs.map((d, i) => `
                    <tr>
                        <td>${start + i + 1}</td>
                        <td title="${escapeHtml(d.content)}">
                            <div class="content-wrap">${escapeHtml(d.content)}</div>
                        </td>
                        <td style="font-size:12px;color:#6c757d;max-width:200px;overflow:hidden;text-overflow:ellipsis;">${escapeHtml(JSON.stringify(d.metadata))}</td>
                        <td class="actions-cell">
                            <div class="actions-wrapper">
                                <button class="btn btn-edit" onclick="openModal('${d.id}')">编辑</button>
                                <button class="btn btn-delete" onclick="deleteDoc('${d.id}')">删除</button>
                            </div>
                        </td>
                    </tr>
                `).join('');
            }

            // 修改分页：首页默认展示1、2、3、4、5
            function renderPagination(current, total) {
                const pag = document.getElementById('pagination');
                let html = `<button onclick="changePage(1)" ${current===1?'disabled':''}>首页</button>`;
                html += `<button onclick="changePage(${current-1})" ${current===1?'disabled':''}>上一页</button>`;
                
                // 固定起始页码，默认从1开始展示5个页码：1,2,3,4,5
                let startNum = 1;
                // 如果当前页大于3，则动态居中；只有第一页时强制1-5
                if (current > 3) {
                    startNum = current - 2;
                }
                let endNum = startNum + 4;
                // 不超过总页数
                if (endNum > total) {
                    endNum = total;
                    startNum = Math.max(1, endNum - 4);
                }

                for (let i = startNum; i <= endNum; i++) {
                    html += `<button onclick="changePage(${i})" class="${i===current?'current':''}">${i}</button>`;
                }

                html += `<button onclick="changePage(${current+1})" ${current===total?'disabled':''}>下一页</button>`;
                html += `<button onclick="changePage(${total})" ${current===total?'disabled':''}>末页</button>`;
                pag.innerHTML = html;
            }

            function changePage(p) {
                currentPage = p;
                loadDocuments();
                window.scrollTo({top:0,behavior:'smooth'});
            }

            async function openModal(id) {
                const res = await fetch(`/manage_doc/api/documents/${id}`);
                const doc = await res.json();
                currentDocId = id;
                document.getElementById('editId').value = doc.id;
                document.getElementById('editContent').value = doc.content;
                document.getElementById('editModal').style.display = 'block';
            }

            function closeModal() {
                document.getElementById('editModal').style.display = 'none';
                currentDocId = null;
            }

            async function saveDoc() {
                const content = document.getElementById('editContent').value;
                if (!content.trim()) { alert('内容不能为空'); return; }
                const res = await fetch(`/manage_doc/api/documents/${currentDocId}`, {
                    method: 'PUT',
                    headers: {'Content-Type':'application/json'},
                    body: JSON.stringify({content})
                });
                if (res.ok) {
                    alert('更新成功');
                    closeModal();
                    loadDocuments();
                } else {
                    const err = await res.json();
                    alert('失败: ' + err.detail);
                }
            }

            async function deleteDoc(id) {
                if (!confirm('确定删除？不可恢复！')) return;
                const res = await fetch(`/manage_doc/api/documents/${id}`, {method:'DELETE'});
                if (res.ok) {
                    alert('删除成功');
                    loadDocuments();
                } else {
                    const err = await res.json();
                    alert('失败: ' + err.detail);
                }
            }

            function escapeHtml(t) {
                const d = document.createElement('div');
                d.textContent = t;
                return d.innerHTML;
            }

            window.onclick = e => {
                if (e.target.id === 'editModal') closeModal();
            };

            window.onload = () => {
                loadSources();
                loadDocuments();
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)