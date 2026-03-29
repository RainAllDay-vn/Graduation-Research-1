import os
import sqlite3
import json
import logging
from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

DB_PATH = os.path.join(os.getcwd(), "cache", "ai_cache.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/api/cache', methods=['GET'])
def get_cache():
    try:
        query = request.args.get('q', '')
        model = request.args.get('model', '')
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            sql = "SELECT * FROM ai_cache WHERE 1=1"
            params = []
            
            if query:
                sql += " AND (user_prompt LIKE ? OR response LIKE ?)"
                params.extend([f"%{query}%", f"%{query}%"])
            
            if model:
                sql += " AND model_name LIKE ?"
                params.append(f"%{model}%")
            
            dataset = request.args.get('dataset', '')
            if dataset:
                sql += " AND dataset_name LIKE ?"
                params.append(f"%{dataset}%")
                
            sql += " ORDER BY model_name ASC"
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                data = dict(row)
                # Try to parse response as JSON if it is structured
                try:
                    loaded_resp = json.loads(data['response'])
                    if isinstance(loaded_resp, dict):
                        data['parsed_response'] = loaded_resp
                    else:
                        data['parsed_response'] = {"content": loaded_resp}
                except (json.JSONDecodeError, TypeError):
                    data['parsed_response'] = {"content": data['response']}
                
                # Split thinking if present
                content = data['parsed_response'].get('content', '')
                if "</think>" in content:
                    parts = content.split("</think>", 1)
                    data['thinking'] = parts[0].replace("<think>", "").strip()
                    data['answer'] = parts[1].strip()
                else:
                    data['thinking'] = ""
                    data['answer'] = content
                
                results.append(data)
                
            return jsonify(results)
    except Exception as e:
        logger.error(f"Error fetching cache: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/cache', methods=['PUT'])
def update_cache():
    try:
        data = request.json
        model_name = data.get('model_name')
        system_prompt = data.get('system_prompt')
        user_prompt = data.get('user_prompt')
        new_content = data.get('content')
        
        if not all([model_name, system_prompt, user_prompt, new_content]):
            return jsonify({"error": "Missing required fields"}), 400
            
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if it was JSON before
            cursor.execute("SELECT response FROM ai_cache WHERE model_name=? AND system_prompt=? AND user_prompt=?", 
                           (model_name, system_prompt, user_prompt))
            row = cursor.fetchone()
            if not row:
                return jsonify({"error": "Entry not found"}), 404
                
            original_response = row['response']
            try:
                original_json = json.loads(original_response)
                if isinstance(original_json, dict) and 'content' in original_json:
                    original_json['content'] = new_content
                    final_response = json.dumps(original_json, ensure_ascii=False)
                else:
                    final_response = new_content
            except:
                final_response = new_content
                
            cursor.execute('''
                UPDATE ai_cache SET response = ? 
                WHERE model_name = ? AND system_prompt = ? AND user_prompt = ?
            ''', (final_response, model_name, system_prompt, user_prompt))
            conn.commit()
            
            return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Error updating cache: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/cache', methods=['DELETE'])
def delete_cache():
    try:
        data = request.json
        model_name = data.get('model_name')
        system_prompt = data.get('system_prompt')
        user_prompt = data.get('user_prompt')
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM ai_cache 
                WHERE model_name = ? AND dataset_name = ? AND system_prompt = ? AND user_prompt = ?
            ''', (data.get('model_name'), data.get('dataset_name'), data.get('system_prompt'), data.get('user_prompt')))
            conn.commit()
            return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Error deleting cache: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT model_name FROM ai_cache")
            rows = cursor.fetchall()
            return jsonify([row['model_name'] for row in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT dataset_name FROM ai_cache")
            rows = cursor.fetchall()
            return jsonify([row['dataset_name'] for row in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/sql', methods=['POST'])
def run_sql():
    try:
        sql = request.json.get('sql', '')
        if not sql:
            return jsonify({"error": "No SQL provided"}), 400
            
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            
            # If it's a mutation, commit
            if any(kw in sql.upper() for kw in ["UPDATE", "DELETE", "INSERT", "CREATE", "DROP", "REPLACE"]):
                conn.commit()
                return jsonify({"status": "success", "message": f"Query executed successfully. Rows affected: {cursor.rowcount}"})
            
            # If it's a SELECT, return rows
            rows = cursor.fetchall()
            return jsonify({
                "status": "success", 
                "columns": rows[0].keys() if rows else [], 
                "rows": [dict(r) for r in rows]
            })
    except Exception as e:
        logger.error(f"SQL Error: {e}")
        return jsonify({"error": str(e)}), 500

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Cache Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background: #0f172a; color: #f1f5f9; }
        .glass { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); }
        .neon-border { border: 1px solid rgba(56, 189, 248, 0.3); box-shadow: 0 0 15px rgba(56, 189, 248, 0.1); }
        .card-glow:hover { box-shadow: 0 0 25px rgba(139, 92, 246, 0.2); transform: translateY(-2px); transition: all 0.3s ease; }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #1e293b; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #475569; }
    </style>
</head>
<body class="p-6">
    <div id="app" v-cloak>
        <!-- Header -->
        <header class="mb-8 flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
                <h1 class="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
                    AI Cache Manager
                </h1>
                <p class="text-slate-400 text-sm">Managing experimental model responses and thinking traces.</p>
            </div>
            
            <div class="flex items-center gap-3">
                <div class="relative">
                    <i class="fas fa-search absolute left-3 top-1/2 -translate-y-1/2 text-slate-500"></i>
                    <input v-model="searchQuery" @input="debouncedFetch" 
                        class="bg-slate-800 border border-slate-700 rounded-lg pl-10 pr-4 py-2 w-64 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all" 
                        placeholder="Search cache...">
                </div>
                <select v-model="selectedModel" @change="fetchCache" 
                    class="bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500">
                    <option value="">All Models</option>
                    <option v-for="m in models" :key="m" :value="m">{{m}}</option>
                </select>
                <select v-model="selectedDataset" @change="fetchCache" 
                    class="bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500">
                    <option value="">All Datasets</option>
                    <option v-for="d in datasets" :key="d" :value="d">{{d}}</option>
                </select>
                <div class="flex bg-slate-800 rounded-lg p-1 border border-slate-700">
                    <button @click="viewMode = 'grid'" :class="viewMode === 'grid' ? 'bg-blue-600' : 'hover:bg-slate-700'" class="px-3 py-1 rounded-md transition-all text-xs">
                        <i class="fas fa-th-large"></i>
                    </button>
                    <button @click="viewMode = 'table'" :class="viewMode === 'table' ? 'bg-blue-600' : 'hover:bg-slate-700'" class="px-3 py-1 rounded-md transition-all text-xs">
                        <i class="fas fa-list"></i>
                    </button>
                </div>
                <button @click="showSql = !showSql" class="bg-slate-700 hover:bg-slate-600 px-4 py-2 rounded-lg transition-colors flex items-center gap-2">
                    <i class="fas fa-terminal text-xs"></i> SQL Console
                </button>
                <button @click="fetchCache" class="bg-blue-600 hover:bg-blue-500 p-2 rounded-lg transition-colors">
                    <i class="fas fa-sync-alt" :class="{'fa-spin': loading}"></i>
                </button>
            </div>
        </header>

        <!-- SQL Console Section -->
        <div v-if="showSql" class="glass rounded-2xl p-6 mb-8 neon-border animate-in fade-in slide-in-from-top-4 duration-300">
            <div class="flex justify-between items-center mb-4">
                <h3 class="text-sm font-bold uppercase tracking-wider text-blue-400">Database SQL Console</h3>
                <button @click="showSql = false" class="text-slate-500 hover:text-white"><i class="fas fa-times"></i></button>
            </div>
            <div class="flex flex-col gap-4">
                <div class="relative">
                    <textarea v-model="sqlQuery" 
                        class="w-full bg-slate-900 border border-slate-700 rounded-xl p-4 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 h-24"
                        placeholder="UPDATE ai_cache SET model_name = 'new_name' WHERE ..."></textarea>
                    <button @click="runSql" 
                        class="absolute bottom-3 right-3 bg-blue-600 hover:bg-blue-500 px-4 py-1.5 rounded-lg text-sm font-bold transition-all disabled:opacity-50"
                        :disabled="!sqlQuery || sqlLoading">
                        <i class="fas fa-play mr-2" v-if="!sqlLoading"></i>
                        <i class="fas fa-spinner fa-spin mr-2" v-else></i>
                        Execute Query
                    </button>
                </div>
                
                <!-- SQL Result -->
                <div v-if="sqlResult" class="bg-black/40 rounded-xl p-4 border border-slate-800 overflow-x-auto">
                    <div v-if="sqlResult.error" class="text-red-400 text-sm font-mono flex items-center gap-2">
                        <i class="fas fa-exclamation-triangle"></i> {{sqlResult.error}}
                    </div>
                    <div v-else-if="sqlResult.columns" class="overflow-hidden">
                        <table class="w-full text-xs text-left">
                            <thead>
                                <tr class="text-slate-500 border-b border-slate-800">
                                    <th v-for="col in sqlResult.columns" :key="col" class="py-2 px-3">{{col}}</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr v-for="(row, i) in sqlResult.rows" :key="i" class="border-b border-slate-800/50 hover:bg-slate-800/20">
                                    <td v-for="col in sqlResult.columns" :key="col" class="py-2 px-3 text-slate-300 font-mono">{{row[col]}}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    <div v-else class="text-emerald-400 text-sm font-bold">
                         <i class="fas fa-check-circle"></i> {{sqlResult.message}}
                    </div>
                </div>
            </div>
        </div>

        <!-- Stats -->
        <div class="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
            <div class="glass p-4 rounded-xl flex items-center gap-4">
                <div class="bg-blue-500/20 p-3 rounded-lg text-blue-400"><i class="fas fa-database text-xl"></i></div>
                <div><p class="text-xs text-slate-400 uppercase">Total Entries</p><h3 class="text-2xl font-bold">{{cache.length}}</h3></div>
            </div>
            <div class="glass p-4 rounded-xl flex items-center gap-4">
                <div class="bg-purple-500/20 p-3 rounded-lg text-purple-400"><i class="fas fa-microchip text-xl"></i></div>
                <div><p class="text-xs text-slate-400 uppercase">Models</p><h3 class="text-2xl font-bold">{{models.length}}</h3></div>
            </div>
            <div class="glass p-4 rounded-xl flex items-center gap-4">
                <div class="bg-emerald-500/20 p-3 rounded-lg text-emerald-400"><i class="fas fa-bolt text-xl"></i></div>
                <div><p class="text-xs text-slate-400 uppercase">Status</p><h3 class="text-2xl font-bold">Online</h3></div>
            </div>
        </div>

        <!-- Main Content (Grid View) -->
        <div v-if="viewMode === 'grid'" class="grid grid-cols-1 lg:grid-cols-2 2xl:grid-cols-3 gap-6">
            <div v-for="item in cache" :key="item.user_prompt + item.model_name" 
                class="glass rounded-2xl p-6 relative card-glow flex flex-col h-[400px]">
                
                <div class="flex justify-between items-start mb-4">
                    <div class="flex gap-2">
                        <span class="text-[10px] uppercase tracking-widest bg-blue-500/20 text-blue-400 px-2 py-1 rounded font-bold">
                            {{item.model_name}}
                        </span>
                        <span class="text-[10px] uppercase tracking-widest bg-purple-500/20 text-purple-400 px-2 py-1 rounded font-bold">
                            {{item.dataset_name}}
                        </span>
                    </div>
                    <div class="flex gap-2">
                        <button @click="editItem(item)" class="text-slate-500 hover:text-blue-400 transition-colors"><i class="fas fa-edit"></i></button>
                        <button @click="deleteItem(item)" class="text-slate-500 hover:text-red-400 transition-colors"><i class="fas fa-trash"></i></button>
                    </div>
                </div>

                <div class="mb-4">
                    <h4 class="text-xs font-semibold text-slate-500 uppercase mb-1">User Prompt</h4>
                    <p class="text-sm line-clamp-2 italic text-slate-300">"{{item.user_prompt}}"</p>
                </div>

                <div class="flex-grow overflow-hidden flex flex-col gap-3">
                    <div v-if="item.thinking" class="bg-slate-900/50 rounded-lg p-3 border border-slate-700/50 flex flex-col h-1/2">
                        <h4 class="text-[10px] font-bold text-slate-500 uppercase mb-1 flex items-center gap-2">
                            <i class="fas fa-brain text-purple-400"></i> Thinking Trace
                        </h4>
                        <div class="text-[13px] text-slate-400 overflow-y-auto pr-2 custom-scrollbar">
                            {{item.thinking}}
                        </div>
                    </div>
                    
                    <div class="bg-blue-900/20 border border-blue-500/20 rounded-lg p-3 flex flex-col" :class="item.thinking ? 'h-1/2' : 'h-full'">
                        <h4 class="text-[10px] font-bold text-blue-400 uppercase mb-1 flex items-center gap-2">
                            <i class="fas fa-terminal"></i> Response Content
                        </h4>
                        <pre class="text-[13px] text-slate-200 overflow-y-auto whitespace-pre-wrap pr-2 font-mono"><code>{{item.answer}}</code></pre>
                    </div>
                </div>
            </div>
            
            <!-- Empty State -->
            <div v-if="cache.length === 0 && !loading" class="col-span-full py-20 text-center">
                <i class="fas fa-ghost text-6xl text-slate-700 mb-4"></i>
                <p class="text-slate-500">No cache entries found mapping your search.</p>
            </div>
        </div>

        <!-- Main Content (Table View) -->
        <div v-if="viewMode === 'table'" class="glass rounded-2xl overflow-hidden min-h-[400px]">
            <table class="w-full text-sm text-left border-collapse">
                <thead>
                    <tr class="bg-slate-800/50 text-slate-400 border-b border-slate-700">
                        <th class="py-4 px-6 font-semibold uppercase text-xs">Model</th>
                        <th class="py-4 px-6 font-semibold uppercase text-xs">Dataset</th>
                        <th class="py-4 px-6 font-semibold uppercase text-xs">System Prompt</th>
                        <th class="py-4 px-6 font-semibold uppercase text-xs">User Prompt</th>
                        <th class="py-4 px-6 font-semibold uppercase text-xs">Final Answer</th>
                        <th class="py-4 px-6 font-semibold uppercase text-xs w-24">Actions</th>
                    </tr>
                </thead>
                <tbody>
                    <tr v-for="item in cache" :key="item.user_prompt + item.model_name" class="border-b border-slate-800 hover:bg-slate-800/30 transition-colors">
                        <td class="py-4 px-6">
                            <span class="bg-blue-500/10 text-blue-400 text-[10px] px-2 py-1 rounded font-bold inline-block">{{item.model_name}}</span>
                        </td>
                        <td class="py-4 px-6">
                            <span class="bg-purple-500/10 text-purple-400 text-[10px] px-2 py-1 rounded font-bold inline-block">{{item.dataset_name}}</span>
                        </td>
                        <td class="py-4 px-6 text-slate-500 italic truncate max-w-[150px]" :title="item.system_prompt">{{item.system_prompt}}</td>
                        <td class="py-4 px-6 text-slate-400 italic truncate max-w-[200px]" :title="item.user_prompt">{{item.user_prompt}}</td>
                        <td class="py-4 px-6 font-mono text-emerald-400 truncate max-w-[300px]" :title="item.answer">{{item.answer}}</td>
                        <td class="py-4 px-6">
                            <div class="flex gap-4">
                                <button @click="editItem(item)" class="text-slate-500 hover:text-blue-400"><i class="fas fa-edit"></i></button>
                                <button @click="deleteItem(item)" class="text-slate-500 hover:text-red-400"><i class="fas fa-trash"></i></button>
                            </div>
                        </td>
                    </tr>
                </tbody>
            </table>
            <!-- Empty State -->
            <div v-if="cache.length === 0 && !loading" class="py-20 text-center">
                <i class="fas fa-ghost text-6xl text-slate-700 mb-4"></i>
                <p class="text-slate-500">No cache entries found mapping your search.</p>
            </div>
        </div>

        <!-- Edit Modal -->
        <div v-if="editingItem" class="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
            <div class="glass w-full max-w-4xl max-h-[90vh] rounded-3xl overflow-hidden flex flex-col neon-border">
                <div class="p-6 border-b border-slate-700 flex justify-between items-center">
                    <h2 class="text-xl font-bold">Edit Cache Entry</h2>
                    <button @click="editingItem = null" class="text-slate-400 hover:text-white"><i class="fas fa-times"></i></button>
                </div>
                <div class="p-6 overflow-y-auto flex-grow flex flex-col gap-6">
                    <div>
                        <label class="text-xs font-bold text-slate-500 uppercase mb-2 block">Context</label>
                        <div class="grid grid-cols-2 gap-4">
                            <div class="bg-slate-800 p-3 rounded-lg">
                                <span class="text-[10px] text-slate-500 uppercase block">Model</span>
                                <span class="text-sm">{{editingItem.model_name}}</span>
                            </div>
                            <div class="bg-slate-800 p-3 rounded-lg overflow-hidden">
                                <span class="text-[10px] text-slate-500 uppercase block">User Prompt</span>
                                <span class="text-sm truncate block">{{editingItem.user_prompt}}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="flex-grow flex flex-col">
                        <label class="text-xs font-bold text-slate-500 uppercase mb-2 block">Full Response Content</label>
                        <textarea v-model="editBuffer" 
                            class="flex-grow w-full bg-slate-900 border border-slate-700 rounded-xl p-4 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 h-96"
                            spellcheck="false"></textarea>
                    </div>
                </div>
                <div class="p-6 border-t border-slate-700 flex justify-end gap-3">
                    <button @click="editingItem = null" class="px-6 py-2 rounded-lg hover:bg-slate-800 transition-colors">Cancel</button>
                    <button @click="saveEdit" class="bg-blue-600 hover:bg-blue-500 px-6 py-2 rounded-lg font-bold transition-all shadow-lg shadow-blue-500/20">
                        Save Changes
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script>
        const { createApp, ref, onMounted } = Vue;

        createApp({
            setup() {
                const cache = ref([]);
                const models = ref([]);
                const datasets = ref([]);
                const searchQuery = ref('');
                const selectedModel = ref('');
                const selectedDataset = ref('');
                const loading = ref(false);
                const editingItem = ref(null);
                const editBuffer = ref('');
                const showSql = ref(false);
                const sqlQuery = ref('');
                const sqlResult = ref(null);
                const sqlLoading = ref(false);
                const viewMode = ref('grid');

                const fetchCache = async () => {
                    loading.value = true;
                    try {
                        const url = new URL('/api/cache', window.location.origin);
                        if (searchQuery.value) url.searchParams.append('q', searchQuery.value);
                        if (selectedModel.value) url.searchParams.append('model', selectedModel.value);
                        if (selectedDataset.value) url.searchParams.append('dataset', selectedDataset.value);
                        
                        const res = await fetch(url);
                        cache.value = await res.json();
                    } catch (e) {
                        console.error(e);
                    } finally {
                        loading.value = false;
                    }
                };

                const fetchModels = async () => {
                    try {
                        const [mRes, dRes] = await Promise.all([
                            fetch('/api/models'),
                            fetch('/api/datasets')
                        ]);
                        models.value = await mRes.json();
                        datasets.value = await dRes.json();
                    } catch (e) { console.error(e); }
                };

                let debounceTimer;
                const debouncedFetch = () => {
                    clearTimeout(debounceTimer);
                    debounceTimer = setTimeout(fetchCache, 300);
                };

                const editItem = (item) => {
                    editingItem.value = JSON.parse(JSON.stringify(item));
                    editBuffer.value = item.parsed_response.content || item.response;
                };

                const saveEdit = async () => {
                    try {
                        const res = await fetch('/api/cache', {
                            method: 'PUT',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                model_name: editingItem.value.model_name,
                                system_prompt: editingItem.value.system_prompt,
                                user_prompt: editingItem.value.user_prompt,
                                content: editBuffer.value
                            })
                        });
                        if (res.ok) {
                            editingItem.value = null;
                            fetchCache();
                        }
                    } catch (e) { console.error(e); }
                };

                const deleteItem = async (item) => {
                    if (!confirm('Are you sure you want to delete this cache entry?')) return;
                    try {
                        const res = await fetch('/api/cache', {
                            method: 'DELETE',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(item)
                        });
                        if (res.ok) fetchCache();
                    } catch (e) { console.error(e); }
                };

                const runSql = async () => {
                    sqlLoading.value = true;
                    try {
                        const res = await fetch('/api/sql', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ sql: sqlQuery.value })
                        });
                        sqlResult.value = await res.json();
                        if (!sqlResult.value.error) {
                            fetchCache();
                            fetchModels();
                        }
                    } catch (e) { 
                        sqlResult.value = { error: e.toString() };
                    } finally {
                        sqlLoading.value = false;
                    }
                };

                onMounted(() => {
                    fetchCache();
                    fetchModels();
                });

                return {
                    cache, models, datasets, searchQuery, selectedModel, selectedDataset, loading,
                    editingItem, editBuffer, showSql, sqlQuery, sqlResult, sqlLoading, viewMode,
                    fetchCache, debouncedFetch, editItem, saveEdit, deleteItem, runSql
                };
            }
        }).mount('#app');
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    if not os.path.exists(DB_PATH):
        logger.warning(f"Database not found at {DB_PATH}. Please check the path.")
    
    # Run development server
    app.run(host='0.0.0.0', port=5000, debug=True)
