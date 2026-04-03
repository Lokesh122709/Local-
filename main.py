"""
Wingo 20‑Server Ultimate System v13.1 (Fully Fixed)
- No Internal Server Error
- Dropdown to switch between 20 independent AI engines
- Auto-refresh every 5 seconds
- Each engine learns from live draws and persists its state
"""

import requests
import json
import time
import math
import threading
import os
from datetime import datetime
from collections import Counter, defaultdict, deque
from flask import Flask, render_template_string, jsonify, request, redirect

# ============================================================
# CONFIGURATION
# ============================================================
API_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
HEADERS = {"User-Agent": "Mozilla/5.0"}
DRAW_INTERVAL = 30
MAX_HISTORY = 200
REQUEST_TIMEOUT = 5

NUMBER_TO_COLOR = {
    1: "GREEN", 3: "GREEN", 5: "GREEN", 7: "GREEN", 9: "GREEN",
    2: "RED", 4: "RED", 6: "RED", 8: "RED", 0: "RED"
}
NUMBER_TO_VIOLET = {0: "VIOLET", 5: "VIOLET"}
NUMBER_TO_SIZE = {0: "SMALL", 1: "SMALL", 2: "SMALL", 3: "SMALL", 4: "SMALL",
                  5: "BIG", 6: "BIG", 7: "BIG", 8: "BIG", 9: "BIG"}

# ============================================================
# PERSISTENCE
# ============================================================
def get_data_file(server_id):
    return f"wingo_server_{server_id}_data.json"

def save_server_data(server_id, data):
    try:
        with open(get_data_file(server_id), 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except:
        return False

def load_server_data(server_id):
    fname = get_data_file(server_id)
    if not os.path.exists(fname):
        return None
    try:
        with open(fname, 'r') as f:
            return json.load(f)
    except:
        return None

# ============================================================
# BASE ENGINE (Fixed Initialization)
# ============================================================
class BaseEngine:
    def __init__(self, server_id, config):
        self.server_id = server_id
        self.config = config
        self.history_numbers = []
        self.history_colors = []
        self.history_sizes = []
        self.seasonal_cache = defaultdict(list)
        self.model_predictions_history = {}
        self.last_save_time = time.time()
        # Will be filled by subclass
        self.models = {}
        self.model_names = []
        self.model_weights = {}
        self.model_accuracies = {}
        self._load_state()

    def _load_state(self):
        saved = load_server_data(self.server_id)
        if saved:
            self._saved_data = saved
        else:
            self._saved_data = None

    def _apply_saved_state(self):
        if self._saved_data:
            for name in self.model_names:
                if name in self._saved_data.get("model_weights", {}):
                    self.model_weights[name] = self._saved_data["model_weights"][name]
                if name in self._saved_data.get("model_accuracies", {}):
                    self.model_accuracies[name] = deque(self._saved_data["model_accuracies"][name], maxlen=30)
            sc = self._saved_data.get("seasonal_cache", {})
            for k, v in sc.items():
                self.seasonal_cache[int(k)] = v
            print(f"🔄 Server {self.server_id} loaded previous state")
        else:
            self.model_weights = {name: 1.0 for name in self.model_names}
            self.model_accuracies = {name: deque(maxlen=30) for name in self.model_names}
            print(f"🆕 Server {self.server_id} initialized fresh")

    def _save_state(self):
        data = {
            "model_weights": self.model_weights,
            "model_accuracies": {k: list(v) for k, v in self.model_accuracies.items()},
            "seasonal_cache": {str(k): v for k, v in self.seasonal_cache.items()}
        }
        save_server_data(self.server_id, data)
        self.last_save_time = time.time()

    def update_history(self, games):
        chrono = list(reversed(games))
        self.history_numbers = [int(g['number']) for g in chrono]
        self.history_colors = [g['color'] for g in chrono]
        self.history_sizes = [g['size'] for g in chrono]
        for g in chrono:
            try:
                period = g.get('period', '')
                if len(period) >= 14:
                    minute = int(period[12:14])
                else:
                    minute = datetime.now().minute
                num = int(g['number'])
                self.seasonal_cache[minute].append(num)
                if len(self.seasonal_cache[minute]) > 50:
                    self.seasonal_cache[minute].pop(0)
            except:
                pass

    def predict(self, games):
        self.update_history(games)
        if len(self.history_numbers) < 10:
            return None
        model_preds = {}
        for name, func in self.models.items():
            pred = func()
            if pred and pred[0] is not None:
                model_preds[name] = {"number": pred[0], "confidence": pred[1]}
        if not model_preds:
            return None
        num_votes = defaultdict(float)
        for name, p in model_preds.items():
            w = self.model_weights.get(name, 1.0)
            num_votes[p["number"]] += p["confidence"] * w
        best_num = max(num_votes, key=num_votes.get)
        total = sum(num_votes.values())
        raw_conf = int((num_votes[best_num]/total)*100) if total>0 else 50
        conf = min(95, raw_conf)
        self.model_predictions_history = model_preds
        return {"number": best_num, "confidence": conf, "model_predictions": model_preds}

    def learn(self, actual_number):
        if not self.model_predictions_history:
            return
        lr = self.config.get("learning_rate", 0.15)
        for name, pred in self.model_predictions_history.items():
            correct = (pred["number"] == actual_number)
            if correct:
                self.model_weights[name] += lr
            else:
                self.model_weights[name] -= lr * 0.6
            self.model_weights[name] = max(0.2, min(2.5, self.model_weights[name]))
            self.model_accuracies[name].append(1 if correct else 0)
        if time.time() - self.last_save_time > 30:
            self._save_state()

    def get_weights_summary(self):
        summ = {}
        for name in self.model_names:
            acc = sum(self.model_accuracies[name])/len(self.model_accuracies[name]) if self.model_accuracies[name] else 0.5
            summ[name] = {"weight": round(self.model_weights[name],2), "accuracy": round(acc*100,1)}
        return summ

# ============================================================
# PREDICTOR FACTORIES
# ============================================================
def make_markov(engine, order, decay):
    def predictor():
        if len(engine.history_numbers) < order+1:
            return None, 0
        trans = defaultdict(lambda: defaultdict(float))
        for i in range(len(engine.history_numbers)-order):
            state = tuple(engine.history_numbers[i:i+order])
            nxt = engine.history_numbers[i+order]
            weight = decay ** (len(engine.history_numbers)-order-i)
            trans[state][nxt] += weight
        cur = tuple(engine.history_numbers[-order:])
        if cur not in trans:
            return None, 0
        total = sum(trans[cur].values())
        if total == 0:
            return None, 0
        most = max(trans[cur].items(), key=lambda x: x[1])
        conf = int((most[1]/total)*100)
        return most[0], min(95, conf)
    return predictor

def make_pattern_miner(engine, max_len=8):
    def predictor():
        if len(engine.history_numbers) < 10:
            return None, 0
        for length in range(max_len, 1, -1):
            suffix = tuple(engine.history_numbers[-length:])
            occ = []
            for i in range(len(engine.history_numbers)-length):
                if tuple(engine.history_numbers[i:i+length]) == suffix:
                    if i+length < len(engine.history_numbers):
                        occ.append(engine.history_numbers[i+length])
            if occ:
                c = Counter(occ)
                most = c.most_common(1)[0]
                conf = min(85, 50 + (most[1]/len(occ))*35)
                return most[0], conf
        return None, 0
    return predictor

def make_seasonal(engine):
    def predictor():
        cur_min = datetime.now().minute
        if cur_min not in engine.seasonal_cache or len(engine.seasonal_cache[cur_min]) < 5:
            return None, 0
        recent = engine.seasonal_cache[cur_min][-30:]
        c = Counter(recent)
        most = c.most_common(1)[0]
        conf = min(80, 50 + (most[1]/len(recent))*30)
        return most[0], conf
    return predictor

def make_fourier(engine, min_period=3, max_period=20):
    def predictor():
        if len(engine.history_numbers) < 30:
            return None, 0
        best_period, best_score = None, 0
        for p in range(min_period, min(max_period, len(engine.history_numbers)//2)):
            matches = 0
            limit = min(len(engine.history_numbers), p+50)
            for i in range(p, limit):
                if engine.history_numbers[i] == engine.history_numbers[i-p]:
                    matches += 1
            score = matches / (limit-p)
            if score > best_score:
                best_score = score
                best_period = p
        if best_period and best_score > 0.4:
            pred = engine.history_numbers[-best_period]
            conf = min(80, 50 + int(best_score*30))
            return pred, conf
        return None, 0
    return predictor

def make_gap_analysis(engine):
    def predictor():
        if len(engine.history_numbers) < 10:
            return None, 0
        last_seen = {}
        for i, n in enumerate(engine.history_numbers):
            last_seen[n] = i
        max_gap = max((len(engine.history_numbers)-last_seen.get(n, len(engine.history_numbers))) for n in range(10))
        scores = []
        for n in range(10):
            gap = len(engine.history_numbers) - last_seen.get(n, len(engine.history_numbers))
            score = gap / max_gap if max_gap>0 else 0
            scores.append((n, score))
        best = max(scores, key=lambda x: x[1])
        conf = min(75, 40 + int(best[1]*35))
        return best[0], conf
    return predictor

def make_neighbor(engine):
    def predictor():
        if len(engine.history_numbers) < 10:
            return None, 0
        neighbors = []
        for n in engine.history_numbers[:5]:
            for d in [-2,-1,1,2]:
                neighbors.append((n+d)%10)
        c = Counter(neighbors)
        most = c.most_common(1)[0]
        conf = min(70, 40 + (most[1]/len(neighbors))*30)
        return most[0], conf
    return predictor

# ============================================================
# CREATE 20 ENGINES WITH DIFFERENT CONFIGURATIONS
# ============================================================
def create_engine(server_id, config):
    class CustomEngine(BaseEngine):
        def __init__(self, sid, cfg):
            super().__init__(sid, cfg)
            # Build models using self (the engine instance)
            self.models = {}
            if cfg.get("use_markov1"):
                self.models["Markov1"] = make_markov(self, 1, cfg.get("decay", 0.96))
            if cfg.get("use_markov2"):
                self.models["Markov2"] = make_markov(self, 2, cfg.get("decay", 0.96))
            if cfg.get("use_markov3"):
                self.models["Markov3"] = make_markov(self, 3, cfg.get("decay", 0.96))
            if cfg.get("use_pattern"):
                self.models["PatternMiner"] = make_pattern_miner(self, cfg.get("pattern_len", 8))
            if cfg.get("use_seasonal"):
                self.models["Seasonal"] = make_seasonal(self)
            if cfg.get("use_fourier"):
                self.models["Fourier"] = make_fourier(self, 3, 20)
            if cfg.get("use_gap"):
                self.models["GapAnalysis"] = make_gap_analysis(self)
            if cfg.get("use_neighbor"):
                self.models["NeighborInfluence"] = make_neighbor(self)
            if not self.models:
                self.models["Fallback"] = lambda: (5, 50)
            self.model_names = list(self.models.keys())
            self._apply_saved_state()
    return CustomEngine(server_id, config)

server_configs = [
    {"id": 1, "name": "Quantum Pulse", "theme": "dark_red", "decay":0.98, "use_markov1":True, "use_pattern":True, "use_gap":True, "learning_rate":0.2},
    {"id": 2, "name": "Markov Master", "theme": "blue_ocean", "decay":0.95, "use_markov1":True, "use_markov2":True, "use_markov3":True, "learning_rate":0.15},
    {"id": 3, "name": "Pattern Hunter", "theme": "green_forest", "use_pattern":True, "pattern_len":10, "use_gap":True, "learning_rate":0.18},
    {"id": 4, "name": "Seasonal Sage", "theme": "gold_sunset", "use_seasonal":True, "use_fourier":True, "learning_rate":0.12},
    {"id": 5, "name": "Gap Analyst", "theme": "purple_nebula", "use_gap":True, "use_neighbor":True, "learning_rate":0.22},
    {"id": 6, "name": "Full Ensemble", "theme": "rainbow", "use_markov1":True, "use_markov2":True, "use_pattern":True, "use_seasonal":True, "use_fourier":True, "use_gap":True, "use_neighbor":True, "learning_rate":0.1},
    {"id": 7, "name": "Quantum Markov", "theme": "cyan_tech", "decay":0.99, "use_markov1":True, "use_markov2":True, "use_gap":True, "learning_rate":0.17},
    {"id": 8, "name": "Lightning", "theme": "yellow_flash", "decay":0.93, "use_markov1":True, "use_pattern":True, "learning_rate":0.25},
    {"id": 9, "name": "Steady Hand", "theme": "gray_minimal", "use_markov2":True, "use_seasonal":True, "learning_rate":0.1},
    {"id":10, "name": "Fourier Vision", "theme": "indigo_deep", "use_fourier":True, "use_pattern":True, "learning_rate":0.14},
    {"id":11, "name": "Neighbor Watch", "theme": "orange_vibrant", "use_neighbor":True, "use_gap":True, "learning_rate":0.2},
    {"id":12, "name": "Combo Striker", "theme": "pink_dream", "use_pattern":True, "use_neighbor":True, "learning_rate":0.16},
    {"id":13, "name": "AI Hybrid", "theme": "dark_blue", "use_markov1":True, "use_markov2":True, "use_markov3":True, "use_pattern":True, "use_gap":True, "use_neighbor":True, "learning_rate":0.13},
    {"id":14, "name": "Eco Mode", "theme": "light_green", "use_markov1":True, "use_gap":True, "learning_rate":0.09},
    {"id":15, "name": "Deep Research", "theme": "midnight", "decay":0.97, "use_markov1":True, "use_markov2":True, "use_pattern":True, "use_seasonal":True, "use_fourier":True, "use_gap":True, "use_neighbor":True, "learning_rate":0.11},
    {"id":16, "name": "Fast & Furious", "theme": "hot_pink", "decay":0.92, "use_markov1":True, "use_neighbor":True, "learning_rate":0.27},
    {"id":17, "name": "Precision", "theme": "silver_metal", "use_markov2":True, "use_pattern":True, "use_fourier":True, "learning_rate":0.12},
    {"id":18, "name": "Oracle", "theme": "mystic", "decay":0.94, "use_markov1":True, "use_markov2":True, "use_markov3":True, "use_pattern":True, "use_seasonal":True, "use_fourier":True, "use_gap":True, "use_neighbor":True, "learning_rate":0.08},
    {"id":19, "name": "Balanced", "theme": "classic", "use_markov1":True, "use_pattern":True, "use_gap":True, "learning_rate":0.15},
    {"id":20, "name": "Ultimate Beast", "theme": "blood_red", "decay":0.99, "use_markov1":True, "use_markov2":True, "use_markov3":True, "use_pattern":True, "use_seasonal":True, "use_fourier":True, "use_gap":True, "use_neighbor":True, "learning_rate":0.2},
]

engines = {}
for cfg in server_configs:
    engines[cfg["id"]] = create_engine(cfg["id"], cfg)

# ============================================================
# SHARED GAME DATA & BACKGROUND UPDATER
# ============================================================
game_history = []
last_seen_period = None

def fetch_games():
    global game_history, last_seen_period
    try:
        ts = int(datetime.now().timestamp() * 1000)
        resp = requests.get(f"{API_URL}?ts={ts}", headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            if "data" in data and "list" in data["data"]:
                new_games = []
                for item in data["data"]["list"]:
                    period = item.get("period","")
                    num_str = item.get("number","")
                    num = int(num_str) if num_str.isdigit() else 0
                    color = NUMBER_TO_COLOR.get(num, "RED")
                    if num in NUMBER_TO_VIOLET:
                        color = NUMBER_TO_VIOLET[num]
                    new_games.append({
                        "period": period,
                        "number": str(num),
                        "color": color,
                        "size": NUMBER_TO_SIZE[num]
                    })
                if new_games and (last_seen_period is None or new_games[0]['period'] != last_seen_period):
                    if last_seen_period is not None:
                        actual_num = int(new_games[0]['number'])
                        print(f"🎯 New draw: {actual_num} - updating all servers")
                        for eng in engines.values():
                            eng.learn(actual_num)
                    last_seen_period = new_games[0]['period']
                game_history = new_games[:MAX_HISTORY]
                print(f"✅ Fetched {len(game_history)} games")
    except Exception as e:
        print(f"❌ Fetch error: {e}")

def background_updater():
    while True:
        fetch_games()
        time.sleep(DRAW_INTERVAL)

threading.Thread(target=background_updater, daemon=True).start()
fetch_games()

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)

def get_theme_css(theme_name):
    themes = {
        "dark_red": "background:#1a0a0a; .card{background:#2a1010; border:2px solid #ff4444;} .color{color:#ff4444;}",
        "blue_ocean": "background:#0a1a2a; .card{background:#102a3f; border:2px solid #44aaff;} .color{color:#44aaff;}",
        "green_forest": "background:#0a1a0a; .card{background:#103f10; border:2px solid #44ff44;} .color{color:#44ff44;}",
        "gold_sunset": "background:#1a1a0a; .card{background:#2a2a10; border:2px solid #ffcc44;} .color{color:#ffcc44;}",
        "purple_nebula": "background:#1a0a2a; .card{background:#2a1040; border:2px solid #cc66ff;} .color{color:#cc66ff;}",
        "rainbow": "background:#0a0a2a; .card{background:linear-gradient(135deg,#ff4444,#44ff44,#44aaff);} .color{color:#fff; text-shadow:0 0 5px black;}",
        "cyan_tech": "background:#0a1a1a; .card{background:#104040; border:2px solid #00ffff;} .color{color:#00ffff;}",
        "yellow_flash": "background:#1a1a00; .card{background:#2a2a00; border:2px solid #ffff44;} .color{color:#ffff44;}",
        "gray_minimal": "background:#222; .card{background:#333; border:1px solid #888;} .color{color:#ddd;}",
        "indigo_deep": "background:#0a0a2a; .card{background:#1a1a4a; border:2px solid #5f5fff;} .color{color:#5f5fff;}",
        "orange_vibrant": "background:#2a1a0a; .card{background:#4a2a10; border:2px solid #ff8844;} .color{color:#ff8844;}",
        "pink_dream": "background:#2a0a2a; .card{background:#4a1a4a; border:2px solid #ff66cc;} .color{color:#ff66cc;}",
        "dark_blue": "background:#0a0a2a; .card{background:#0f1a3a; border:2px solid #3a6ea5;} .color{color:#3a6ea5;}",
        "light_green": "background:#1a2a1a; .card{background:#2a4a2a; border:2px solid #88ff88;} .color{color:#88ff88;}",
        "midnight": "background:#0a0a1a; .card{background:#1a1a3a; border:2px solid #8888ff;} .color{color:#8888ff;}",
        "hot_pink": "background:#2a0a1a; .card{background:#4a1a2a; border:2px solid #ff66aa;} .color{color:#ff66aa;}",
        "silver_metal": "background:#2a2a2a; .card{background:#4a4a4a; border:2px solid #cccccc;} .color{color:#cccccc;}",
        "mystic": "background:#1a0a2a; .card{background:#2a1a4a; border:2px solid #aa66ff;} .color{color:#aa66ff;}",
        "classic": "background:#1a1a2a; .card{background:#16213e; border:2px solid #e94560;} .color{color:#ffd700;}",
        "blood_red": "background:#2a0a0a; .card{background:#4a0a0a; border:3px solid #ff0000;} .color{color:#ff4444; text-shadow:0 0 5px red;}"
    }
    return themes.get(theme_name, themes["classic"])

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Wingo 20-Server Ultimate</title>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="5">
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; transition: all 0.3s; }
        .container { max-width: 800px; margin: auto; }
        .card { border-radius: 20px; padding: 20px; box-shadow: 0 0 20px rgba(0,0,0,0.5); text-align: center; }
        .color { font-size: 48px; font-weight: bold; margin: 10px; }
        .confidence { font-size: 24px; margin: 10px; }
        .details { display: flex; justify-content: center; gap: 30px; margin: 20px 0; }
        .detail { background: rgba(0,0,0,0.3); padding: 10px 20px; border-radius: 15px; }
        .label { font-size: 14px; opacity: 0.8; }
        .value { font-size: 28px; font-weight: bold; }
        .vote-box { background: rgba(0,0,0,0.3); border-radius: 15px; padding: 15px; margin-top: 20px; text-align: left; max-height: 300px; overflow-y: auto; }
        .vote { border-bottom: 1px solid rgba(255,255,255,0.2); padding: 5px; font-size: 12px; }
        .footer { margin-top: 20px; font-size: 12px; opacity: 0.7; }
        .badge { background: #ff4444; display: inline-block; padding: 3px 8px; border-radius: 20px; font-size: 12px; margin-left: 10px; }
        .weights-panel { background: rgba(0,0,0,0.3); border-radius: 10px; padding: 10px; margin-top: 15px; font-size: 11px; text-align: left; max-height: 200px; overflow-y: auto; }
        .weight-item { display: inline-block; margin: 3px 8px; }
        .good { color: #44ff44; }
        h3 { margin: 5px 0; font-size: 14px; }
        select, button { padding: 8px 12px; margin: 10px; border-radius: 8px; border: none; cursor: pointer; background: #2a2a5a; color: white; }
        button:hover { background: #e94560; }
        .server-selector { margin-bottom: 20px; }
    </style>
    <style id="theme-style">{{ theme_css|safe }}</style>
</head>
<body>
<div class="container">
    <div class="server-selector">
        <label>🔀 Switch Server: </label>
        <select id="serverSelect" onchange="changeServer()">
            {% for sid, name in server_list %}
            <option value="{{ sid }}" {% if sid == current_server %}selected{% endif %}>{{ sid }}. {{ name }}</option>
            {% endfor %}
        </select>
        <button onclick="changeServer()">Go</button>
    </div>
    <div class="card">
        <h1>🎲 {{ server_name }} <span class="badge">v13</span></h1>
        <div class="color {{ color_class }}">{{ color }}</div>
        <div class="confidence">Confidence: {{ confidence }}%</div>
        <div class="details">
            <div class="detail">
                <div class="label">🔢 NUMBER</div>
                <div class="value">{{ number }}</div>
            </div>
            <div class="detail">
                <div class="label">📏 BIG/SMALL</div>
                <div class="value">{{ size }}</div>
            </div>
        </div>
        <div class="vote-box">
            <strong>🧠 ENGINE VOTES</strong>
            {% for vote in votes %}
            <div class="vote">{{ vote[2] }} → {{ vote[0] }} ({{ vote[1] }}%)</div>
            {% endfor %}
        </div>
        <div class="weights-panel">
            <h3>🤖 Model Weights & Accuracy</h3>
            {% for name, data in weights.items() %}
            <div class="weight-item">
                {{ name }}: <span class="good">{{ data.weight }}</span> (acc {{ data.accuracy }}%)
            </div>
            {% endfor %}
        </div>
        <div class="footer">Auto-refresh 5s | AI learns automatically from live draws</div>
    </div>
</div>
<script>
function changeServer() {
    var sid = document.getElementById('serverSelect').value;
    window.location.href = '/server/' + sid;
}
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return redirect('/server/1')

@app.route('/server/<int:sid>')
def server_page(sid):
    if sid not in engines:
        return "Server not found", 404
    eng = engines[sid]
    if not game_history:
        return render_template_string(HTML_TEMPLATE,
            theme_css=get_theme_css(server_configs[sid-1]["theme"]),
            server_list=[(c["id"], c["name"]) for c in server_configs],
            current_server=sid,
            server_name=server_configs[sid-1]["name"],
            color="Loading...", color_class="", confidence=0,
            number="?", size="?", votes=[], weights={})
    pred = eng.predict(game_history)
    if not pred:
        color = "N/A"
        confidence = 0
        number = "?"
        size = "?"
        votes = []
        weights = {}
    else:
        num = pred["number"]
        color = NUMBER_TO_COLOR.get(num, "RED")
        if num in NUMBER_TO_VIOLET:
            color = NUMBER_TO_VIOLET[num]
        size = NUMBER_TO_SIZE[num]
        confidence = pred["confidence"]
        votes = []
        for mname, mdata in pred["model_predictions"].items():
            mnum = mdata["number"]
            mcolor = NUMBER_TO_COLOR.get(mnum, "RED")
            if mnum in NUMBER_TO_VIOLET:
                mcolor = NUMBER_TO_VIOLET[mnum]
            votes.append((mcolor, mdata["confidence"], mname))
        number = num
        weights = eng.get_weights_summary()
    return render_template_string(HTML_TEMPLATE,
        theme_css=get_theme_css(server_configs[sid-1]["theme"]),
        server_list=[(c["id"], c["name"]) for c in server_configs],
        current_server=sid,
        server_name=server_configs[sid-1]["name"],
        color=color,
        color_class=color.lower(),
        confidence=confidence,
        number=number,
        size=size,
        votes=votes[:15],
        weights=weights)

@app.route('/api/predict/<int:sid>')
def api_predict(sid):
    if sid not in engines or not game_history:
        return jsonify({"error": "No data"})
    pred = engines[sid].predict(game_history)
    if not pred:
        return jsonify({"error": "Prediction failed"})
    num = pred["number"]
    return jsonify({
        "number": num,
        "color": NUMBER_TO_COLOR.get(num, "RED"),
        "size": NUMBER_TO_SIZE[num],
        "confidence": pred["confidence"],
        "model_details": pred["model_predictions"]
    })

if __name__ == '__main__':
    print("🚀 Wingo 20-Server Ultimate System starting...")
    print("📡 Fetching initial data...")
    time.sleep(2)
    print("✅ Server running at http://localhost:5000")
    print("💡 Use dropdown to switch between 20 independent prediction engines")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
