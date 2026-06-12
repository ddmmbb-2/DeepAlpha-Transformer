import tkinter as tk
from tkinter import scrolledtext, ttk
import subprocess
import threading
import queue
import sys
import os

class QuantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DeepAlpha-Transformer 量化交易控制台 🚀")
        self.root.geometry("850x650")
        self.root.configure(padx=20, pady=20)
        
        # 建立 UI 區塊
        self.create_widgets()
        
        # 用於安全跨執行緒更新 UI 的隊列
        self.log_queue = queue.Queue()
        self.root.after(100, self.process_log_queue)
        
        # 標記目前是否正在執行
        self.is_running = False

    def create_widgets(self):
        # 標題
        title_label = tk.Label(self.root, text="V5 實盤自動化管線", font=("Helvetica", 18, "bold"))
        title_label.pack(pady=(0, 10))

        # 按鈕框架
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.X, pady=10)

        # 單步執行按鈕
        ttk.Button(btn_frame, text="1. 下載最新數據 (dl.py)", width=25, command=lambda: self.run_script("dl.py")).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(btn_frame, text="2. 重構特徵 (add.py)", width=25, command=lambda: self.run_script("add.py")).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(btn_frame, text="3. 啟動訓練 (v1.py)", width=25, command=self.run_train).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(btn_frame, text="4. 實盤推薦 (infer.py)", width=25, command=self.run_infer).grid(row=1, column=1, padx=5, pady=5)

        # 一鍵自動化按鈕 (醒目顏色)
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Helvetica", 12, "bold"), foreground="blue")
        ttk.Button(btn_frame, text="🔥 一鍵全自動執行 (Run All)", width=52, style="Accent.TButton", command=self.run_all).grid(row=2, column=0, columnspan=2, padx=5, pady=15)

        # 日誌顯示區
        tk.Label(self.root, text="系統執行日誌 (Console Output):", font=("Helvetica", 12)).pack(anchor=tk.W)
        self.log_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=100, height=20, bg="black", fg="lightgreen", font=("Consolas", 10))
        self.log_area.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 清除按鈕
        ttk.Button(self.root, text="清除日誌", command=lambda: self.log_area.delete(1.0, tk.END)).pack(anchor=tk.E)

    def log(self, message):
        """將訊息推入隊列，等待主執行緒更新 UI"""
        self.log_queue.put(message)

    def process_log_queue(self):
        """主執行緒定時從隊列取出訊息並顯示，避免 UI 卡死"""
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            self.log_area.insert(tk.END, msg)
            self.log_area.see(tk.END) # 自動捲動到底部
        self.root.after(100, self.process_log_queue)

    def execute_command(self, cmd_list, completion_callback=None):
        """在背景執行緒中執行命令"""
        if self.is_running:
            self.log("⚠️ 警告：已有程式正在執行中，請稍後...\n")
            return
            
        self.is_running = True
        self.log(f"▶️ 開始執行指令: {' '.join(cmd_list)}\n")
        self.log("-" * 60 + "\n")
        
        def target():
            try:
                # 🆕 複製當下環境，並強制加入 Python 的 UTF-8 輸出指令
                my_env = os.environ.copy()
                my_env["PYTHONIOENCODING"] = "utf-8"

                process = subprocess.Popen(
                    cmd_list,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    env=my_env,            # 🆕 使用我們加料過的環境變數
                    cwd=os.getcwd()        
                )
                
                # 即時讀取輸出
                for line in process.stdout:
                    self.log(line)
                    
                process.wait()
                self.log("-" * 60 + "\n")
                self.log(f"✅ 執行完畢 (退出碼: {process.returncode})\n\n")
                
            except Exception as e:
                self.log(f"❌ 執行發生錯誤: {str(e)}\n\n")
            finally:
                self.is_running = False
                if completion_callback and process.returncode == 0:
                    completion_callback()
                    
        # 啟動背景執行緒
        threading.Thread(target=target, daemon=True).start()

    def run_script(self, script_name):
        """執行無參數的基礎腳本"""
        self.execute_command(["python", script_name])

    def run_train(self):
        """執行帶有複雜參數的 v1.py 訓練"""
        cmd = [
            "python", "v1.py",
            "--data", "processed_data/features.npz",
            "--max_stocks", "500",
            "--batch_size", "4",
            "--gru_hidden", "192",
            "--num_layers", "4",
            "--embed_dim", "32",
            "--pos_weight", "5.0",
            "--epochs", "20",
            "--lr", "5e-5",
            "--train_window", "2000",
            "--step_size", "250",
            "--device", "cuda"
        ]
        self.execute_command(cmd)

    def run_infer(self, completion_callback=None):
        """執行 infer.py 推論"""
        cmd = [
            "python", "infer.py",
            "--model", "checkpoints/fold_3_best.pth",
            "--gru_hidden", "192",
            "--embed_dim", "32",
            "--top_k", "50",
            "--device", "cuda"
        ]
        self.execute_command(cmd, completion_callback)

    def run_all(self):
        """串聯執行所有步驟：dl -> add -> v1 -> infer"""
        if self.is_running:
            self.log("⚠️ 系統正在忙碌中...\n")
            return
            
        self.log("🚀 啟動【一鍵全自動管線】...\n")
        
        # 定義回呼函數 (Callback) 來達成順序執行
        def step4_infer():
            self.log("🔄 [步驟 4/4] 啟動實盤預測...\n")
            self.run_infer(lambda: self.log("🎉 全自動管線執行完畢！請查看 recommendations.csv\n"))

        def step3_train():
            self.log("🔄 [步驟 3/4] 啟動 V5 模型更新訓練...\n")
            # 注意：訓練指令與獨立按鈕相同
            cmd = ["python", "v1.py", "--data", "processed_data/features.npz", "--max_stocks", "500", "--batch_size", "4", "--gru_hidden", "192", "--num_layers", "4", "--embed_dim", "32", "--pos_weight", "5.0", "--epochs", "20", "--lr", "5e-5", "--train_window", "2000", "--step_size", "250", "--device", "cuda"]
            self.execute_command(cmd, step4_infer)

        def step2_add():
            self.log("🔄 [步驟 2/4] 開始重構特徵...\n")
            self.execute_command(["python", "add.py"], step3_train)

        def step1_dl():
            self.log("🔄 [步驟 1/4] 開始下載最新日線數據...\n")
            self.execute_command(["python", "dl.py"], step2_add)

        # 啟動第一步
        step1_dl()

if __name__ == "__main__":
    root = tk.Tk()
    app = QuantApp(root)
    root.mainloop()