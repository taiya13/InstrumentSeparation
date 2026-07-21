"""
gui.py — Windows-friendly research GUI for orchestral instrument separation.

Thin view over app.engine.SeparationEngine (which does all the real work). The engine
runs in a background thread; progress/results are marshalled back to the Tk main thread
via a queue, so the UI stays responsive.

Model-agnostic: the model dropdown is populated from configs/app_default.yaml. Swapping or
adding a model = edit that config / drop a new checkpoint; this file never changes.

Run (Windows):  scripts\run_gui.bat      (or)  python src\app\gui.py
Depends only on the Python standard library (tkinter) + the project's engine deps.
"""
import os
import sys
import threading
import queue

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from app.engine import SeparationEngine

AUDIO_TYPES = [("Audio files", "*.wav *.flac *.mp3 *.ogg *.aiff *.aif"), ("All files", "*.*")]
DEFAULT_CONFIG = os.path.join(_SRC, "..", "configs", "app_default.yaml")


class SeparationGUI:
    def __init__(self, root, config_path=DEFAULT_CONFIG):
        self.root = root
        self.root.title("Orchestra Instrument Separation — Research Tool")
        self.root.geometry("820x620")
        self.engine = SeparationEngine(config_path)
        self.models = self.engine.list_models()
        self._disp2key = {m["display"] + ("" if m["available"] else "  [未整備]"): m["key"]
                          for m in self.models}
        self.q = queue.Queue()
        self.worker = None
        self._build()
        self._log(f"device={self.engine.device} | GPU={self.engine.gpu_info.get('gpu_name')} "
                  f"({self.engine.gpu_info.get('vram_gb')} GB)")
        self.root.after(100, self._poll)

    # ---------------------------------------------------------------- UI
    def _build(self):
        pad = dict(padx=8, pady=4)
        frm = ttk.Frame(self.root)
        frm.pack(fill="x", **pad)

        # input file
        ttk.Label(frm, text="音源ファイル (WAV/FLAC/MP3):").grid(row=0, column=0, sticky="w")
        self.in_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.in_var, width=70).grid(row=1, column=0, sticky="we")
        ttk.Button(frm, text="選択...", command=self._browse_in).grid(row=1, column=1)

        # output folder
        ttk.Label(frm, text="出力フォルダ:").grid(row=2, column=0, sticky="w")
        self.out_var = tk.StringVar(value=os.path.abspath("separated"))
        ttk.Entry(frm, textvariable=self.out_var, width=70).grid(row=3, column=0, sticky="we")
        ttk.Button(frm, text="選択...", command=self._browse_out).grid(row=3, column=1)

        # reference manifest (optional, for evaluation)
        ttk.Label(frm, text="参照 manifest.json (任意・評価用):").grid(row=4, column=0, sticky="w")
        self.ref_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.ref_var, width=70).grid(row=5, column=0, sticky="we")
        ttk.Button(frm, text="選択...", command=self._browse_ref).grid(row=5, column=1)

        # model + presence
        opt = ttk.Frame(self.root)
        opt.pack(fill="x", **pad)
        ttk.Label(opt, text="モデル:").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar()
        vals = list(self._disp2key.keys())
        self.model_cb = ttk.Combobox(opt, textvariable=self.model_var, values=vals,
                                     width=48, state="readonly")
        if vals:
            self.model_cb.current(0)
        self.model_cb.grid(row=0, column=1, sticky="w")
        ttk.Label(opt, text="   Presence:").grid(row=0, column=2, sticky="w")
        self.presence_var = tk.StringVar(value="all")
        ttk.Combobox(opt, textvariable=self.presence_var, values=["all", "heuristic"],
                     width=10, state="readonly").grid(row=0, column=3, sticky="w")

        # run button + progress
        run = ttk.Frame(self.root)
        run.pack(fill="x", **pad)
        self.run_btn = ttk.Button(run, text="分離開始", command=self._start)
        self.run_btn.pack(side="left")
        self.open_btn = ttk.Button(run, text="出力フォルダを開く", command=self._open_out,
                                   state="disabled")
        self.open_btn.pack(side="left", padx=6)
        self.progress = ttk.Progressbar(run, mode="determinate", maximum=100)
        self.progress.pack(side="left", fill="x", expand=True, padx=8)
        self.status = tk.StringVar(value="待機中")
        ttk.Label(run, textvariable=self.status, width=28).pack(side="left")

        # results table
        res = ttk.LabelFrame(self.root, text="結果")
        res.pack(fill="both", expand=True, **pad)
        self.summary = tk.StringVar(value="")
        ttk.Label(res, textvariable=self.summary, justify="left").pack(anchor="w", padx=6, pady=4)
        cols = ("id", "楽器", "ファイル", "SI-SDR")
        self.tree = ttk.Treeview(res, columns=cols, show="headings", height=8)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=160 if c != "id" else 90)
        self.tree.pack(fill="both", expand=True, padx=6, pady=4)

        # log
        logf = ttk.LabelFrame(self.root, text="ログ")
        logf.pack(fill="both", expand=False, **pad)
        self.logtext = tk.Text(logf, height=6, wrap="word")
        self.logtext.pack(fill="both", expand=True, padx=6, pady=4)

    # ---------------------------------------------------------------- helpers
    def _browse_in(self):
        p = filedialog.askopenfilename(filetypes=AUDIO_TYPES)
        if p:
            self.in_var.set(p)

    def _browse_out(self):
        p = filedialog.askdirectory()
        if p:
            self.out_var.set(p)

    def _browse_ref(self):
        p = filedialog.askopenfilename(filetypes=[("Manifest", "*.json"), ("All", "*.*")])
        if p:
            self.ref_var.set(p)

    def _open_out(self):
        d = self._last_out or self.out_var.get()
        try:
            if sys.platform.startswith("win"):
                os.startfile(d)                      # noqa (Windows only)
            elif sys.platform == "darwin":
                os.system(f'open "{d}"')
            else:
                os.system(f'xdg-open "{d}"')
        except Exception as e:
            self._log(f"フォルダを開けません: {e}")

    def _log(self, msg):
        self.logtext.insert("end", msg + "\n")
        self.logtext.see("end")

    # ---------------------------------------------------------------- run
    def _start(self):
        inp = self.in_var.get().strip()
        if not inp or not os.path.exists(inp):
            messagebox.showerror("エラー", "音源ファイルを選択してください。")
            return
        if not self.model_var.get():
            messagebox.showerror("エラー", "モデルを選択してください。")
            return
        model_key = self._disp2key[self.model_var.get()]
        out = self.out_var.get().strip() or "separated"
        ref = self.ref_var.get().strip() or None
        presence = self.presence_var.get()
        self.run_btn.config(state="disabled")
        self.open_btn.config(state="disabled")
        self.progress["value"] = 0
        self.tree.delete(*self.tree.get_children())
        self.summary.set("")
        self._last_out = out
        self._log(f"開始: {os.path.basename(inp)} | model={model_key} | presence={presence}")

        def work():
            def cb(frac, msg):
                self.q.put(("progress", frac, msg))
            rec = self.engine.separate_file(inp, out, model_key, presence_mode=presence,
                                            reference_manifest=ref, progress=cb)
            self.q.put(("done", rec))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                if item[0] == "progress":
                    _, frac, msg = item
                    self.progress["value"] = int(frac * 100)
                    self.status.set(msg)
                elif item[0] == "done":
                    self._done(item[1])
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _done(self, rec):
        self.run_btn.config(state="normal")
        self.open_btn.config(state="normal")
        if rec.get("error"):
            self.status.set("エラー")
            self._log("ERROR: " + rec["error"]["type"] + ": " + rec["error"]["message"])
            messagebox.showerror("分離エラー", rec["error"]["message"])
            return
        m, mt = rec["model"], rec["metrics"]
        g = mt["gpu"]
        self.summary.set(
            f"モデル: {m['display']} ({m['backbone']}, trained={m['trained']}, {m['params_m']}M)\n"
            f"推論時間: {mt['inference_time_sec']} s  (RTF {mt['realtime_factor']}, "
            f"音声長 {mt['audio_duration_sec']} s) | device={rec['device']}\n"
            f"GPU使用率(avg/peak): {g['gpu_util_avg_pct']}/{g['gpu_util_peak_pct']} % | "
            f"VRAM: {g['vram_used_peak_mb']}/{g['vram_total_mb']} MB (src={g['source']})\n"
            f"分離楽器数: {rec['num_instruments']}  |  出力: {rec['output_subdir']}")
        # per-instrument SI-SDR if evaluated
        eval_cat = (rec.get("evaluation") or {}).get("per_category") or {}
        for ins in rec["instruments"]:
            sisdr = ""
            if ins["id"] in eval_cat:
                sisdr = eval_cat[ins["id"]].get("mean_si_sdr", "")
            self.tree.insert("", "end", values=(ins["id"], ins["name_ja"], ins["file"], sisdr))
        self.status.set("完了")
        self._log(f"完了: {rec['num_instruments']} 楽器を保存 -> {rec['output_subdir']}")

    _last_out = None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    a = ap.parse_args()
    root = tk.Tk()
    SeparationGUI(root, a.config)
    root.mainloop()


if __name__ == "__main__":
    main()
