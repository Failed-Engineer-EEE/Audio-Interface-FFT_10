import time
import os
import csv
import re
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.ticker import ScalarFormatter
import sounddevice as sd
import scipy.signal

class AudioEMCGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Interface FFT")
        self.root.geometry("1450x850")
        
        # --- UI Safety: Handle Window Close Gracefully ---
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.default_font = ('Arial', 11)
        self.root.option_add('*Font', self.default_font)
        
        plt.rcParams.update({
            'font.size': 11, 
            'axes.labelsize': 13, 
            'axes.titlesize': 14,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11
        })
        
        # Continuous Streaming Variables
        self.is_continuous = False
        self.stream = None
        self.loop_id = None
        self.audio_buffer = np.zeros(2000000, dtype='float32')
        
        # High-Speed Rendering Variables
        self.trace_line = None
        self.limit_line = None
        self.margin_line = None
        self.peak_annotations = []
        
        # Locked Display State (Updated only by the Apply Button)
        self.active_s_hz = 30.0
        self.active_e_hz = 96000.0
        self.active_y_min = -20.0
        self.active_y_max = 140.0
        
        # Data Storage
        self.limit_points = []
        self.correction_files = [] 
        
        self.last_freqs = None
        self.last_vals = None
        self.last_unit = "dBuV"
        
        self.create_widgets()
        self.populate_audio_devices()
        
    def populate_audio_devices(self):
        devices = sd.query_devices()
        seen_names = set()
        input_devices = []
        
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                name = dev['name']
                # Clean deduplication: Only add if we haven't seen this exact device name yet
                if name not in seen_names:
                    seen_names.add(name)
                    input_devices.append(f"{i}: {name}")
                    
        self.device_combo['values'] = input_devices
        if input_devices:
            scarlett_idx = next((i for i, v in enumerate(input_devices) if "Scarlett" in v), 0)
            self.device_combo.current(scarlett_idx)

    def parse_freq_hz(self, text):
        t = str(text).lower().replace(' ', '')
        try:
            if 'ghz' in t: return float(t.replace('ghz', '')) * 1e9
            if 'mhz' in t: return float(t.replace('mhz', '')) * 1e6
            if 'khz' in t: return float(t.replace('khz', '')) * 1000.0
            if 'hz' in t: return float(t.replace('hz', ''))
            return float(t)
        except ValueError:
            return 0.0

    def create_widgets(self):
        # --- LEFT PANEL ---
        control_frame = tk.Frame(self.root, width=550, padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        col1 = tk.Frame(control_frame)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        col2 = tk.Frame(control_frame)
        col2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # --- COLUMN 1 (Audio, Span, Execution) ---
        tk.Label(col1, text="1. Audio Interface Setup", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        self.device_combo = ttk.Combobox(col1, state="readonly", width=35)
        self.device_combo.pack(fill=tk.X, pady=4)
        tk.Button(col1, text="Refresh Devices", command=self.populate_audio_devices).pack(fill=tk.X, pady=4)
        
        tk.Label(col1, text="Sample Rate (Hz):", font=('Arial', 10)).pack(anchor=tk.W, pady=(8,0))
        sr_options = [
            "44100 (Max: 22.05 kHz)",
            "48000 (Max: 24.0 kHz)",
            "88200 (Max: 44.1 kHz)",
            "96000 (Max: 48.0 kHz)",
            "176400 (Max: 88.2 kHz)",
            "192000 (Max: 96.0 kHz)"
        ]
        self.sr_combo = ttk.Combobox(col1, values=sr_options, state="readonly")
        self.sr_combo.set(sr_options[5])
        self.sr_combo.pack(fill=tk.X, pady=4)
        
        tk.Label(col1, text="FFT Size:", font=('Arial', 10)).pack(anchor=tk.W, pady=(8,0))
        block_options = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
        self.bs_combo = ttk.Combobox(col1, values=block_options, state="readonly")
        self.bs_combo.set(65536)
        self.bs_combo.pack(fill=tk.X, pady=4)

        self.cal_offset_entry = self.make_input(col1, "Calibration Offset (dB):", "120.0")

        # Visual Separator with increased padding
        tk.Frame(col1, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, pady=15)
        
        # 2. Graph Span
        tk.Label(col1, text="2. Frequency", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        self.start_entry = self.make_input(col1, "Start Freq (X-Min):", "30Hz")
        self.stop_entry = self.make_input(col1, "Stop Freq (X-Max):", "96kHz")
        
        tk.Label(col1, text="", font=('Arial', 2)).pack() # Minor spacer
        
        self.y_min_entry = self.make_input(col1, "Amplitude Min (Y-Min):", "-20")
        self.y_max_entry = self.make_input(col1, "Amplitude Max (Y-Max):", "140")
        
        tk.Button(col1, text="Apply Graph Settings", command=self.apply_limits, bg="#b2dfdb", font=('Arial', 10, 'bold')).pack(fill=tk.X, pady=10)

        # Visual Separator with increased padding
        tk.Frame(col1, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, pady=15)

        # 3. Execution & Export
        tk.Label(col1, text="3. Execution & Export", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        tk.Button(col1, text="Single Scan", font=('Arial', 11, 'bold'), command=self.single_sweep, bg="#d1e7dd").pack(fill=tk.X, pady=6)
        self.cont_btn = tk.Button(col1, text="Continuous Sweep", font=('Arial', 11, 'bold'), command=self.toggle_continuous, bg="#fff3cd")
        self.cont_btn.pack(fill=tk.X, pady=6)
        
        tk.Label(col1, text="CSV Data Density:", font=('Arial', 10)).pack(anchor=tk.W, pady=(15,0))
        self.density_combo = ttk.Combobox(col1, values=["Export All Points", "Log: 10 Pts/Decade", "Log: 100 Pts/Decade", "Log: 1000 Pts/Decade"], state="readonly")
        self.density_combo.set("Export All Points")
        self.density_combo.pack(fill=tk.X, pady=4)
        
        tk.Button(col1, text="💾 Export to CSV", font=('Arial', 11, 'bold'), command=self.export_csv, bg="#cfe2ff").pack(fill=tk.X, pady=6)

        # Spacer to push everything up in col1 nicely
        tk.Frame(col1).pack(expand=True, fill=tk.BOTH)

        # --- COLUMN 2 (Units, Limits, Probes) ---
        tk.Label(col2, text="4. Select Units", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        self.unit_combo = ttk.Combobox(col2, values=["dBuV", "dBV", "dBm", "dBuA", "dBpT", "dBuV/m"], state="readonly")
        self.unit_combo.set("dBuV")
        self.unit_combo.pack(fill=tk.X, pady=4)
        self.marker_entry = self.make_input(col2, "Top Peaks to Mark:", "3")

        zoom_frame = tk.Frame(col2)
        zoom_frame.pack(fill=tk.X, pady=5)
        tk.Button(zoom_frame, text="⟲ Undo Zoom", command=lambda: self.toolbar.back(), bg="#e0e0e0").pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(zoom_frame, text="🏠 Reset", command=lambda: self.toolbar.home(), bg="#e0e0e0").pack(side=tk.RIGHT, expand=True, fill=tk.X)

        tk.Frame(col2, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, pady=15)

        # 5. Limit Builder
        tk.Label(col2, text="5. Apply Limit", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        self.margin_entry = self.make_input(col2, "Margin (dB):", "6.0")
        
        input_frame = tk.Frame(col2)
        input_frame.pack(fill=tk.X, pady=4)
        tk.Label(input_frame, text="F:").grid(row=0, column=0)
        self.lim_f_entry = tk.Entry(input_frame, width=8); self.lim_f_entry.grid(row=0, column=1)
        tk.Label(input_frame, text="A:").grid(row=0, column=2)
        self.lim_a_entry = tk.Entry(input_frame, width=8); self.lim_a_entry.grid(row=0, column=3)
        tk.Button(input_frame, text="Add", command=self.add_limit_point, bg="#e1f5fe").grid(row=0, column=4, padx=2)
        
        tk.Button(col2, text="Select Limit File", command=self.load_limit_file, bg="#e0f2f1").pack(fill=tk.X, pady=4)
        
        # Expanding Listbox to absorb dead space
        self.point_listbox = tk.Listbox(col2, font=('Arial', 10))
        self.point_listbox.pack(fill=tk.BOTH, expand=True, pady=4)
        
        ctrl_lim_btns = tk.Frame(col2)
        ctrl_lim_btns.pack(fill=tk.X)
        tk.Button(ctrl_lim_btns, text="Del Sel", command=self.delete_selected_limit, bg="#ffcc80").pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(ctrl_lim_btns, text="Clear", command=lambda: [self.limit_points.clear(), self.refresh_point_listbox(), self.force_redraw()], bg="#ffebee").pack(side=tk.RIGHT, expand=True, fill=tk.X)

        tk.Frame(col2, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, pady=15)

        # 6. Probes
        tk.Label(col2, text="6. Correction Factor", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        
        # Expanding Listbox for probes
        self.corr_listbox = tk.Listbox(col2, font=('Arial', 10))
        self.corr_listbox.pack(fill=tk.BOTH, expand=True, pady=4)
        
        btn_frame = tk.Frame(col2)
        btn_frame.pack(fill=tk.X)
        tk.Button(btn_frame, text="Add File", command=self.load_correction_file, bg="#e1f5fe").pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(btn_frame, text="Clear", command=lambda: [self.correction_files.clear(), self.corr_listbox.delete(0, tk.END), self.force_redraw()], bg="#ffebee").pack(side=tk.RIGHT, expand=True, fill=tk.X)

        # --- GRAPH PANEL ---
        self.graph_frame = tk.Frame(self.root, bg="white")
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.graph_frame)
        self.toolbar.update()

    def on_closing(self):
        self.is_continuous = False
        if self.loop_id:
            self.root.after_cancel(self.loop_id)
            self.loop_id = None
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.root.destroy()

    def make_input(self, parent, label_text, default_val):
        tk.Label(parent, text=label_text).pack(anchor=tk.W, pady=(8,0))
        entry = tk.Entry(parent, font=('Arial', 11))
        entry.insert(0, default_val)
        entry.pack(fill=tk.X, pady=4)
        return entry

    def apply_limits(self):
        s_hz = self.parse_freq_hz(self.start_entry.get())
        e_hz = self.parse_freq_hz(self.stop_entry.get())
        
        if s_hz <= 0: s_hz = 30.0
        if e_hz <= 0: e_hz = 10000.0
        if s_hz >= e_hz: 
            e_hz = s_hz + 1000.0
        
        self.active_s_hz = s_hz
        self.active_e_hz = e_hz

        try: y_min = float(self.y_min_entry.get())
        except ValueError: y_min = -20.0
        try: y_max = float(self.y_max_entry.get())
        except ValueError: y_max = 140.0
        
        if y_min >= y_max:
            y_max = y_min + 10.0

        self.active_y_min = y_min
        self.active_y_max = y_max
        
        self.force_redraw()

    def force_redraw(self):
        self.trace_line = None
        self.peak_annotations.clear() 

    def export_csv(self):
        if self.last_freqs is None or len(self.last_freqs) == 0:
            messagebox.showwarning("No Data", "Run a scan first to collect data.")
            return

        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")], title="Save Sweep Data")
        if not filepath: return
            
        try:
            density_mode = self.density_combo.get()
            if "All Points" in density_mode:
                out_freqs, out_vals = self.last_freqs, self.last_vals
            else:
                pts_per_decade = int(re.search(r'\d+', density_mode).group())
                s_hz, e_hz = self.last_freqs[0], self.last_freqs[-1]
                num_decades = np.log10(e_hz) - np.log10(s_hz)
                total_pts = int(pts_per_decade * num_decades)
                out_freqs = np.logspace(np.log10(s_hz), np.log10(e_hz), total_pts)
                out_vals = np.interp(out_freqs, self.last_freqs, self.last_vals)

            with open(filepath, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Frequency (Hz)", f"Amplitude ({self.last_unit})"])
                for freq, val in zip(out_freqs, out_vals):
                    writer.writerow([f"{freq:.4f}", f"{val:.4f}"])
            messagebox.showinfo("Success", f"Data saved ({len(out_freqs)} points) to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save CSV:\n{e}")

    def load_limit_file(self):
        filepath = filedialog.askopenfilename(title="Select Limit File", filetypes=[("Limit Files", "*.lim *.csv *.txt"), ("All Files", "*.*")])
        if not filepath: return
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                current_f = None
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith('['): continue
                    if '=' in line:
                        key, val = [x.strip() for x in line.split('=', 1)]
                        if key.lower().startswith('freq'):
                            try: current_f = float(val.replace('.', ''))
                            except ValueError: current_f = None
                        elif key.lower().startswith('lev') and current_f is not None:
                            try:
                                self.limit_points.append((current_f, float(val)))
                                current_f = None
                            except ValueError: pass
                        continue
                    parts = line.replace(',', ' ').replace(';', ' ').split()
                    numeric = [float(p) for p in parts if p.replace('.','',1).replace('-','',1).isdigit()]
                    if len(numeric) >= 2:
                        self.limit_points.append((numeric[-2], numeric[-1]))
            
            self.limit_points.sort(key=lambda x: x[0])
            self.refresh_point_listbox()
            self.force_redraw()
        except Exception as e: messagebox.showerror("Error", f"Failed to parse file:\n{e}")

    def add_limit_point(self):
        freq_hz = self.parse_freq_hz(self.lim_f_entry.get().strip())
        try: amp = float(self.lim_a_entry.get().strip())
        except ValueError: return
        self.limit_points.append((freq_hz, amp))
        self.limit_points.sort(key=lambda x: x[0])
        self.refresh_point_listbox()
        self.force_redraw()

    def delete_selected_limit(self):
        sel = self.point_listbox.curselection()
        if sel:
            self.limit_points.pop(sel[0])
            self.refresh_point_listbox()
            self.force_redraw()

    def refresh_point_listbox(self):
        self.point_listbox.delete(0, tk.END)
        for f, a in self.limit_points:
            display_f = f"{f:.1f} Hz" if f < 1000 else f"{f/1000:.2f} kHz"
            self.point_listbox.insert(tk.END, f"{display_f} -> {a}")

    def load_correction_file(self):
        filepath = filedialog.askopenfilename(title="Select Correction File", filetypes=[("EMC Files", "*.csv *.cor *.txt *.lsc")])
        if not filepath: return
        try:
            freqs, factors = [], []
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                current_f = None
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith('['): continue
                    if '=' in line:
                        key, val = [x.strip() for x in line.split('=', 1)]
                        if key.lower().startswith('freq'):
                            try: current_f = float(val.replace('.', ''))
                            except ValueError: current_f = None
                        elif key.lower().startswith('lev') and current_f is not None:
                            try:
                                freqs.append(current_f); factors.append(float(val)); current_f = None
                            except ValueError: pass
                        continue
                    parts = line.replace(',', ' ').replace(';', ' ').split()
                    numeric_vals = [float(p) for p in parts if p.replace('.','',1).replace('-','',1).isdigit()]
                    if len(numeric_vals) >= 2:
                        freqs.append(numeric_vals[-2]); factors.append(numeric_vals[-1])
            if freqs:
                idx = np.argsort(freqs)
                self.correction_files.append((os.path.basename(filepath), np.array(freqs)[idx], np.array(factors)[idx]))
                self.corr_listbox.insert(tk.END, os.path.basename(filepath))
                self.force_redraw()
        except Exception as e: messagebox.showerror("Error", f"Failed to read file:\n{e}")

    # --- High-Speed Audio Engine ---
    def audio_callback(self, indata, frames, time_info, status):
        self.audio_buffer = np.roll(self.audio_buffer, -frames)
        self.audio_buffer[-frames:] = indata[:, 0]

    def fetch_and_plot(self, stream_mode=False):
        device_str = self.device_combo.get()
        if not device_str: return
        device_id = int(device_str.split(':')[0])
        
        try:
            sr_string = self.sr_combo.get()
            active_sr = int(sr_string.split()[0])
            active_bs = int(self.bs_combo.get())

            if stream_mode:
                audio_array = self.audio_buffer[-active_bs:].copy()
            else:
                try:
                    audio_data = sd.rec(active_bs, samplerate=active_sr, channels=1, device=device_id, dtype='float32')
                    sd.wait() 
                    audio_array = audio_data[:, 0]
                except sd.PortAudioError as e:
                    messagebox.showerror("Audio Device Error", f"The audio device does not support a sample rate of {active_sr} Hz.\n\nDetails: {e}")
                    self.is_continuous = False
                    return
            
            # Math
            window = scipy.signal.windows.hann(len(audio_array))
            spectrum = np.fft.rfft(audio_array * window)
            frequencies_hz = np.fft.rfftfreq(active_bs, 1 / active_sr)
            
            mag = np.abs(spectrum) / (active_bs / 2)
            mag[mag < 1e-12] = 1e-12 
            db_fs = 20 * np.log10(mag)
            
            try: cal_offset = float(self.cal_offset_entry.get())
            except: cal_offset = 120.0
            raw_db = db_fs + cal_offset

            # Corrections
            total_correction = np.zeros(len(raw_db))
            for name, f_arr, a_arr in self.correction_files:
                total_correction += np.interp(frequencies_hz, f_arr, a_arr)
            
            final_values = raw_db + total_correction
            
            # Load the locked Limits (from the Apply button)
            s_hz = self.active_s_hz
            e_hz = self.active_e_hz
            y_min = self.active_y_min
            y_max = self.active_y_max
            
            target_unit = self.unit_combo.get()

            mask = (frequencies_hz >= s_hz) & (frequencies_hz <= e_hz)
            plot_freqs = frequencies_hz[mask]
            plot_vals = final_values[mask]

            self.last_freqs = plot_freqs
            self.last_vals = plot_vals
            self.last_unit = target_unit

            # --- OPTIMIZED DRAWING LOGIC ---
            if self.trace_line is None:
                self.ax.clear()
                self.peak_annotations.clear() 
                
                # Setup Grid and Labels ONCE
                self.ax.set_xscale('log')
                formatter = ScalarFormatter()
                formatter.set_scientific(False)
                self.ax.xaxis.set_major_formatter(formatter)
                self.ax.yaxis.set_major_formatter(formatter)
                
                # Apply Limits
                self.ax.set_xlim(left=s_hz, right=e_hz)
                self.ax.set_ylim(bottom=y_min, top=y_max)
                
                self.ax.set_xlabel("Frequency (Hz)", fontweight='bold')
                self.ax.set_ylabel(f"Amplitude ({target_unit})", fontweight='bold')
                self.ax.grid(True, which="both", linestyle="--", alpha=0.5)

                # Draw Limit Lines ONCE
                if len(self.limit_points) >= 2:
                    try: margin_db = float(self.margin_entry.get())
                    except: margin_db = 6.0
                    lim_f = [p[0] for p in self.limit_points] 
                    lim_a = [p[1] for p in self.limit_points]
                    limit_curve = np.interp(np.log10(np.clip(plot_freqs, 1, None)), np.log10(np.clip(lim_f, 1, None)), lim_a)
                    self.limit_line, = self.ax.plot(plot_freqs, limit_curve, label="Limit", color='red', linewidth=2)
                    self.margin_line, = self.ax.plot(plot_freqs, limit_curve - margin_db, label="Margin", color='red', linestyle=':', linewidth=1.5)

                # Initialize Trace Line ONCE
                self.trace_line, = self.ax.plot(plot_freqs, plot_vals, label=f"Trace ({target_unit})", color='blue', alpha=0.8)
                self.ax.legend(loc="upper right")
            else:
                # FAST UPDATE: Just swap out the Y-data of the trace!
                self.trace_line.set_ydata(plot_vals)
                self.trace_line.set_xdata(plot_freqs) 

            # Peak Finding (Safely remove old markers)
            for ann in self.peak_annotations:
                try:
                    ann.remove() 
                except ValueError:
                    pass 
            self.peak_annotations.clear()
            
            try: num_markers = int(self.marker_entry.get())
            except: num_markers = 1
            if len(plot_vals) > 2:
                local_maxes = (plot_vals[1:-1] > plot_vals[:-2]) & (plot_vals[1:-1] > plot_vals[2:])
                peak_indices = np.where(local_maxes)[0] + 1
                if len(peak_indices) > 0:
                    top_peak_indices = peak_indices[np.argsort(plot_vals[peak_indices])][::-1][:num_markers]
                    colors = ['green', 'orange', 'purple']
                    for i, idx in enumerate(top_peak_indices):
                        color = colors[i % len(colors)]
                        pf, pa = plot_freqs[idx], plot_vals[idx]
                        disp_f = f"{pf:.1f} Hz" if pf < 1000 else f"{pf/1000:.2f} kHz"
                        
                        ann = self.ax.annotate(f'#{i+1}: {disp_f}\n{pa:.1f}', xy=(pf, pa), xytext=(pf, pa + 10),
                                         color=color, fontweight='bold', fontsize=10, arrowprops=dict(facecolor=color, shrink=0.05, width=1, headwidth=5))
                        self.peak_annotations.append(ann)
            
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"Sweep error: {e}")

    def single_sweep(self):
        if self.is_continuous:
            self.toggle_continuous()
            self.root.after(200, self._execute_single_sweep)
        else:
            self._execute_single_sweep()

    def _execute_single_sweep(self):
        self.apply_limits()
        self.fetch_and_plot(stream_mode=False)

    def toggle_continuous(self):
        if self.is_continuous:
            self.is_continuous = False
            self.cont_btn.config(text="Continuous Sweep", bg="#fff3cd")
            
            if self.loop_id:
                self.root.after_cancel(self.loop_id)
                self.loop_id = None
                
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
        else:
            self.is_continuous = True
            self.cont_btn.config(text="STOP Sweep", bg="#f8d7da")
            
            sr_string = self.sr_combo.get()
            active_sr = int(sr_string.split()[0])
            device_id = int(self.device_combo.get().split(':')[0])
            
            self.apply_limits() 
            
            try:
                self.stream = sd.InputStream(samplerate=active_sr, channels=1, device=device_id, callback=self.audio_callback)
                self.stream.start()
                self.continuous_loop()
            except sd.PortAudioError as e:
                messagebox.showerror("Audio Device Error", f"The audio device does not support a sample rate of {active_sr} Hz.\n\nDetails: {e}")
                self.is_continuous = False
                self.cont_btn.config(text="Continuous Sweep", bg="#fff3cd")

    def continuous_loop(self):
        if self.is_continuous and self.root.winfo_exists():
            self.fetch_and_plot(stream_mode=True)
            self.loop_id = self.root.after(80, self.continuous_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioEMCGUI(root)
    root.mainloop()