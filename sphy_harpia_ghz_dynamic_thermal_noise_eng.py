# ───────────────────────────────────────────────────────────────
# File: sphy_harpia_ghz_dynamic_thermal_noise_eng.py
# Author: deywe@QLZ + Symbiotic AI Compilation Llamma + GPT-4 Refactoring
# 💫 Quantum GHZ Simulation with UID under Vibrational Dynamics and Entropy
# ───────────────────────────────────────────────────────────────

import os
import csv
import re
import hashlib
import random
import subprocess
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import strawberryfields as sf
from strawberryfields.ops import Sgate, BSgate, MeasureFock


# ───────────────────────────────
# 📁 Log Directory
# ───────────────────────────────
LOG_DIR = "logs_ghz_sf_noise"
os.makedirs(LOG_DIR, exist_ok=True)


# ───────────────────────────────
# 📊 User Input
# ───────────────────────────────
def input_parameters():
    """Prompts the user for simulation parameters."""
    try:
        num_qubits = int(input("🔢 Modes (photonic qubits - GHZ): "))
        total_frames = int(input("🔁 Frames to simulate: "))
        apply_noise = input("🌡 Apply Thermal Noise? (y/n): ").lower() == 'y'
        return num_qubits, total_frames, apply_noise
    except ValueError:
        print("❌ Invalid input. Please enter integers.")
        exit(1)


# ───────────────────────────────
# 🤖 External Entanglement Boost : Quantum decoherence AI controller
# ───────────────────────────────
def compute_F_opt(H, S, C, I, T):
    """
    Calls an external binary to compute the optimization factor.
    Returns a float or 0.0 on failure.
    """
    try:
        result = subprocess.run(
            ["./sphy_simbiotic_entangle_ai", str(H), str(S), str(C), str(I), str(T)],
            capture_output=True, text=True, check=True
        )
        match = re.search(r"([-+]?\d*\.\d+|\d+)", result.stdout)
        return float(match.group(0)) if match else 0.0
    except Exception:
        return 0.0  # Fallback on any error


# ───────────────────────────────
# 🧪 Photonic GHZ Circuit
# ───────────────────────────────
def simulate_ghz_state(num_modes: int, cutoff: int = 2, backend_name: str = "fock", thermal_noise: bool = False) -> str:
    """
    Creates and runs a GHZ circuit, returning the measurement result as a string.
    """
    prog = sf.Program(num_modes)
    with prog.context as q:
        for i in range(num_modes):
            Sgate(1.0) | q[i]

        for i in range(num_modes - 1):
            BSgate(np.pi / 4, 0) | (q[0], q[i + 1])

        if thermal_noise:
            for i in range(num_modes):
                Sgate(np.random.normal(0, 1.0)) | q[i]

        if backend_name == "fock":
            MeasureFock() | q

    eng = sf.Engine(backend=backend_name, backend_options={"cutoff_dim": cutoff})
    try:
        result = eng.run(prog)
        return ''.join(map(str, result.samples[0]))
    except Exception:
        return "ERROR"


# ───────────────────────────────
# 🔐 UID Generation (Frame Hash)
# ───────────────────────────────
def generate_sha256_from_frame(frame_id, result, coherence, uid):
    """Generates a unique SHA256 hash for a simulation frame."""
    raw = f"{frame_id}_{result}_{coherence:.4f}_{uid}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ───────────────────────────────
# 🚀 Experiment Execution
# ───────────────────────────────
def run_simulation(num_modes, total=100, thermal_noise=False):
    """
    Main simulation loop.
    Generates GHZ states, applies an external boost, and logs the results.
    """
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(LOG_DIR, f"ghz_{num_modes}q_{now}.csv")
    img_file = os.path.join(LOG_DIR, f"ghz_{num_modes}q_plot_{now}.png")

    ideal_0, ideal_1 = '0' * num_modes, '1' * num_modes
    coherence = 90.0
    accepted = 0
    log_data, evolution = [], []
    current_cutoff = 1 if num_modes >= 10 else 2

    print("=" * 64)
    print(f"🧬 GHZ ∆(Φ)∇ • Modes: {num_modes} • Frames: {total:,} • Noise: {'ON' if thermal_noise else 'OFF'}")
    print("=" * 64)

    for frame in tqdm(range(1, total + 1), desc="⟳ Running"):
        result = simulate_ghz_state(num_modes, cutoff=current_cutoff, thermal_noise=thermal_noise)

        if result == "ERROR":
            is_valid = False
            uid = "SIM_ERROR"
            H = S = C = I = boost = "N/A"
        else:
            H = random.uniform(0.95, 1.0)
            S = random.uniform(0.94, 1.0)
            C = coherence / 100.0
            I = abs(H - S)
            boost = compute_F_opt(H, S, C, I, frame)
            delta = boost * 0.7
            coherence = min(100.0, coherence + delta)

            is_valid = result in [ideal_0, ideal_1] and delta > 0.3
            uid = hashlib.sha256(f"{result}_{frame}_{boost}".encode()).hexdigest()[:12] if is_valid else "REJECTED"

            if is_valid:
                accepted += 1

        sha256_hash = generate_sha256_from_frame(frame, result, coherence, uid)
        evolution.append(coherence)

        log_data.append([
            frame, result,
            round(H, 4) if isinstance(H, float) else H,
            round(S, 4) if isinstance(S, float) else S,
            round(C, 4) if isinstance(C, float) else C,
            round(I, 4) if isinstance(I, float) else I,
            round(boost, 4) if isinstance(boost, float) else boost,
            round(coherence, 4), uid, "✅" if is_valid else "❌", sha256_hash
        ])

    acc = 100 * accepted / total if total else 0
    print(f"\n✅ Accepted UIDs: {accepted}/{total} • Accuracy: {acc:.2f}%")

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Result", "H", "S", "C", "I", "Boost", "Coherence", "UID", "Valid", "SHA256"])
        writer.writerows(log_data)

    # 🔽 Plot coherence
    colors = ["green" if row[-2] == "✅" else "red" for row in log_data]
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, total + 1), evolution, label="SPHY Coherence (%)", color="blue", linewidth=1.5)
    plt.scatter(range(1, total + 1), evolution, c=colors, s=10, alpha=0.6)
    plt.axhline(90, color="gray", linestyle="--", linewidth=1, label="Baseline")
    plt.xlabel("Frame")
    plt.ylabel("Coherence (%)")
    plt.title(f"GHZ UID Evolution – {num_modes} modes")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(img_file)
    plt.show()

    print(f"📁 Results saved in:\n→ CSV: {csv_file}\n→ Plot: {img_file}")


# ───────────────────────────────
# 🔁 MAIN
# ───────────────────────────────
if __name__ == "__main__":
    modes, frames, noise = input_parameters()
    if modes > 13:
        print("⚠️ Recommended maximum: 13 modes.")
        modes = 13
    run_simulation(modes, frames, thermal_noise=noise)
