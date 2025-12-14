# boot/qosgenesis.py 
#by Deywe Okabe

from boot.stages.qorigin_stage import initialize_origin
from boot.stages.simulation_stage import run_simulation
from boot.stages.qfs_stage import activate_quantum_fs
from boot.stages.memory_stage import regenerate_memory
from boot.stages.aquanet_stage import initialize_aquanet
from boot.stages.terminal_stage import start_shell
from boot.qos0.qos0_pipeline import start_qos0_stage

import time

class QOSGenesis:
    def __init__(self):
        # Identificador de inicialização do kernel quântico
        self.qkernel_ativo = False

        print("🛸 Iniciando núcleo QOSGenesis...")

        try:
            # 👁️ Tentativa simbiótica de inicializar QiskitContexto
            from qfs_e.qiskit.contexto_quantico import QiskitContexto
            self.qkernel = QiskitContexto()
            self.qkernel_ativo = True
            print("✅ QiskitContexto inicializado com sucesso no boot.")
        except Exception as e:
            print("⚠️ Modo simbiótico normal ativado (Qiskit indisponível):", str(e))
            self.qkernel = None

        # Apenas identificação em log para outros núcleos
        if self.qkernel_ativo:
            self.qstatus = "🧠 QKernel ATIVO: Simulações Quânticas disponíveis."
        else:
            self.qstatus = "🧘‍♂️ QKernel INATIVO: Operação simbiótica convencional."

        # Verifica ambiente sem display gráfico e ajusta matplotlib (modo headless seguro)
        try:
            import matplotlib
            matplotlib.use('Agg')
            print("🖼️ Detecção: Ambiente sem GUI usando backend 'Agg'")
        except Exception:
            pass

        # Mostrar mensagem se Qiskit estiver operacional
        if self.qkernel_ativo:
            print("⚛️ Qiskit detectado! Ativando ponte quântica...")

            # Criar um circuito simples ilustrativo
            from qiskit import QuantumCircuit
            resultado = None

            try:
                qc = QuantumCircuit(1)
                qc.h(0)
                qc.measure_all()

                resultado = self.qkernel.simular_circuito(qc, shots=1024)
                print(f"🧪 Resultado Qiskit: {resultado}")
            except Exception as e:
                print(f"⚠️ QiskitContexto não conseguiu simular circuito de teste: {e}")

            print("🌀 Injetando fase externa (A = +0.300) no QOrigin...\n")

    def run_all(self):
        print(f"\n🚦 Status do QKernel: {self.qstatus}\n")
        time.sleep(0.5)

        initialize_origin()
        run_simulation()
        activate_quantum_fs()
        regenerate_memory()
        initialize_aquanet()
        start_qos0_stage()
        start_shell()

    def only_boot_shell(self):
        print(f"\n🔹 Rodando terminal direto – {self.qstatus}")
        start_shell()

# Execução direta (modo standalone)
if __name__ == "__main__":
    system = QOSGenesis()
    system.run_all()
