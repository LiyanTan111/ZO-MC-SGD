"""Sanity check: verify numpy + PySpice + ngspice shared library work end-to-end."""
import sys

import numpy as np


def check_imports():
    print(f"numpy {np.__version__} OK")
    try:
        import scipy

        print(f"scipy {scipy.__version__} OK")
    except ImportError as e:
        print(f"scipy MISSING: {e}")
    try:
        import matplotlib

        print(f"matplotlib {matplotlib.__version__} OK")
    except ImportError as e:
        print(f"matplotlib MISSING: {e}")
    try:
        import chaospy

        print(f"chaospy {chaospy.__version__} OK")
    except ImportError as e:
        print(f"chaospy MISSING: {e}")


def check_pyspice():
    try:
        import PySpice
        from PySpice.Spice.Netlist import Circuit
        from PySpice.Unit import u_V, u_kOhm

        print(f"PySpice {PySpice.__version__} OK")
    except ImportError as e:
        print(f"PySpice MISSING: {e}")
        return False

    try:
        from PySpice.Spice.NgSpice.Shared import NgSpiceShared

        ng = NgSpiceShared.new_instance()
        print(f"ngspice shared library OK (version line: {ng.exec_command('version -f')[0][:80]})")
    except Exception as e:
        print(f"ngspice shared library FAILED: {e}")
        return False

    # Smallest possible netlist: V source + R, DC operating point
    try:
        circuit = Circuit("smoke")
        circuit.V("in", "in", circuit.gnd, 5 @ u_V)
        circuit.R(1, "in", circuit.gnd, 1 @ u_kOhm)
        sim = circuit.simulator()
        analysis = sim.operating_point()
        # access the 'in' node via the nodes dict (avoid Python keyword collision)
        v = float(analysis.nodes["in"][0])
        print(f"DC OP smoke test: V(in) = {v:.6f} V (expected 5.0)")
        if abs(v - 5.0) > 1e-3:
            return False
    except Exception as e:
        print(f"DC OP smoke test FAILED: {e}")
        return False

    return True


if __name__ == "__main__":
    check_imports()
    ok = check_pyspice()
    sys.exit(0 if ok else 1)
