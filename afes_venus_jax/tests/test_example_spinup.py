import afes_venus_jax.examples.t42l60_venus_dry_spinup as spinup
import json
import os
import subprocess
import sys


def _spinup_env():
    env = os.environ.copy()
    env.update(
        {
            "AFES_VENUS_JAX_LMAX": "10",
            "AFES_VENUS_JAX_NLAT": "32",
            "AFES_VENUS_JAX_NLON": "64",
            "AFES_VENUS_JAX_L": "10",
            "AFES_VENUS_JAX_FAST_TESTS": "1",
            "AFES_VENUS_JAX_USE_S2FFT": "0",
            "AFES_VENUS_JAX_ENABLE_X64": "True",
            # Allow JIT for performance in this integration test.
            "JAX_DISABLE_JIT": "0",
        }
    )
    return env


def test_initial_condition_passes_sanity_check():
    mstate = spinup.initial_condition()
    spinup.sanity_check_initial_condition(mstate)


def test_spinup_runs_ten_steps_without_instability(tmp_path):
    script = f"""
import json
import os
import jax
jax.config.update('jax_disable_jit', False)
import afes_venus_jax.examples.t42l60_venus_dry_spinup as spinup

os.chdir(r"{tmp_path}")
state = spinup.run_t42l60_venus_spinup(nsteps=10, save_snapshots=False)
diag = spinup._diagnostics(state)
print(json.dumps(diag))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=_spinup_env(),
    )

    diag = json.loads(result.stdout.strip().splitlines()[-1])
    assert diag["max_u"] < 150.0
    assert diag["max_v"] < 1.0
    assert diag["max_T_prime"] < 1.0
    ps_min, ps_max = diag["ps_range"]
    assert (ps_max - ps_min) < 1.0e4
