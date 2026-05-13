"""
Rectangular waveguide (WR-42) with generic WaveguidePort using HDF5 mode files.

This is the test counterpart of Tutorials/Rect_Waveguide.py.  Instead of
AddRectWaveGuidePort (which generates the analytic mode functions internally),
it writes the identical TE10 field distributions to HDF5 mode files and uses
the generic AddWaveGuidePort.  Results should be indistinguishable.

Assertions:
  - S11 < -40 dB across the passband
  - S21 > -0.1  dB across the passband
  - mode purity > 99 % at every time step where the recorded signal exceeds
    1 % of its peak amplitude (checked for both U and I probes of both ports)

(c) 2026 Thorsten Liebig <thorsten.liebig@gmx.de>
"""

import os
import tempfile
import numpy as np
import h5py

from CSXCAD  import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0


def make_rect_wg_te10_hdf5(filename, field_type, a_draw, b_draw, N=60):
    """Write a TE10 rectangular waveguide mode profile to an HDF5 file.

    Parameters
    ----------
    filename : str
        Output HDF5 file path.
    field_type : 'E' or 'H'
        Which field to store.
    a_draw, b_draw : float
        Waveguide width / height in drawing units.
    N : int
        Number of grid points along the width (height scaled proportionally).
    """
    x = np.linspace(0, a_draw, N)
    y = np.linspace(0, b_draw, max(int(N * b_draw / a_draw), 2))
    XX, _ = np.meshgrid(x, y, indexing='ij')

    if field_type == 'E':
        # TE10: E_x = 0,  E_y = -sin(pi*x/a) / a
        Vx = np.zeros_like(XX)
        Vy = -np.sin(np.pi * XX / a_draw) / a_draw
    else:
        # TE10: H_x = sin(pi*x/a) / a,  H_y = 0
        Vx = np.sin(np.pi * XX / a_draw) / a_draw
        Vy = np.zeros_like(XX)

    with h5py.File(filename, 'w') as f:
        f.attrs['Version'] = 1.0
        f.create_dataset('x',  data=x)
        f.create_dataset('y',  data=y)
        f.create_dataset('Vx', data=Vx)
        f.create_dataset('Vy', data=Vy)


def check_mode_purity(label, signal, purity, threshold=0.99, sig_frac=0.001):
    """Assert mode purity > threshold where the signal is above sig_frac * peak.

    Parameters
    ----------
    label : str
        Descriptive name for the assertion message.
    signal : array
        Time-domain signal amplitude (column 1 of probe file).
    purity : array or None
        Mode purity values (column 2 of probe file), or None if not available.
    threshold : float
        Minimum acceptable mode purity (default 0.99 = 99 %).
    sig_frac : float
        Ignore time steps where |signal| < sig_frac * max(|signal|).

    Notes
    -----
    Purity can be negative when the wave travels in the opposite direction
    (e.g. the receive port seeing the transmitted wave), so |purity| is used.
    """
    if purity is None:
        return
    mask = np.abs(signal) >= sig_frac * np.max(np.abs(signal))
    if not np.any(mask):
        return
    min_purity = np.min(np.abs(purity[mask]))
    print('{}: min mode purity = {:.1f}% ({:.1f}% of samples considered)'.format(
        label, 100*min_purity, 100*np.sum(mask)/len(signal)))
    assert min_purity >= threshold, \
        '{}: mode purity {:.1f}% below {:.0f}% threshold'.format(
            label, 100*min_purity, 100*threshold)


# ---------------------------------------------------------------------------
# Geometry & simulation parameters  (WR-42, identical to tutorial)
# ---------------------------------------------------------------------------
unit   = 1e-6       # drawing unit: 1 µm
a      = 10700      # waveguide width  (µm)
b      = 4300       # waveguide height (µm)
length = 50000      # waveguide length (µm)

f_start = 20e9
f_0     = 24e9
f_stop  = 26e9
lambda0 = C0 / f_0 / unit

mesh_res = lambda0 / 30

kc = np.pi / (a * unit)     # TE10 cutoff wavenumber (rad/m)

# ---------------------------------------------------------------------------
# Paths & mode files
# ---------------------------------------------------------------------------
Sim_Path  = os.path.join(tempfile.gettempdir(), 'Rect_WG_HDF5')

# Mode files live outside Sim_Path — cleanup=True would delete them otherwise
Mode_Path = os.path.join(tempfile.gettempdir(), 'Rect_WG_HDF5_modes')
os.makedirs(Mode_Path, exist_ok=True)

E_file = os.path.join(Mode_Path, 'TE10_E.h5')
H_file = os.path.join(Mode_Path, 'TE10_H.h5')

make_rect_wg_te10_hdf5(E_file, 'E', a, b)
make_rect_wg_te10_hdf5(H_file, 'H', a, b)

# ---------------------------------------------------------------------------
# FDTD & CSX setup
# ---------------------------------------------------------------------------
FDTD = openEMS(NrTS=1e4)
FDTD.SetGaussExcite(0.5*(f_start+f_stop), 0.5*(f_stop-f_start))
FDTD.SetBoundaryCond([0, 0, 0, 0, 3, 3])

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)

mesh.AddLine('x', [0, a])
mesh.AddLine('y', [0, b])
mesh.AddLine('z', [0, length])

# ---------------------------------------------------------------------------
# Waveguide ports using HDF5 mode files, local_origin='corner' so the mode
# file grid (defined from x=0,y=0) aligns with the lower-left port corner
# ---------------------------------------------------------------------------
ports = []

start = [0, 0, 10*mesh_res]
stop  = [a, b, 15*mesh_res]
mesh.AddLine('z', [start[2], stop[2]])
ports.append(FDTD.AddWaveGuidePort(1, start, stop, 'z',
                                    E_file=E_file, H_file=H_file,
                                    kc=kc, excite=1, local_origin='corner'))

start = [0, 0, length - 10*mesh_res]
stop  = [a, b, length - 15*mesh_res]
mesh.AddLine('z', [start[2], stop[2]])
ports.append(FDTD.AddWaveGuidePort(2, start, stop, 'z',
                                    E_file=E_file, H_file=H_file,
                                    kc=kc, excite=0, local_origin='corner'))

mesh.SmoothMeshLines('all', mesh_res, ratio=1.4)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
FDTD.Run(Sim_Path, cleanup=True)

# ---------------------------------------------------------------------------
# S-parameter check (frequency domain) — also populates u/i_mode_purity
# ---------------------------------------------------------------------------
freq = np.linspace(f_start, f_stop, 201)
for port in ports:
    port.CalcPort(Sim_Path, freq)

# ---------------------------------------------------------------------------
# Mode purity check (data already read by CalcPort)
# ---------------------------------------------------------------------------
for port in ports:
    lbl = 'port {}'.format(port.number)
    check_mode_purity(lbl + ' U', port.u_data.ui_val[0], port.u_mode_purity[0])
    check_mode_purity(lbl + ' I', port.i_data.ui_val[0], port.i_mode_purity[0])

s11 = ports[0].uf_ref / ports[0].uf_inc
s21 = ports[1].uf_ref / ports[0].uf_inc

s11_dB = 20 * np.log10(np.abs(s11))
s21_dB = 20 * np.log10(np.abs(s21))

print('S11 max = {:.1f} dB'.format(np.max(s11_dB)))
print('S21 min = {:.1f} dB'.format(np.min(s21_dB)))

assert np.max(s11_dB) < -40, 'S11 too high: {:.1f} dB'.format(np.max(s11_dB))
assert np.min(s21_dB) > -0.1,  'S21 too low:  {:.1f} dB'.format(np.min(s21_dB))

print('PASS')
