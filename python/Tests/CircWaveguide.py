# -*- coding: utf-8 -*-
"""
 Circular Waveguide TE11 Test

 Verifies CircWGPort using two facing TE11 ports in a circular waveguide
 (radius 10 mm, TE11 cutoff ≈ 8.79 GHz) operating from 10 to 14 GHz.
 Boundary: PEC cylindrical shell at r=R, PML at z-faces.

 Pass criteria:
   max(dB(S11)) < -25 dB   (low reflection)
   min(dB(S21)) > -0.1 dB  (near-lossless transmission)
   max(dB(S21)) < +0.01 dB (sign-error guard)

 Tested with
  - python 3.14
  - openEMS v0.0.36+

 (c) 2026 Thorsten Liebig <thorsten.liebig@gmx.de>

"""

import os, tempfile
import numpy as np

from CSXCAD  import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0
from openEMS.ports import CircWGPort
from openEMS.utilities import check_mode_purity

### Setup the simulation
Sim_Path = os.path.join(tempfile.gettempdir(), 'CircWG')

unit   = 1e-3  # drawing unit: mm
R      = 10    # waveguide radius (mm) → TE11 cutoff = 1.841·c/(2π·R) ≈ 8.79 GHz
length = 50    # waveguide length (mm)

f_start = 10e9
f_0     = 12e9
f_stop  = 14e9
lambda0 = C0 / f_0 / unit   # free-space wavelength at f_0 in mm

mesh_res = 0.5

### FDTD & excitation
FDTD = openEMS(NrTS=1e4)
FDTD.SetGaussExcite(0.5*(f_start + f_stop), 0.5*(f_stop - f_start))
FDTD.SetBoundaryCond(['PEC', 'PEC', 'PEC', 'PEC', 'PML_8', 'PML_8'])  # PML at z-faces

### CSXCAD geometry & mesh
CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)

mesh.AddLine('x', [-R, R])
mesh.AddLine('y', [-R, R])
mesh.AddLine('z', [0, length])

### Ports — pin the port planes to the mesh before smoothing
ports = []

start = [-R, -R, 10*mesh_res]
stop  = [ R,  R, 15*mesh_res]
mesh.AddLine('z', [start[2], stop[2]])
ports.append(CircWGPort(CSX, 1, start, stop, 'z', R*unit, 'TE11', excite=1))

start = [-R, -R, length - 10*mesh_res]
stop  = [ R,  R, length - 15*mesh_res]
mesh.AddLine('z', [start[2], stop[2]])
ports.append(CircWGPort(CSX, 2, start, stop, 'z', R*unit, 'TE11'))

### Outer PEC wall
pec = CSX.AddMetal('WG_Wall')
wall_thickness = 2*mesh_res
pec.AddCylindricalShell([0, 0, 0], [0, 0, length], R+wall_thickness/2, wall_thickness)

mesh.SmoothMeshLines('all', mesh_res, ratio=1.4)

if 0:  # debugging only
    CSX_file = os.path.join(Sim_Path, 'circ_wg.xml')
    if not os.path.exists(Sim_Path):
        os.mkdir(Sim_Path)
    CSX.Write2XML(CSX_file)
    from CSXCAD import AppCSXCAD_BIN
    os.system(AppCSXCAD_BIN + ' "{}"'.format(CSX_file))

### Run
FDTD.Run(Sim_Path, cleanup=True)

### Post-processing
freq = np.linspace(f_start, f_stop, 201)
for port in ports:
    port.CalcPort(Sim_Path, freq)

s11 = ports[0].uf_ref / ports[0].uf_inc
s21 = ports[1].uf_ref / ports[0].uf_inc

s11_dB = 20 * np.log10(np.abs(s11))
s21_dB = 20 * np.log10(np.abs(s21))

### Mode purity check
for port in ports:
    lbl = 'port {}'.format(port.number)
    check_mode_purity(lbl + ' U', port.u_data.ui_val[0], port.u_mode_purity[0], threshold=0.97, sig_frac=0.05)
    check_mode_purity(lbl + ' I', port.i_data.ui_val[0], port.i_mode_purity[0], threshold=0.97, sig_frac=0.05)

### Pass / fail checks
print(f'max(dB(S11)) = {np.max(s11_dB):.1f} dB')
print(f'min(dB(S21)) = {np.min(s21_dB):.1f} dB,  max(dB(S21)) = {np.max(s21_dB):.2f} dB')

assert np.max(s11_dB) < -25, \
    f'FAIL: max(dB(S11)) = {np.max(s11_dB):.1f} dB, expected < -25 dB'
assert np.min(s21_dB) > -0.1, \
    f'FAIL: min(dB(S21)) = {np.min(s21_dB):.1f} dB, expected > -0.1 dB'
assert np.max(s21_dB) < 0.01, \
    f'FAIL: max(dB(S21)) = {np.max(s21_dB):.2f} dB, expected < +0.01 dB (sign error?)'

print('PASS')

if 0:  # set to 1 for debugging plots
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(num='S-Parameters', tight_layout=True)
    axis.plot(freq/1e9, s11_dB, 'k-',  linewidth=2, label='$S_{11}$')
    axis.plot(freq/1e9, s21_dB, 'r--', linewidth=2, label='$S_{21}$')
    axis.grid()
    axis.set_xmargin(0)
    axis.set_xlabel('Frequency (GHz)')
    axis.set_ylabel('S-Parameter (dB)')
    axis.legend()

    fig, axes = plt.subplots(len(ports), 1, num='Mode Purity', tight_layout=True,
                             sharex=True)
    for ax, port in zip(axes, ports):
        t_u = port.u_data.ui_time[0] * 1e9  # s → ns
        t_i = port.i_data.ui_time[0] * 1e9
        if port.u_mode_purity[0] is not None:
            ax.plot(t_u, 100*np.abs(port.u_mode_purity[0]), 'k-',  linewidth=1.5, label='U purity')
        if port.i_mode_purity[0] is not None:
            ax.plot(t_i, 100*np.abs(port.i_mode_purity[0]), 'r--', linewidth=1.5, label='I purity')
        ax.axhline(99, color='gray', linestyle=':', linewidth=1, label='99 % threshold')
        ax.set_title('port {}'.format(port.number))
        ax.set_ylabel('Mode purity (%)')
        ax.set_ylim([0, 105])
        ax.set_xmargin(0)
        ax.grid()
        ax.legend()
    axes[-1].set_xlabel('Time (ns)')

    plt.show()
