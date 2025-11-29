#%% ================== IMPORT LIBRARY ==================
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

#%% ================== KONEKSI KE COPPELIASIM ==================
print("Program Started")

# Buat client ZMQ dan ambil modul 'sim'
client = RemoteAPIClient()
sim = client.require('sim')

# Mode non-stepping: simulasi jalan bebas 
sim.setStepping(False)
sim.startSimulation()

#%% ================== AMBIL HANDLE OBJEK DI SCENE ==================
# Handle motor kanan & kiri (joint P3DX)
wR_Handle   = sim.getObject("/rightMotor")
wL_Handle   = sim.getObject("/leftMotor")

# (opsional) sensor ultrasonik
s3_Handle   = sim.getObject("/ultrasonicSensor[3]")

# Handle body robot, disc (orientasi target), path, dan marker visual
p3dx_Handle = sim.getObject("/PioneerP3DX")
disc_Handle = sim.getObject("/Disc")
path_Handle = sim.getObject("/Path")
LH_Handle   = sim.getObject("/LH")    # marker untuk look-ahead point
perp_Handle = sim.getObject("/Perp")  # marker untuk titik path yang dikejar


#%% ================== FUNGSI TRANSFORMASI 4x4 ==================
def transformMat(alpha, beta, gamma, tx, ty, tz):
    """
    Membangun matriks transformasi homogen 4x4 dari:
    - Rotasi Euler (alpha=X, beta=Y, gamma=Z)
    - Translasi (tx, ty, tz)
    """
    # Rotasi sekitar X
    rotx = np.array([
        [1, 0, 0],
        [0, math.cos(alpha), -math.sin(alpha)],
        [0, math.sin(alpha),  math.cos(alpha)]
    ])
    # Rotasi sekitar Y
    roty = np.array([
        [ math.cos(beta), 0, math.sin(beta)],
        [0, 1, 0],
        [-math.sin(beta), 0, math.cos(beta)]
    ])
    # Rotasi sekitar Z
    rotz = np.array([
        [math.cos(gamma), -math.sin(gamma), 0],
        [math.sin(gamma),  math.cos(gamma), 0],
        [0, 0, 1]
    ])

    # Total rotasi: R = R_x * R_y * R_z (konvensi yang dipakai di sini)
    rot_total = np.matmul(rotx, roty)
    rot_total = np.matmul(rot_total, rotz)

    # Vektor translasi (3x1)
    trans_vector = np.array([
        [tx],
        [ty],
        [tz]
    ])

    # Gabungkan jadi matriks 3x4: [R | t]
    R_t_3x4 = np.hstack((rot_total, trans_vector))

    # Tambahkan baris homogen [0 0 0 1]
    homogeneous_row = np.array([[0, 0, 0, 1]])

    # Matriks transformasi homogen 4x4
    transform_matrix_4x4 = np.vstack((R_t_3x4, homogeneous_row))
    return transform_matrix_4x4
    

#%% ================== INISIALISASI LOG DATA ==================
d_xyyaw      = []                    # menyimpan [x, y, yaw] robot di world
d_t          = []                    # menyimpan waktu
dat_disc2rob = np.zeros((4, 1))      # posisi target path di frame body (homogen)
dat_errors   = np.zeros((3, 1))      # [e_d; e_h; e_o]

t_prv = 0.0

sim.addLog(1, "get vel start")
time.sleep(2)          
start_time = time.time()


#%% ================== AMBIL DATA PATH DARI OBJEK /Path ==================
# format double table [x,y,z,qx,qy,qz,qw, x2,y2,z2,...]
pathBuf = sim.getBufferProperty(path_Handle, 'customData.PATH', {'noError': True})
if not pathBuf:
    raise RuntimeError(
        "Path tidak punya customData.PATH.\n"
        "Pastikan objek /Path benar-benar tipe Path (bukan dummy lama)."
    )

# Unpack ke list of float
pathData = sim.unpackDoubleTable(pathBuf)

# Bentuk jadi matriks N x 7, tiap baris = [x, y, z, qx, qy, qz, qw]
np_path = np.array(pathData).reshape(-1, 7)


#%% ================== LOOP UTAMA KONTROL ==================
while True:
    # waktu relatif sejak start
    t_now = time.time() - start_time
    # batasi durasi simulasi (misal 60 detik)
    if t_now > 30:
        break

    # -------- 1. AMBIL POSE ROBOT DI WORLD FRAME --------
    # Posisi: [x_w, y_w, z_w]
    bod_pos_xyz = sim.getObjectPosition(p3dx_Handle)
    # Orientasi: [alpha, beta, gamma] (roll, pitch, yaw)
    bod_pos_abg = sim.getObjectOrientation(p3dx_Handle)
    yaw = bod_pos_abg[2]  # yaw = rotasi di sekitar Z (heading)

    # -------- 2. HITUNG LOOK-AHEAD POINT DI DEPAN ROBOT --------
    # Look-ahead distance (L_H) 
    look_dist = 0.8   # meter

    # Rumus: titik L_H = posisi robot + L_H * arah-heading
    look_ahead_pt = np.array([
        bod_pos_xyz[0] + look_dist * math.cos(yaw),
        bod_pos_xyz[1] + look_dist * math.sin(yaw)
    ])

    # Tampilkan marker look-ahead di scene (objek /LH)
    sim.setObjectPosition(
        LH_Handle,
        [look_ahead_pt[0], look_ahead_pt[1], 0.14]  # tinggi 0.14 agar kelihatan
    )

    # -------- 3. CARI TITIK PATH PALING DEKAT KE LOOK-AHEAD --------
    # Ambil hanya koordinat (x, y) dari path
    np_path_xy = np_path[:, 0:2]   # shape: (N, 2)

    # Posisi relatif setiap titik path terhadap look-ahead:
    # r_i = [x_i - x_LH, y_i - y_LH]
    np_path_rel = np_path_xy - look_ahead_pt   # shape: (N, 2)

    # Jarak Euclidean masing-masing titik ke look-ahead
    # d_i = ||r_i||
    path_sse = np.linalg.norm(np_path_rel, axis=1)  # shape: (N,)

    # Cari index titik dengan jarak minimum (paling dekat)
    pendic_idx = int(np.argmin(path_sse))
    # (check keamanan – meski argmin pasti < len)
    if pendic_idx > len(path_sse) - 1:
        pendic_idx = 0

    # -------- 4. AMBIL POSE DISC (TARGET ORIENTASI) --------
    disc_pos_xyz = sim.getObjectPosition(disc_Handle)
    disc_pos_abg = sim.getObjectOrientation(disc_Handle)

    # -------- 5. BENTUK KOORDINAT HOMOGEN TITIK PATH TARGET (WORLD) --------
    # Titik path yang dikejar (pendic_idx) di world:
    path_pos_xyz_hom = np.array([
        [np_path[pendic_idx, 0]],  # x_w
        [np_path[pendic_idx, 1]],  # y_w
        [0],                       # z_w (anggap 0 di lantai)
        [1]                        # koordinat homogen
    ])

    # Tampilkan marker titik path yang sedang dikejar (objek /Perp)
    sim.setObjectPosition(
        perp_Handle,
        [np_path[pendic_idx, 0], np_path[pendic_idx, 1], 0.14]
    )

    # -------- 6. TRANSFORM TITIK PATH KE FRAME BODY ROBOT --------
    # Bangun transformasi ^wT_B (world <- body)
    path2body_mat = transformMat(
        0,                 # roll (abaikan)
        0,                 # pitch (abaikan)
        bod_pos_abg[2],    # yaw
        bod_pos_xyz[0],    # tx
        bod_pos_xyz[1],    # ty
        0                  # tz (anggap 0)
    )

    # Kita butuh ^B p_target = (^wT_B)^(-1) * ^w p_target
    path2body_pos = np.matmul(
        np.linalg.inv(path2body_mat),  # ^B T_w
        path_pos_xyz_hom               # ^w p_target
    )
    # Hasilnya shape (4,1): [x_B, y_B, z_B, 1]^T

    # Ambil x_B dan y_B sebagai skalar
    xB = float(path2body_pos[0, 0])
    yB = float(path2body_pos[1, 0])

    # -------- 7. HITUNG ERROR --------
    # e_d: seberapa jauh titik di depan robot (arah sumbu x_B)
    ed = xB

    # e_h: error heading ke titik (arah belok untuk mengarah ke titik)
    eh = math.atan2(yB, xB)   # >0 berarti titik ada di "kiri" robot

    # e_o: error orientasi terhadap disc (yaw target - yaw robot)
    eo = disc_pos_abg[2] - bod_pos_abg[2]

    # Jarak absolut titik target di frame body
    abs_d = math.sqrt(xB**2 + yB**2)

    # Susun vector error untuk logging
    errors = np.vstack(([ed], [eh], [eo]))

    # -------- 8. KONTROLER (V, W) + BLENDING ORIENTASI --------
    # Parameter kinematika
    rw = 0.195 / 2   # wheel radius (m)
    rb = 0.381 / 2   # half wheelbase (m) = jarak dari center ke roda

    # Parameter blending berdasarkan jarak |d|
    d      = 0.05
    mode   = math.exp(-abs_d / d)   # mode ∈ (0,1), makin dekat → makin besar
    kp_lin = 1.2
    kp_ang = 1 * (1 - mode)       # fokus ke e_h kalau masih jauh
    kp_ori = 10.0 * mode             # fokus ke e_o kalau sudah dekat

    # Kecepatan linear & angular
    # v: gerak maju, w: kecepatan rotasi (yaw rate)
    v = kp_lin * ed
    w = kp_ang * eh + kp_ori * eo

    # Hubungan differential drive:
    kin_mat = np.array([
        [1,  rb],
        [1, -rb]
    ])
    vel_vec = np.vstack(([v], [w]))  # [v; w]^T

    # Hitung kecepatan linear roda kanan & kiri
    v_rl = np.matmul(kin_mat, vel_vec)
    v_R = float(v_rl[0, 0])
    v_L = float(v_rl[1, 0])

    # -------- 9. KONVERSI KE KECEPATAN SUDUT & KIRIM KE MOTOR --------
    # Joint target velocity = kecepatan sudut roda = v / rw
    sim.setJointTargetVelocity(wR_Handle, v_R / rw)
    sim.setJointTargetVelocity(wL_Handle, v_L / rw)

    # -------- 10. LOG DATA UNTUK PLOTTING --------
    dat_disc2rob = np.hstack((dat_disc2rob, path2body_pos))
    dat_errors   = np.hstack((dat_errors, errors))

    d_xyyaw.append([
        bod_pos_xyz[0],     # x_w
        bod_pos_xyz[1],     # y_w
        bod_pos_abg[2]      # yaw
    ])
    d_t.append(t_now)

    t_prv = t_now

    # Log singkat ke status bar CoppeliaSim
    sim.addLog(
        1,
        f"ed, eh, mode, t="
        f"{ed:.2f}m, "
        f"{np.rad2deg(eh):.2f}deg, "
        f"{mode:.2f}, "
        f"{t_now:.2f}s"
    )

#%% ================== MATIKAN ROBOT & SIMULASI SELESAI ==================
sim.setJointTargetVelocity(wR_Handle, 0)
sim.setJointTargetVelocity(wL_Handle, 0)
sim.addLog(1, "sim com ended")

#%% ================== OLAH DATA & NORMALISASI YAW ==================
dat_xyyaw    = np.array(d_xyyaw)
dat_t        = np.array(d_t)
dat_disc2rob = dat_disc2rob[:, 1:]   # buang kolom pertama (nol)
dat_errors   = dat_errors[:, 1:]     # buang kolom pertama (nol)

# Wrap yaw ke [-pi, pi] biar grafik rapi
dat_xyyaw[:, 2] = np.atan2(
    np.sin(dat_xyyaw[:, 2]),
    np.cos(dat_xyyaw[:, 2])
)

#%% ================== PLOTTING TRAJEKTORI XY ==================
plt.figure(figsize=(8, 6))
plt.plot(dat_xyyaw[:, 0], dat_xyyaw[:, 1],
         linewidth=2, label='$^wB$ (trajektori robot)')
plt.scatter(dat_xyyaw[0, 0], dat_xyyaw[0, 1],
            marker='o', s=100, color='red', label='Start')
plt.scatter(dat_xyyaw[-1, 0], dat_xyyaw[-1, 1],
            marker='x', s=100, color='green', label='End')
plt.xlabel('$x_w$ (m)', fontsize=12)
plt.ylabel('$y_w$ (m)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Simpan ke file SVG dengan timestamp
now = datetime.now()
filename = now.strftime("%y%m%d%H%M_xy_track") + ".svg"
plt.savefig(filename, format='svg')
print(f"Plot saved successfully as '{filename}'")

#%% ================== PLOTTING ERROR e_d TERHADAP WAKTU ==================
plt.figure(figsize=(8, 6))
plt.plot(dat_t, dat_errors[0],
         linewidth=2, label='$e_d$ (error jarak)')
plt.xlabel('$t$ (sec)', fontsize=12)
plt.ylabel('$e_d$ (m)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

now = datetime.now()
filename = now.strftime("%y%m%d%H%M_ed_track") + ".svg"
plt.savefig(filename, format='svg')
print(f"Plot saved successfully as '{filename}'")

#%% ================== PLOTTING ERROR e_h TERHADAP WAKTU ==================
plt.figure(figsize=(8, 6))
plt.plot(dat_t, np.rad2deg(dat_errors[1]),
         linewidth=2, label='$e_h$ (error heading)')
plt.xlabel('$t$ (sec)', fontsize=12)
plt.ylabel('$e_h$ (deg)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

now = datetime.now()
filename = now.strftime("%y%m%d%H%M_eh_track") + ".svg"
plt.savefig(filename, format='svg')
print(f"Plot saved successfully as '{filename}'")
