import os
import re
import pandas as pd
import numpy as np

# === Utilities ===

def parse_length_and_axis(header):
    pattern = r'"[lL]"\s*([\d.eE+\-]+)\)\s*([XY])'
    match = re.search(pattern, str(header))
    if not match:
        raise ValueError(f"Could not parse column header: {header}")
    return float(match.group(1)), match.group(2)

def read_techplot_csv(path):
    df = pd.read_csv(path)
    length_map = {}
    for col in df.columns:
        try:
            L, axis = parse_length_and_axis(col)
            length_map.setdefault(L, {})[axis] = col
        except:
            continue
    return df, length_map

def find_closest_entry(df, length_map, L, gm_id_target):
    col_x = length_map[L]['X']
    col_y = length_map[L]['Y']
    gm_id = pd.to_numeric(df[col_x], errors='coerce')
    y_val = pd.to_numeric(df[col_y], errors='coerce')
    valid = gm_id[gm_id >= gm_id_target]
    if valid.empty:
        raise ValueError(f"No gm/Id ≥ {gm_id_target} for length {L}")
    idx = (valid - gm_id_target).abs().idxmin()
    return idx, gm_id[idx], y_val[idx]

def calculate_stage_gains(total_gain_db, gmro3_val):
    gain_total = 10 ** (total_gain_db / 20)
    gain12 = gain_total / gmro3_val
    stage1 = 2 * np.sqrt(gain12)
    stage2 = stage1 / 4
    gmro1 = stage1 * 2
    return gain12, stage1, stage2, gmro1

def round_gate_voltage(v, mode='down'):
    steps = np.arange(0.1, 1.01, 0.1)
    if mode == 'down':
        return steps[steps <= v].max()
    else:
        return steps[steps >= v].min()

def compute_vov(gm_id): return 2 / gm_id

def compute_gmn_rop(Id, gm_id_n, gm_id_p, stage2_gain):
    gmn = gm_id_n * Id
    gmp = gm_id_p * Id
    rop = stage2_gain / gmn
    return gmn, gmp, rop

# === PMOS Current Mirror Design ===

def extract_pmos_current_mirror_params(VDS, gm_id_target, Id_pmos):
    vds_folder = f"PMOS_{VDS:.1f}"
    base_path = os.path.join("D:/Desktop/DMC/PMOS_1V", vds_folder)
    idw_path = os.path.join(base_path, "idw.csv")

    if not os.path.exists(idw_path):
        return None

    try:
        df_idw = pd.read_csv(idw_path)
    except FileNotFoundError:
        return None

    lengths = []
    x_columns = [col for col in df_idw.columns if col.endswith('X')]
    for col in x_columns:
        length = parse_length_and_axis(col)[0]
        if length is not None and length not in lengths:
            lengths.append(length)

    if not lengths:
        return None

    # Find length closest to 2 µm (2e-6 meters)
    target_length = 2e-6
    best_length = min(lengths, key=lambda x: abs(x - target_length))
    
    x_col_idw = next(col for col in df_idw.columns if col.endswith('X') and str(best_length) in col)
    y_col_idw = next(col for col in df_idw.columns if col.endswith('Y') and str(best_length) in col)
    
    idw_data = df_idw[[x_col_idw, y_col_idw]].dropna()
    idw_data.columns = ['X', 'Y']
    closest_gm_id_row = idw_data.iloc[(idw_data['X'] - gm_id_target).abs().idxmin()]
    idw_value = closest_gm_id_row['Y']
    w = Id_pmos / idw_value

    return dict(
        L=best_length,
        W=w,
        idw=idw_value
    )

# === NMOS Current Mirror Design ===

def extract_nmos_current_mirror_params(VDS, gm_id_target, Id_nmos):
    vds_folder = f"nmos_{VDS:.1f}"
    base_path = os.path.join("D:/Desktop/DMC/NMOS_1V", vds_folder)
    idw_path = os.path.join(base_path, "idw.csv")

    if not os.path.exists(idw_path):
        return None

    try:
        df_idw = pd.read_csv(idw_path)
    except FileNotFoundError:
        return None

    lengths = []
    x_columns = [col for col in df_idw.columns if col.endswith('X')]
    for col in x_columns:
        length = parse_length_and_axis(col)[0]
        if length is not None and length not in lengths:
            lengths.append(length)

    if not lengths:
        return None

    # Find length closest to 2 µm (2e-6 meters)
    target_length = 2e-6
    best_length = min(lengths, key=lambda x: abs(x - target_length))
    
    x_col_idw = next(col for col in df_idw.columns if col.endswith('X') and str(best_length) in col)
    y_col_idw = next(col for col in df_idw.columns if col.endswith('Y') and str(best_length) in col)
    
    idw_data = df_idw[[x_col_idw, y_col_idw]].dropna()
    idw_data.columns = ['X', 'Y']
    closest_gm_id_row = idw_data.iloc[(idw_data['X'] - gm_id_target).abs().idxmin()]
    idw_value = closest_gm_id_row['Y']
    w = Id_nmos / idw_value

    return dict(
        L=best_length,
        W=w,
        idw=idw_value
    )

# === PMOS First Stage Design ===

def extract_pmos1_params(VDS, gm_id_target, Id_pmos, gmro_target):
    vds_folder = f"PMOS_{VDS:.1f}"
    base_path = os.path.join("D:/Desktop/DMC/PMOS_1V", vds_folder)
    gmro_path = os.path.join(base_path, "gmro.csv")
    idw_path = os.path.join(base_path, "idw.csv")

    if not os.path.exists(gmro_path) or not os.path.exists(idw_path):
        return None

    try:
        df_gmro = pd.read_csv(gmro_path)
        df_idw = pd.read_csv(idw_path)
    except FileNotFoundError:
        return None

    lengths = []
    x_columns = [col for col in df_gmro.columns if col.endswith('X')]
    for col in x_columns:
        length = parse_length_and_axis(col)[0]
        if length is not None and length not in lengths:
            lengths.append(length)

    lengths.sort()
    best_length = None
    min_gmro_diff = float('inf')
    best_gmro = None

    for length in lengths:
        x_col = next(col for col in df_gmro.columns if col.endswith('X') and str(length) in col)
        y_col = next(col for col in df_gmro.columns if col.endswith('Y') and str(length) in col)
        length_data = df_gmro[[x_col, y_col]].dropna()
        length_data.columns = ['X', 'Y']
        closest_gm_id_row = length_data.iloc[(length_data['X'] - gm_id_target).abs().idxmin()]
        gmro_value = closest_gm_id_row['Y']
        gmro_diff = abs(gmro_value - gmro_target)
        if gmro_diff < min_gmro_diff:
            min_gmro_diff = gmro_diff
            best_length = length
            best_gmro = gmro_value

    if best_length is None:
        return None

    x_col_idw = next(col for col in df_idw.columns if col.endswith('X') and str(best_length) in col)
    y_col_idw = next(col for col in df_idw.columns if col.endswith('Y') and str(best_length) in col)
    idw_data = df_idw[[x_col_idw, y_col_idw]].dropna()
    idw_data.columns = ['X', 'Y']
    closest_gm_id_row = idw_data.iloc[(idw_data['X'] - gm_id_target).abs().idxmin()]
    idw_value = closest_gm_id_row['Y']
    w = Id_pmos / idw_value

    return dict(
        gmro=best_gmro,
        L=best_length,
        W=w
    )

# === NMOS First Stage Design ===

def extract_nmos1_params(VDS, gm_id_target, Id_nmos, gmro_target):
    vds_folder = f"nmos_{VDS:.1f}"
    base_path = os.path.join("D:/Desktop/DMC/NMOS_1V", vds_folder)
    gmro_path = os.path.join(base_path, "gmro.csv")
    idw_path = os.path.join(base_path, "idw.csv")

    if not os.path.exists(gmro_path) or not os.path.exists(idw_path):
        return None

    try:
        df_gmro = pd.read_csv(gmro_path)
        df_idw = pd.read_csv(idw_path)
    except FileNotFoundError:
        return None

    lengths = []
    x_columns = [col for col in df_gmro.columns if col.endswith('X')]
    for col in x_columns:
        length = parse_length_and_axis(col)[0]
        if length is not None and length not in lengths:
            lengths.append(length)

    lengths.sort()
    best_length = None
    min_gmro_diff = float('inf')
    best_gmro = None

    for length in lengths:
        x_col = next(col for col in df_gmro.columns if col.endswith('X') and str(length) in col)
        y_col = next(col for col in df_gmro.columns if col.endswith('Y') and str(length) in col)
        length_data = df_gmro[[x_col, y_col]].dropna()
        length_data.columns = ['X', 'Y']
        closest_gm_id_row = length_data.iloc[(length_data['X'] - gm_id_target).abs().idxmin()]
        gmro_value = closest_gm_id_row['Y']
        gmro_diff = abs(gmro_value - gmro_target)
        if gmro_diff < min_gmro_diff:
            min_gmro_diff = gmro_diff
            best_length = length
            best_gmro = gmro_value

    if best_length is None:
        return None

    x_col_idw = next(col for col in df_idw.columns if col.endswith('X') and str(best_length) in col)
    y_col_idw = next(col for col in df_idw.columns if col.endswith('Y') and str(best_length) in col)
    idw_data = df_idw[[x_col_idw, y_col_idw]].dropna()
    idw_data.columns = ['X', 'Y']
    closest_gm_id_row = idw_data.iloc[(idw_data['X'] - gm_id_target).abs().idxmin()]
    idw_value = closest_gm_id_row['Y']
    w = Id_nmos / idw_value

    return dict(
        gmro=best_gmro,
        L=best_length,
        W=w
    )

# === PMOS Second Stage Design ===

def extract_pmos2_params(VDS, gm_id_target, Id_pmos, gmro_target):
    vds_folder = f"PMOS_{VDS:.1f}"
    base_path = os.path.join("D:/Desktop/DMC/PMOS_1V", vds_folder)
    gmro_path = os.path.join(base_path, "gmro.csv")
    idw_path = os.path.join(base_path, "idw.csv")

    if not os.path.exists(gmro_path) or not os.path.exists(idw_path):
        return None

    try:
        df_gmro = pd.read_csv(gmro_path)
        df_idw = pd.read_csv(idw_path)
    except FileNotFoundError:
        return None

    lengths = []
    x_columns = [col for col in df_gmro.columns if col.endswith('X')]
    for col in x_columns:
        length = parse_length_and_axis(col)[0]
        if length is not None and length not in lengths:
            lengths.append(length)

    best_length = None
    min_gmro_diff = float('inf')
    best_gmro = None

    for length in lengths:
        x_col = next(col for col in df_gmro.columns if col.endswith('X') and str(length) in col)
        y_col = next(col for col in df_gmro.columns if col.endswith('Y') and str(length) in col)
        length_data = df_gmro[[x_col, y_col]].dropna()
        length_data.columns = ['X', 'Y']
        closest_gm_id_row = length_data.iloc[(length_data['X'] - gm_id_target).abs().idxmin()]
        gmro_value = closest_gm_id_row['Y']
        if gmro_value > gmro_target:
            gmro_diff = abs(gmro_value - gmro_target)
            if gmro_diff < min_gmro_diff:
                min_gmro_diff = gmro_diff
                best_length = length
                best_gmro = gmro_value

    if best_length is None:
        return None

    x_col_idw = next(col for col in df_idw.columns if col.endswith('X') and str(best_length) in col)
    y_col_idw = next(col for col in df_idw.columns if col.endswith('Y') and str(best_length) in col)
    idw_data = df_idw[[x_col_idw, y_col_idw]].dropna()
    idw_data.columns = ['X', 'Y']
    closest_gm_id_row = idw_data.iloc[(idw_data['X'] - gm_id_target).abs().idxmin()]
    idw_value = closest_gm_id_row['Y']
    w = Id_pmos / idw_value

    return dict(
        gmro=best_gmro,
        L=best_length,
        W=w
    )

# === NMOS Second Stage Design ===

def extract_ron_nmos(VDS, gm_id_target, gmn_input, Id_nmos):
    vds_folder = f"nmos_{VDS:.1f}"
    base_path = os.path.join("D:/Desktop/DMC/NMOS_1V", vds_folder)
    gmro_path = os.path.join(base_path, "gmro.csv")
    idw_path = os.path.join(base_path, "idw.csv")

    if not os.path.exists(gmro_path) or not os.path.exists(idw_path):
        return None

    try:
        df_gmro = pd.read_csv(gmro_path)
        df_idw = pd.read_csv(idw_path)
    except FileNotFoundError:
        return None

    df_gmro, map_gmro = read_techplot_csv(gmro_path)
    best_gmnron = -np.inf
    best_L = None
    best_gm_id = None

    for L, axis_map in map_gmro.items():
        gm_ids = pd.to_numeric(df_gmro[axis_map['X']], errors='coerce')
        gmnrons = pd.to_numeric(df_gmro[axis_map['Y']], errors='coerce')
        valid = gm_ids[gm_ids >= gm_id_target]
        if valid.empty:
            continue
        idx = (valid - gm_id_target).abs().idxmin()
        gmnron = gmnrons[idx]
        gm_id_val = gm_ids[idx]
        if gmnron > best_gmnron:
            best_gmnron = gmnron
            best_L = L
            best_gm_id = gm_id_val

    if best_L is None:
        return None

    x_col_idw = next(col for col in df_idw.columns if col.endswith('X') and str(best_L) in col)
    y_col_idw = next(col for col in df_idw.columns if col.endswith('Y') and str(best_L) in col)
    idw_data = df_idw[[x_col_idw, y_col_idw]].dropna()
    idw_data.columns = ['X', 'Y']
    closest_gm_id_row = idw_data.iloc[(idw_data['X'] - gm_id_target).abs().idxmin()]
    idw_value = closest_gm_id_row['Y']
    w = Id_nmos / idw_value

    return dict(
        gm_id=best_gm_id,
        gmnron=best_gmnron,
        L=best_L,
        gmn=gmn_input,
        W=w
    )

# === Main LDO 3-Stage Design ===

def design_ldo_passfet_with_auto_length():
    VDD = 1.4
    VOUT = 1.0
    GMI_PASS = 20
    ID_PASS = 5e-3
    VTP = 0.58
    VTN = 0.16
    GM_ID_NMOS2 = 10
    GM_ID_PMOS2 = 20
    ID_STAGE2 = 2e-7
    ID_STAGE1 = 4e-7
    TARGET_GAIN_DB = 100
    BASE_DIR = "D:/Desktop/DMC/PMOS_2V"

    idw_path = os.path.join(BASE_DIR, "idw.csv")
    df_idw, map_idw = read_techplot_csv(idw_path)
    L_min = min(map_idw.keys())

    idx, gm_id3, idw3 = find_closest_entry(df_idw, map_idw, L_min, GMI_PASS)
    W3 = ID_PASS / idw3

    gmro_path = os.path.join(BASE_DIR, "gmro.csv")
    df_gmro, map_gmro = read_techplot_csv(gmro_path)
    _, _, gmro3 = find_closest_entry(df_gmro, map_gmro, L_min, gm_id3)

    vov_p = compute_vov(gm_id3)
    vds_passfet = VDD - VOUT
    vds_nmos_passfet = VDD - vds_passfet

    gain12, stage1, stage2, gmro1 = calculate_stage_gains(TARGET_GAIN_DB, gmro3)

    vov_n = compute_vov(GM_ID_NMOS2)
    vg_n_calc = VTN + vov_n
    vg_n = round_gate_voltage(vg_n_calc, 'up')

    gmn, gmp, rop = compute_gmn_rop(ID_STAGE2, GM_ID_NMOS2, GM_ID_PMOS2, stage2)
    gmro_target = gmp * rop

    vov_p = compute_vov(gm_id3)
    vg_max = VDD - VTP - vov_p
    vg_pass = round_gate_voltage(vg_max, 'down')
    vds_pmos2 = VDD - vg_pass
    vds_nmos2 = vg_pass

    VCM_DD = 0.2
    vds_pmos1 = VDD - VCM_DD - vg_n
    vds_nmos1 = vg_n

    print("\n=== [PASSFET DESIGN] ===")
    print(f"  L = {L_min}, W = {W3:.2e} m, gm/Id = {gm_id3:.2f}, Id/W = {idw3:.2e}, gm*ro = {gmro3:.2f}")
    print(f"  VDS = {vds_passfet:.2f} V")

    print("\n=== [GAIN ALLOCATION] ===")
    print(f"  Stage 1 Gain = {stage1:.2f}")
    print(f"  Stage 2 Gain = {stage2:.2f}")
    print(f"  PassFET Gain = {gmro3:.2f}")

    return {
        "vds_nmos2": vds_nmos2,
        "gm_id_nmos2": GM_ID_NMOS2,
        "gmn": gmn,
        "Id_nmos": ID_STAGE2,
        "vds_pmos2": vds_pmos2,
        "gm_id_pmos2": GM_ID_PMOS2,
        "Id_pmos": ID_STAGE2,
        "gmro_target": gmro_target,
        "vds_nmos1": vds_nmos1,
        "vds_pmos1": vds_pmos1,
        "Id_stage1": ID_STAGE1,
        "gmro1": gmro1,
        "vds_nmos_passfet": vds_nmos_passfet,
        "L_passfet": L_min,
        "gmp": gmp,
        "rop": rop
    }

# === Run End-to-End ===

if __name__ == "__main__":
    design_result = design_ldo_passfet_with_auto_length()
    vds_nmos2 = design_result['vds_nmos2']
    gm_id_nmos2 = design_result['gm_id_nmos2']
    gmn_val = design_result['gmn']
    Id_nmos2 = design_result['Id_nmos']
    vds_pmos2 = design_result['vds_pmos2']
    gm_id_pmos2 = design_result['gm_id_pmos2']
    Id_pmos2 = design_result['Id_pmos']
    gmro_target = design_result['gmro_target']
    vds_nmos1 = design_result['vds_nmos1']
    vds_pmos1 = design_result['vds_pmos1']
    Id_stage1 = design_result['Id_stage1']
    gmro1 = design_result['gmro1']
    vds_nmos_passfet = design_result['vds_nmos_passfet']
    L_passfet = design_result['L_passfet']
    gmp = design_result['gmp']
    rop = design_result['rop']

    print("\n=== [CURRENT MIRROR RESULTS] ===")
    pmos_cm_data = extract_pmos_current_mirror_params(VDS=0.2, gm_id_target=20, Id_pmos=8e-7)
    if pmos_cm_data:
        print(f"  PMOS Current Mirror (First Stage): L = {pmos_cm_data['L']:.2e}, W = {pmos_cm_data['W']:.2e}, Id/W = {pmos_cm_data['idw']:.2e}, VDS = 0.20 V")
    pmos_cm2_data = extract_pmos_current_mirror_params(VDS=0.2, gm_id_target=20, Id_pmos=2e-7)
    if pmos_cm2_data:
        print(f"  PMOS Current Mirror (Second Stage): L = {pmos_cm2_data['L']:.2e}, W = {pmos_cm2_data['W']:.2e}, Id/W = {pmos_cm2_data['idw']:.2e}, VDS = 0.20 V")
    nmos_cm_data = extract_nmos_current_mirror_params(VDS=vds_nmos_passfet, gm_id_target=20, Id_nmos=8e-7)
    if nmos_cm_data:
        print(f"  NMOS Current Mirror: L = {nmos_cm_data['L']:.2e}, W = {nmos_cm_data['W']:.2e}, Id/W = {nmos_cm_data['idw']:.2e}, VDS = {vds_nmos_passfet:.2f} V")

    print("\n=== [FIRST STAGE RESULTS] ===")
    nmos1_data = extract_nmos1_params(VDS=vds_nmos1, gm_id_target=20, Id_nmos=Id_stage1, gmro_target=gmro1)
    if nmos1_data:
        print(f"  NMOS1: L = {nmos1_data['L']:.2e}, W = {nmos1_data['W']:.2e}, gm*ro = {nmos1_data['gmro']:.2f}, VDS = {vds_nmos1:.2f} V")
    pmos1_data = extract_pmos1_params(VDS=vds_pmos1, gm_id_target=gm_id_pmos2, Id_pmos=Id_stage1, gmro_target=gmro1)
    if pmos1_data:
        print(f"  PMOS1: L = {pmos1_data['L']:.2e}, W = {pmos1_data['W']:.2e}, gm*ro = {pmos1_data['gmro']:.2f}, VDS = {vds_pmos1:.2f} V")

    print("\n=== [SECOND STAGE RESULTS] ===")
    nmos2_data = extract_ron_nmos(VDS=vds_nmos2, gm_id_target=gm_id_nmos2, gmn_input=gmn_val, Id_nmos=Id_nmos2)
    if nmos2_data:
        ro_nmos2 = nmos2_data['gmnron'] / nmos2_data['gmn'] if nmos2_data['gmn'] != 0 else float('inf')
        print(f"  NMOS2: L = {nmos2_data['L']:.2e}, W = {nmos2_data['W']:.2e}, gm/Id = {nmos2_data['gm_id']:.2f}, gm*ro = {nmos2_data['gmnron']:.2f}, gm = {nmos2_data['gmn']:.2e}, ro = {ro_nmos2:.2e}, VDS = {vds_nmos2:.2f} V")
    pmos2_data = extract_pmos2_params(VDS=vds_pmos2, gm_id_target=gm_id_pmos2, Id_pmos=Id_pmos2, gmro_target=gmro_target)
    if pmos2_data:
        ro_pmos2 = pmos2_data['gmro'] / gmp if gmp != 0 else float('inf')
        print(f"  PMOS2: L = {pmos2_data['L']:.2e}, W = {pmos2_data['W']:.2e}, gm/Id = {gm_id_pmos2:.2f}, gm*ro = {pmos2_data['gmro']:.2f}, gm = {gmp:.2e}, ro = {ro_pmos2:.2e}, VDS = {vds_pmos2:.2f} V")
