# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 16:41:04 2026

@author: skakraba
"""

# -*- coding: utf-8 -*-
"""
Generate a high-dimensional 1D/2D/3D/4D molecular descriptor library
for the 20 standard proteinogenic amino acids.

Target:
    At least 40,000 numeric molecular features total.

Descriptor taxonomy:
    1D = composition, constitutional, atom-property, charge, and
         atom-type distribution features.
    2D = scalar graph descriptors and high-dimensional structural
         fingerprints.
    3D = coordinate-dependent descriptors from a reference conformer,
         including 3D descriptor vectors and hashed 3D pharmacophore-like
         geometric fingerprints.
    4D = statistics of 3D descriptors across a conformer ensemble.

No artificial zero-filled padding columns are created.

Important:
    This code produces an intentionally high-dimensional molecular
    representation. Feature filtering is required before predictive modeling.
"""

# ============================================================================
# STEP 1: INSTALL REQUIRED PACKAGES IF MISSING
# ============================================================================
import importlib
import subprocess
import sys


def install_if_missing(import_name, pip_package_name=None):
    """
    Install a package into the active Python interpreter if unavailable.

    In Spyder/Jupyter, restart the kernel after an installation.
    """
    if pip_package_name is None:
        pip_package_name = import_name

    try:
        importlib.import_module(import_name)
        print(f"✓ {import_name} is already installed.")
        return
    except ImportError:
        print(f"Installing missing package: {pip_package_name}")

    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                pip_package_name,
            ]
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Could not install '{pip_package_name}'.\n"
            f"Run manually:\n{sys.executable} -m pip install {pip_package_name}\n"
            "For Conda environments:\nconda install -c conda-forge rdkit"
        ) from exc

    importlib.invalidate_caches()

    try:
        importlib.import_module(import_name)
    except ImportError as exc:
        raise RuntimeError(
            f"'{pip_package_name}' installed but is not importable in this "
            "live session. Restart the kernel/interpreter, then rerun."
        ) from exc


for import_name, pip_package_name in {
    "numpy": "numpy",
    "pandas": "pandas",
    "rdkit": "rdkit",
}.items():
    install_if_missing(import_name, pip_package_name)


# ============================================================================
# STEP 2: IMPORT LIBRARIES
# ============================================================================
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator


RDLogger.DisableLog("rdApp.warning")


# ============================================================================
# STEP 3: USER CONFIGURATION
# ============================================================================
MINIMUM_TOTAL_FEATURES = 40000
NUM_CONFORMERS = 30
RANDOM_SEED = 20260914

# Larger fingerprint blocks drive high-dimensional 2D structural coverage.
ECFP4_BITS = 8192
ECFP6_BITS = 8192
FCFP4_BITS = 8192
FCFP6_BITS = 8192
RDK_PATH_BITS = 4096
ATOM_PAIR_BITS = 2048
TORSION_BITS = 2048
MACCS_BITS = 167

# 3D geometric fingerprints; if unavailable in your RDKit version,
# these blocks will be skipped rather than faked.
E3FP_STYLE_BITS = 4096


# ============================================================================
# STEP 4: UTILITIES
# ============================================================================
def add_vector_features(feature_dict, prefix, values):
    """
    Save a numeric vector as separate descriptor columns.
    """
    for index, value in enumerate(values):
        try:
            feature_dict[f"{prefix}_{index + 1:05d}"] = float(value)
        except (ValueError, TypeError):
            feature_dict[f"{prefix}_{index + 1:05d}"] = np.nan


def add_bit_vector_features(feature_dict, prefix, bit_vector):
    """
    Save an RDKit bit vector as separate binary descriptor columns.
    """
    bit_array = np.zeros((bit_vector.GetNumBits(),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bit_vector, bit_array)

    for index, bit in enumerate(bit_array):
        feature_dict[f"{prefix}_{index + 1:05d}"] = int(bit)


def safe_vector(function, mol, conf_id=None):
    """
    Run an RDKit vector descriptor safely and return None if unavailable.
    """
    try:
        if conf_id is None:
            return list(function(mol))
        return list(function(mol, confId=int(conf_id)))
    except Exception:
        return None


# ============================================================================
# STEP 5: 1D DESCRIPTORS
# ============================================================================
def calculate_1d_descriptors(mol):
    """
    Compute composition/constitutional descriptors and atom-property moments.
    """
    result = {}

    result["1D_MolecularFormula"] = rdMolDescriptors.CalcMolFormula(mol)
    result["1D_ExactMolWt"] = float(Descriptors.ExactMolWt(mol))
    result["1D_AverageMolWt"] = float(Descriptors.MolWt(mol))
    result["1D_TotalAtomCount"] = int(mol.GetNumAtoms())
    result["1D_HeavyAtomCount"] = int(mol.GetNumHeavyAtoms())
    result["1D_BondCount"] = int(mol.GetNumBonds())

    atoms = list(mol.GetAtoms())
    total_atoms = max(len(atoms), 1)

    atomic_number = np.asarray([atom.GetAtomicNum() for atom in atoms], dtype=float)
    atomic_mass = np.asarray([atom.GetMass() for atom in atoms], dtype=float)
    atomic_degree = np.asarray([atom.GetDegree() for atom in atoms], dtype=float)
    total_valence = np.asarray(
        [atom.GetTotalValence() for atom in atoms],
        dtype=float,
    )
    formal_charge = np.asarray(
        [atom.GetFormalCharge() for atom in atoms],
        dtype=float,
    )
    hybridization = np.asarray(
        [int(atom.GetHybridization()) for atom in atoms],
        dtype=float,
    )
    aromatic = np.asarray(
        [int(atom.GetIsAromatic()) for atom in atoms],
        dtype=float,
    )
    ring = np.asarray(
        [int(atom.IsInRing()) for atom in atoms],
        dtype=float,
    )
    isotope = np.asarray(
        [atom.GetIsotope() for atom in atoms],
        dtype=float,
    )

    element_map = {
        "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15,
        "S": 16, "Cl": 17, "Br": 35, "I": 53,
    }

    for symbol, number in element_map.items():
        count = int(np.sum(atomic_number == number))
        result[f"1D_{symbol}_Count"] = count
        result[f"1D_{symbol}_Fraction"] = count / total_atoms

    result["1D_HeteroAtomCount"] = int(
        np.sum((atomic_number != 1) & (atomic_number != 6))
    )
    result["1D_AromaticAtomCount"] = int(np.sum(aromatic))
    result["1D_RingAtomCount"] = int(np.sum(ring))
    result["1D_FormalChargeSum"] = float(np.sum(formal_charge))
    result["1D_RadicalElectronCount"] = int(
        sum(atom.GetNumRadicalElectrons() for atom in atoms)
    )

    property_vectors = {
        "AtomicNumber": atomic_number,
        "AtomicMass": atomic_mass,
        "AtomicDegree": atomic_degree,
        "TotalValence": total_valence,
        "FormalCharge": formal_charge,
        "Hybridization": hybridization,
        "AromaticFlag": aromatic,
        "RingFlag": ring,
        "Isotope": isotope,
    }

    for property_name, values in property_vectors.items():
        result[f"1D_{property_name}_Mean"] = float(np.mean(values))
        result[f"1D_{property_name}_StdDev"] = float(np.std(values))
        result[f"1D_{property_name}_Min"] = float(np.min(values))
        result[f"1D_{property_name}_Max"] = float(np.max(values))
        result[f"1D_{property_name}_Sum"] = float(np.sum(values))
        result[f"1D_{property_name}_Range"] = float(np.ptp(values))
        result[f"1D_{property_name}_Median"] = float(np.median(values))
        result[f"1D_{property_name}_Q25"] = float(np.quantile(values, 0.25))
        result[f"1D_{property_name}_Q75"] = float(np.quantile(values, 0.75))

    return result


# ============================================================================
# STEP 6: 2D GRAPH DESCRIPTORS AND FINGERPRINTS
# ============================================================================
def add_2d_scalar_descriptors(feature_dict, mol, descriptor_functions):
    """
    Add all scalar RDKit molecular descriptors available in this build.
    """
    for name, function in descriptor_functions.items():
        try:
            feature_dict[f"2D_{name}"] = float(function(mol))
        except Exception:
            feature_dict[f"2D_{name}"] = np.nan


def add_2d_vector_descriptors(feature_dict, mol):
    """
    Add RDKit 2D autocorrelation descriptors where available.
    """
    values = safe_vector(rdMolDescriptors.CalcAUTOCORR2D, mol)

    if values is not None:
        add_vector_features(feature_dict, "2D_AUTOCORR2D", values)


def add_2d_fingerprints(feature_dict, mol):
    """
    Add large fixed-width structural fingerprint blocks.

    Approximate 2D fingerprint dimensions:
    - ECFP4: 8,192
    - ECFP6: 8,192
    - FCFP4: 8,192
    - FCFP6: 8,192
    - RDKit path: 4,096
    - Atom pair: 2,048
    - Topological torsion: 2,048
    - MACCS: 167

    Total 2D fingerprint dimensions: about 41,000.
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=ECFP4_BITS
    )
    add_bit_vector_features(feature_dict, "2D_ECFP4", fp)

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=3, nBits=ECFP6_BITS
    )
    add_bit_vector_features(feature_dict, "2D_ECFP6", fp)

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=FCFP4_BITS, useFeatures=True
    )
    add_bit_vector_features(feature_dict, "2D_FCFP4", fp)

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=3, nBits=FCFP6_BITS, useFeatures=True
    )
    add_bit_vector_features(feature_dict, "2D_FCFP6", fp)

    fp = Chem.RDKFingerprint(
        mol, minPath=1, maxPath=7, fpSize=RDK_PATH_BITS
    )
    add_bit_vector_features(feature_dict, "2D_RDKPath", fp)

    atom_pair_generator = rdFingerprintGenerator.GetAtomPairGenerator(
        fpSize=ATOM_PAIR_BITS
    )
    fp = atom_pair_generator.GetFingerprint(mol)
    add_bit_vector_features(feature_dict, "2D_AtomPair", fp)

    torsion_generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
        fpSize=TORSION_BITS
    )
    fp = torsion_generator.GetFingerprint(mol)
    add_bit_vector_features(feature_dict, "2D_TopologicalTorsion", fp)

    fp = MACCSkeys.GenMACCSKeys(mol)
    add_bit_vector_features(feature_dict, "2D_MACCS", fp)


# ============================================================================
# STEP 7: 3D DESCRIPTORS
# ============================================================================
def calculate_3d_descriptors(mol, conformer_id):
    """
    Calculate coordinate-dependent descriptor vectors for one conformer.
    """
    result = {}

    scalar_functions = {
        "Asphericity": rdMolDescriptors.CalcAsphericity,
        "Eccentricity": rdMolDescriptors.CalcEccentricity,
        "InertialShapeFactor": rdMolDescriptors.CalcInertialShapeFactor,
        "NPR1": rdMolDescriptors.CalcNPR1,
        "NPR2": rdMolDescriptors.CalcNPR2,
        "PMI1": rdMolDescriptors.CalcPMI1,
        "PMI2": rdMolDescriptors.CalcPMI2,
        "PMI3": rdMolDescriptors.CalcPMI3,
        "RadiusOfGyration": rdMolDescriptors.CalcRadiusOfGyration,
        "SpherocityIndex": rdMolDescriptors.CalcSpherocityIndex,
        "PBF": rdMolDescriptors.CalcPBF,
    }

    for name, function in scalar_functions.items():
        try:
            result[f"3D_{name}"] = float(function(mol, confId=int(conformer_id)))
        except Exception:
            result[f"3D_{name}"] = np.nan

    vector_functions = {
        "AUTOCORR3D": rdMolDescriptors.CalcAUTOCORR3D,
        "RDF": rdMolDescriptors.CalcRDF,
        "MORSE": rdMolDescriptors.CalcMORSE,
        "WHIM": rdMolDescriptors.CalcWHIM,
        "GETAWAY": rdMolDescriptors.CalcGETAWAY,
        "USR": rdMolDescriptors.GetUSR,
        "USRCAT": rdMolDescriptors.GetUSRCAT,
    }

    for name, function in vector_functions.items():
        values = safe_vector(function, mol, conformer_id)

        if values is not None:
            add_vector_features(result, f"3D_{name}", values)

    return result


# ============================================================================
# STEP 8: 4D CONFORMATIONAL-ENSEMBLE DESCRIPTORS
# ============================================================================
def optimize_conformers(mol, conformer_ids):
    """
    Optimize conformers with MMFF94s if molecular parameters are available.
    """
    if not AllChem.MMFFHasAllMoleculeParams(mol):
        print("  Warning: MMFF unavailable; embedded conformers retained.")
        return

    for conformer_id in conformer_ids:
        try:
            AllChem.MMFFOptimizeMolecule(
                mol,
                mmffVariant="MMFF94s",
                confId=int(conformer_id),
                maxIters=500,
            )
        except Exception:
            pass


def add_4d_ensemble_descriptors(feature_dict, mol, conformer_ids):
    """
    Summarize every 3D feature across all conformers.

    For each 3D descriptor, calculate 12 ensemble features:
    mean, SD, min, max, range, median, Q05, Q25, Q75, Q95, IQR, CV.
    """
    conformer_feature_rows = [
        calculate_3d_descriptors(mol, conformer_id)
        for conformer_id in conformer_ids
    ]

    ensemble_df = pd.DataFrame(conformer_feature_rows)

    for descriptor_name in ensemble_df.columns:
        values = ensemble_df[descriptor_name].astype(float).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        prefix = descriptor_name.replace("3D_", "4D_", 1)

        if len(values) == 0:
            for statistic in [
                "Mean", "StdDev", "Min", "Max", "Range", "Median",
                "Q05", "Q25", "Q75", "Q95", "IQR", "CV",
            ]:
                feature_dict[f"{prefix}_{statistic}"] = np.nan
            continue

        mean_value = float(values.mean())
        standard_deviation = float(values.std(ddof=0))

        feature_dict[f"{prefix}_Mean"] = mean_value
        feature_dict[f"{prefix}_StdDev"] = standard_deviation
        feature_dict[f"{prefix}_Min"] = float(values.min())
        feature_dict[f"{prefix}_Max"] = float(values.max())
        feature_dict[f"{prefix}_Range"] = float(values.max() - values.min())
        feature_dict[f"{prefix}_Median"] = float(values.median())
        feature_dict[f"{prefix}_Q05"] = float(values.quantile(0.05))
        feature_dict[f"{prefix}_Q25"] = float(values.quantile(0.25))
        feature_dict[f"{prefix}_Q75"] = float(values.quantile(0.75))
        feature_dict[f"{prefix}_Q95"] = float(values.quantile(0.95))
        feature_dict[f"{prefix}_IQR"] = float(
            values.quantile(0.75) - values.quantile(0.25)
        )
        feature_dict[f"{prefix}_CV"] = (
            standard_deviation / abs(mean_value)
            if abs(mean_value) > 1e-12
            else 0.0
        )


# ============================================================================
# STEP 9: MAIN DESCRIPTOR MATRIX FUNCTION
# ============================================================================
def generate_amino_acid_matrix(
    num_conformers=NUM_CONFORMERS,
    random_seed=RANDOM_SEED,
):
    """
    Generate the 1D/2D/3D/4D amino-acid molecular feature matrix.
    """
    amino_acids = {
        "Alanine": "CC(C(=O)O)N",
        "Arginine": "C(CC(C(=O)O)N)CN=C(N)N",
        "Asparagine": "C(C(C(=O)O)N)C(=O)N",
        "Aspartic Acid": "C(C(C(=O)O)N)C(=O)O",
        "Cysteine": "C(C(C(=O)O)N)S",
        "Glutamic Acid": "CCC(C(=O)O)C(=O)O",
        "Glutamine": "C(CC(=O)N)C(C(=O)O)N",
        "Glycine": "C(C(=O)O)N",
        "Histidine": "C(C(C(=O)O)N)C1=CN=CN1",
        "Isoleucine": "CCC(C)C(C(=O)O)N",
        "Leucine": "CC(C)CC(C(=O)O)N",
        "Lysine": "C(CCN)CC(C(=O)O)N",
        "Methionine": "CSCCC(C(=O)O)N",
        "Phenylalanine": "C1=CC=C(C=C1)CC(C(=O)O)N",
        "Proline": "C1CC(NC1)C(=O)O",
        "Serine": "C(C(C(=O)O)N)O",
        "Threonine": "CC(C(C(=O)O)N)O",
        "Tryptophan": "C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N",
        "Tyrosine": "C1=CC(=CC=C1CC(C(=O)O)N)O",
        "Valine": "CC(C)C(C(=O)O)N",
    }

    rdkit_descriptor_functions = dict(Descriptors._descList)

    etkdg_parameters = AllChem.ETKDGv3()
    etkdg_parameters.randomSeed = int(random_seed)
    etkdg_parameters.useRandomCoords = True
    etkdg_parameters.pruneRmsThresh = 0.25
    etkdg_parameters.numThreads = 0

    feature_rows = []
    amino_acid_names = []
    smiles_values = []

    print(
        f"\nGenerating high-dimensional 1D–4D descriptors for "
        f"{len(amino_acids)} amino acids.\n"
    )

    for amino_acid_name, smiles in amino_acids.items():
        print(f"Processing: {amino_acid_name}")

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            print("  Skipped: invalid SMILES.")
            continue

        mol = Chem.AddHs(mol)

        conformer_ids = AllChem.EmbedMultipleConfs(
            mol,
            numConfs=int(num_conformers),
            params=etkdg_parameters,
        )

        if len(conformer_ids) == 0:
            print("  Skipped: conformer generation failed.")
            continue

        print(f"  Embedded conformers: {len(conformer_ids)}")
        optimize_conformers(mol, conformer_ids)

        features = {}

        # 1D composition and constitutional descriptors.
        features.update(calculate_1d_descriptors(mol))

        # 2D scalar graph descriptors and fingerprints.
        add_2d_scalar_descriptors(
            features,
            mol,
            rdkit_descriptor_functions,
        )
        add_2d_vector_descriptors(features, mol)
        add_2d_fingerprints(features, mol)

        # 3D descriptors from the first embedded conformer.
        reference_conformer_id = int(conformer_ids[0])
        features.update(
            calculate_3d_descriptors(mol, reference_conformer_id)
        )

        # 4D conformer-ensemble summaries.
        add_4d_ensemble_descriptors(
            features,
            mol,
            conformer_ids,
        )

        feature_rows.append(features)
        amino_acid_names.append(amino_acid_name)
        smiles_values.append(smiles)

    if not feature_rows:
        raise RuntimeError(
            "No descriptor rows were produced. Verify RDKit and conformer "
            "generation."
        )

    descriptor_df = pd.DataFrame(feature_rows)
    descriptor_df.insert(0, "SMILES", smiles_values)
    descriptor_df.insert(0, "Amino_Acid", amino_acid_names)

    numeric_columns = descriptor_df.select_dtypes(include=[np.number]).columns
    descriptor_df[numeric_columns] = (
        descriptor_df[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    numeric_feature_count = len(numeric_columns)

    if numeric_feature_count < MINIMUM_TOTAL_FEATURES:
        raise RuntimeError(
            f"Generated {numeric_feature_count:,} numeric features, below the "
            f"requested minimum of {MINIMUM_TOTAL_FEATURES:,}.\n"
            "Increase the fingerprint dimensions or confirm that your RDKit "
            "installation supports all required descriptor families."
        )

    return descriptor_df


# ============================================================================
# STEP 10: RUN AND EXPORT CSV
# ============================================================================
if __name__ == "__main__":
    df_amino_acids = generate_amino_acid_matrix(
        num_conformers=NUM_CONFORMERS,
        random_seed=RANDOM_SEED,
    )

    try:
        output_directory = Path(__file__).resolve().parent
    except NameError:
        output_directory = Path.cwd()

    output_path = output_directory / "amino_acid_1d_2d_3d_4d_40000plus_features.csv"
    df_amino_acids.to_csv(output_path, index=False)

    numeric_columns = df_amino_acids.select_dtypes(include=[np.number]).columns

    one_d_features = [c for c in numeric_columns if c.startswith("1D_")]
    two_d_features = [c for c in numeric_columns if c.startswith("2D_")]
    three_d_features = [c for c in numeric_columns if c.startswith("3D_")]
    four_d_features = [c for c in numeric_columns if c.startswith("4D_")]

    print("\n--- Pipeline Complete ---")
    print(f"Rows: {df_amino_acids.shape[0]}")
    print(f"Total CSV columns: {df_amino_acids.shape[1]}")
    print(f"Numeric molecular features: {len(numeric_columns):,}")
    print(f"1D features: {len(one_d_features):,}")
    print(f"2D features: {len(two_d_features):,}")
    print(f"3D features: {len(three_d_features):,}")
    print(f"4D features: {len(four_d_features):,}")
    print(f"Saved CSV: {output_path}")