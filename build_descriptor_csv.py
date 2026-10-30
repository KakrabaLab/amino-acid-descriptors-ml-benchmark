"""Build the full descriptor interpretation table (Supplementary Table S4) for d1-d6977."""
import re, sys
import numpy as np, pandas as pd, openpyxl
from scipy.stats import kruskal, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

path = sys.argv[1]; out = sys.argv[2]
rows = list(openpyxl.load_workbook(path, read_only=True).worksheets[0].iter_rows(values_only=True))
ids, names = list(rows[0][1:]), dict(zip(rows[0][1:], rows[1][1:]))
df = pd.DataFrame([r[1:] for r in rows[2:]], index=[r[0] for r in rows[2:]], columns=ids).apply(pd.to_numeric)
CLASS = {"Glycine":"NP","Alanine":"NP","Valine":"NP","Leucine":"NP","Isoleucine":"NP","Methionine":"NP","Proline":"NP",
         "Phenylalanine":"AR","Tyrosine":"AR","Tryptophan":"AR","Serine":"PU","Threonine":"PU","Cysteine":"PU",
         "Asparagine":"PU","Glutamine":"PU","Lysine":"POS","Arginine":"POS","Histidine":"POS","Aspartic Acid":"NEG","Glutamic Acid":"NEG"}
ONE = {"Alanine":"A","Arginine":"R","Asparagine":"N","Aspartic Acid":"D","Cysteine":"C","Glutamic Acid":"E","Glutamine":"Q",
       "Glycine":"G","Histidine":"H","Isoleucine":"I","Leucine":"L","Lysine":"K","Methionine":"M","Phenylalanine":"F",
       "Proline":"P","Serine":"S","Threonine":"T","Tryptophan":"W","Tyrosine":"Y","Valine":"V"}
CLS = [("NP","Nonpolar/aliphatic"),("AR","Aromatic"),("PU","Polar uncharged"),("POS","Positively charged"),("NEG","Negatively charged")]
y = pd.Series(CLASS)[df.index].values

W7 = ["unweighted","atomic mass","van der Waals volume","electronegativity","polarizability","ionization potential","intrinsic (I-) state"]
W6 = ["atomic mass","van der Waals volume","electronegativity","polarizability","ionization potential","intrinsic (I-) state"]
W8 = W7 + ["covalent radius"]
STAT = {"Mean":"mean","StdDev":"population standard deviation","Min":"minimum","Max":"maximum","Range":"range",
        "Median":"median","Q25":"25th percentile","Q75":"75th percentile","IQR":"interquartile range"}
SCALAR = {
 "ExactMolWt":"Monoisotopic molecular weight","AverageMolWt":"Average molecular weight","MolWt":"Average molecular weight",
 "HeavyAtomMolWt":"Molecular weight ignoring hydrogens","TotalAtomCount":"Total number of atoms (including H)",
 "HeavyAtomCount":"Number of heavy (non-H) atoms","HeteroAtomCount":"Number of heteroatoms (non-C, non-H)",
 "AromaticAtomCount":"Number of aromatic atoms","RingAtomCount":"Number of ring atoms","HeteroAtomFraction":"Fraction of atoms that are heteroatoms",
 "AromaticAtomFraction":"Fraction of atoms that are aromatic","RingAtomFraction":"Fraction of atoms in rings",
 "HeteroatomToCarbonRatio":"Heteroatom-to-carbon ratio","HydrogenToCarbonRatio":"Hydrogen-to-carbon ratio (degree of saturation)",
 "MaxAbsEStateIndex":"Maximum absolute atomic E-state index","MaxEStateIndex":"Maximum atomic E-state index",
 "MinAbsEStateIndex":"Minimum absolute atomic E-state index","MinEStateIndex":"Minimum atomic E-state index",
 "qed":"Quantitative estimate of drug-likeness (QED)","SPS":"Spatial score (molecular complexity/3D character)",
 "NumValenceElectrons":"Number of valence electrons","MaxPartialCharge":"Maximum Gasteiger partial charge",
 "MinPartialCharge":"Minimum Gasteiger partial charge","MaxAbsPartialCharge":"Maximum absolute Gasteiger partial charge",
 "MinAbsPartialCharge":"Minimum absolute Gasteiger partial charge","FpDensityMorgan1":"Morgan fingerprint bit density, radius 1",
 "FpDensityMorgan2":"Morgan fingerprint bit density, radius 2","FpDensityMorgan3":"Morgan fingerprint bit density, radius 3",
 "BCUT2D_MWHI":"BCUT: highest eigenvalue, mass-weighted Burden matrix","BCUT2D_MWLOW":"BCUT: lowest eigenvalue, mass-weighted Burden matrix",
 "BCUT2D_CHGHI":"BCUT: highest eigenvalue, charge-weighted Burden matrix","BCUT2D_CHGLO":"BCUT: lowest eigenvalue, charge-weighted Burden matrix",
 "BCUT2D_LOGPHI":"BCUT: highest eigenvalue, logP-weighted Burden matrix","BCUT2D_LOGPLOW":"BCUT: lowest eigenvalue, logP-weighted Burden matrix",
 "BCUT2D_MRHI":"BCUT: highest eigenvalue, molar refractivity-weighted Burden matrix","BCUT2D_MRLOW":"BCUT: lowest eigenvalue, molar refractivity-weighted Burden matrix",
 "AvgIpc":"Average information content of the coefficients of the characteristic polynomial","Ipc":"Information content of the coefficients of the characteristic polynomial",
 "BalabanJ":"Balaban J topological index (distance-based connectivity)","BertzCT":"Bertz molecular complexity index",
 "HallKierAlpha":"Hall-Kier alpha: sum of covalent-radius deviations from sp3 carbon (more negative = more sp2/N/O atoms)",
 "Kappa1":"Kier kappa-1 shape index (relative cyclicity/linearity)","Kappa2":"Kier kappa-2 shape index (branching/linearity)",
 "Kappa3":"Kier kappa-3 shape index (centrality of branching)","Phi":"Kier flexibility index",
 "LabuteASA":"Labute approximate van der Waals surface area","TPSA":"Topological polar surface area",
 "FractionCSP3":"Fraction of carbon atoms that are sp3","NHOHCount":"Number of N-H and O-H groups","NOCount":"Number of N and O atoms",
 "NumAliphaticHeterocycles":"Number of aliphatic heterocycles","NumAliphaticRings":"Number of aliphatic rings","NumAmideBonds":"Number of amide bonds",
 "NumAromaticCarbocycles":"Number of aromatic carbocycles","NumAromaticHeterocycles":"Number of aromatic heterocycles",
 "NumAromaticRings":"Number of aromatic rings","NumAtomStereoCenters":"Number of atom stereocentres","NumHAcceptors":"Number of hydrogen-bond acceptors",
 "NumHDonors":"Number of hydrogen-bond donors","NumHeteroatoms":"Number of heteroatoms","NumHeterocycles":"Number of heterocycles",
 "NumRotatableBonds":"Number of rotatable bonds (flexibility)","NumSaturatedHeterocycles":"Number of saturated heterocycles",
 "NumSaturatedRings":"Number of saturated rings","NumUnspecifiedAtomStereoCenters":"Number of unspecified stereocentres","RingCount":"Number of rings",
 "MolLogP":"Wildman-Crippen octanol-water partition coefficient (hydrophobicity)","MolMR":"Wildman-Crippen molar refractivity (polarizability/bulk)",
 "Asphericity":"Asphericity (deviation from spherical shape)","Eccentricity":"Shape eccentricity from principal moments of inertia",
 "InertialShapeFactor":"Inertial shape factor","NPR1":"Normalized principal moment ratio I1/I3 (rod vs disc/sphere)",
 "NPR2":"Normalized principal moment ratio I2/I3 (disc vs sphere)","PMI1":"Smallest principal moment of inertia","PMI2":"Middle principal moment of inertia",
 "PMI3":"Largest principal moment of inertia","RadiusOfGyration":"Radius of gyration (overall molecular extent)",
 "SpherocityIndex":"Spherocity index","PBF":"Plane of best fit (deviation from planarity)",
}
FR = {"Al_COO":"aliphatic carboxylic acids","Al_OH":"aliphatic hydroxyl groups","Ar_N":"aromatic nitrogens","Ar_NH":"aromatic NH groups",
      "Ar_OH":"aromatic hydroxyl groups","COO":"carboxylic acids","COO2":"carboxylic acids (alternative pattern)","C_O":"carbonyl O",
      "C_O_noCOO":"carbonyl O excluding COOH","Imine":"imines","NH0":"tertiary amines","NH1":"secondary amines","NH2":"primary amines",
      "Nhpyrrole":"H-pyrrole nitrogens","SH":"thiol groups","amide":"amides","benzene":"benzene rings","bicyclic":"bicyclic ring systems",
      "guanido":"guanidine groups","imidazole":"imidazole rings","para_hydroxylation":"para-hydroxylation sites","phenol":"phenols",
      "phenol_noOrthoHbond":"phenols without ortho H-bond","priamide":"primary amides","sulfide":"thioethers"}
ATOMPROP = {"AtomicNumber":"atomic number","AtomicMass":"atomic mass","Degree":"atom degree","TotalValence":"total valence",
            "Hybridization":"hybridization state","AromaticFlag":"aromaticity flag (0/1)","RingFlag":"ring-membership flag (0/1)"}
AGG = {"Mean":"mean","StdDev":"standard deviation","Max":"maximum","Sum":"sum","Range":"range","Min":"minimum"}
GETAWAY_REF = "Consonni et al. 2002 [58]"

def wl(w):
    return "no atomic weighting (unweighted)" if w == "unweighted" else f"{w} weighting"

def getaway(k):  # 1-based
    j = k - 1
    if j < 4: return ["ITH: total information content on leverage equality","ISH: standardized information content on leverage equality",
                      "HIC: mean information content on leverage magnitude","HGM: geometric mean of leverages"][j], None
    if j < 144:
        w, r = divmod(j - 4, 20)
        if r < 9: return f"H{r}: H-GETAWAY autocorrelation, lag {r}", W7[w]
        if r == 9: return "HT: total H index", W7[w]
        if r < 19: return f"HATS{r-10}: leverage-weighted autocorrelation, lag {r-10}", W7[w]
        return "HATST: total HATS index", W7[w]
    if j == 144: return "RCON: Randic-type R matrix connectivity index", None
    if j == 145: return "RARS: row average sum of the R matrix", None
    if j == 146: return "REIG: first eigenvalue of the R matrix", None
    w, r = divmod(j - 147, 18)
    if r < 8: return f"R{r+1}: R-GETAWAY autocorrelation, lag {r+1}", W7[w]
    if r == 8: return "RT: total R index", W7[w]
    if r < 17: return f"R{r-8}+: maximal R-GETAWAY autocorrelation, lag {r-8}", W7[w]
    return "RTmax: maximal R index", W7[w]

def decode(name):
    """Return family, parameter text, weighting, interpretation."""
    dim = name.split("_")[0]; body = name[len(dim)+1:]
    stat = None
    if dim == "4D":
        body, stat = body.rsplit("_", 1)
    m = re.match(r"([A-Za-z0-9]+)_(\d{4})$", body)
    fam, param, wt, interp = body, "", "", ""
    if m:
        fam, k = m.group(1), int(m.group(2)); j = k - 1
        if fam == "MORSE":
            wt = W7[j // 32]; s = j % 32; param = f"s = {s} 1/A"
            interp = (f"3D-MoRSE electron-diffraction signal at scattering parameter s = {s} 1/A, {wl(wt)}; "
                      + ("sum of weight products over all atom pairs (s = 0)" if s == 0 else
                         "high s: dominated by short (bonded) distances, contrasts C-H versus heavy-atom bond lengths" if s >= 8 else
                         "low s: sensitive to longer interatomic distances and overall molecular extent"))
        elif fam == "RDF":
            wt = W7[j // 30]; R = 1 + 0.5 * (j % 30); param = f"R = {R:.1f} A"
            interp = f"Radial distribution function at interatomic distance R = {R:.1f} A (beta = 100), {wl(wt)}; probability of atom pairs at that distance"
        elif fam == "AUTOCORR3D":
            wt = W8[j // 10]; lag = j % 10 + 1; param = f"lag {lag}"
            interp = f"3D autocorrelation: sum of w_i*w_j*r_ij (Euclidean distance) over atom pairs {lag} bond(s) apart, {wl(wt)}, normalized by N(N-1)"
        elif fam == "AUTOCORR2D":
            form = ["standard Broto-Moreau (ATS, log-scaled)","centered absolute Broto-Moreau (ATSC)","Moran (MATS)","Geary (GATS)"][j // 48]
            wt = W6[(j % 48) // 8]; lag = j % 8 + 1; param = f"lag {lag}"
            interp = f"2D topological autocorrelation, {form}, {wl(wt)}, atom pairs {lag} bond(s) apart"
        elif fam == "USRCAT" or fam == "USR":
            sub = ["all atoms","hydrophobic atoms","aromatic atoms","H-bond acceptor atoms","H-bond donor atoms"][j // 12]
            ref = ["molecular centroid","atom closest to centroid","atom farthest from centroid","atom farthest from that atom"][(j % 12) // 3]
            mom = ["mean","standard deviation","cube-root skewness"][j % 3]
            param = f"{sub}; {ref}; {mom}"
            interp = f"Ultrafast shape recognition: {mom} of distances of {sub} from the {ref}"
        elif fam == "GETAWAY":
            interp, w = getaway(k); wt = w or ""
            interp = f"GETAWAY (leverage/geometry-topology): {interp}" + (f", {wl(wt)}" if wt else "")
        elif fam == "WHIM":
            interp = f"WHIM (weighted holistic invariant molecular) descriptor element {k}: principal-axis size, shape, symmetry or density statistic (element-level decoding depends on RDKit version)"
        elif fam in ("ECFP4","ECFP6","FCFP4","FCFP6"):
            r = fam[-1]; kind = "feature-based (pharmacophoric) Morgan" if fam[0] == "F" else "Morgan circular"
            param = f"bit {k}"; interp = f"{kind} fingerprint, diameter {r} (radius {int(r)//2}), bit {k}: presence (1) or absence (0) of hashed circular substructure(s)"
        elif fam == "RDKPath":
            param = f"bit {k}"; interp = f"RDKit path-based (Daylight-like) fingerprint bit {k}: presence of hashed linear/branched bond paths"
        elif fam == "AtomPair":
            param = f"bit {k}"; interp = f"Atom-pair fingerprint bit {k}: presence of hashed atom-type pairs at a given topological distance"
        elif fam == "TopologicalTorsion":
            param = f"bit {k}"; interp = f"Topological torsion fingerprint bit {k}: presence of hashed four-atom linear torsion paths"
        elif fam == "MACCS":
            param = f"key {k}"; interp = f"MACCS structural key {k}: presence of a predefined substructure"
    else:
        mm = re.match(r"(\w+?)_(Count|Fraction)$", body)
        if body in SCALAR: interp = SCALAR[body]
        elif body.startswith("fr_"): interp = f"Count of {FR.get(body[3:], body[3:])}"
        elif re.match(r"Chi\d[nv]?$", body):
            o = body[3]; t = {"n":"simple-valence","v":"valence"}.get(body[4:], "simple")
            interp = f"Kier-Hall connectivity index chi-{o} ({t}); branching/size"
        elif re.match(r"(PEOE|SMR|SlogP)_VSA\d+$", body):
            p = {"PEOE":"Gasteiger partial charge","SMR":"molar refractivity contribution","SlogP":"logP contribution"}[body.split("_")[0]]
            interp = f"van der Waals surface area of atoms whose {p} falls in bin {re.findall(r'\d+$', body)[0]}"
        elif re.match(r"EState_VSA\d+$", body): interp = f"Sum of E-state indices of atoms whose VSA falls in bin {body[10:]} (EState_VSA)"
        elif re.match(r"VSA_EState\d+$", body): interp = f"Sum of VSA of atoms whose E-state index falls in bin {body[10:]} (VSA_EState)"
        elif mm and dim == "1D":
            el = mm.group(1); interp = f"{'Number' if mm.group(2)=='Count' else 'Fraction'} of {el} atoms"
        elif dim == "1D" and "_" in body:
            p, a = body.rsplit("_", 1); interp = f"{AGG.get(a, a).capitalize()} of {ATOMPROP.get(p, p)} over all atoms"
        else: interp = body
    if stat:
        interp = f"{interp}; {STAT[stat]} across the conformer ensemble (4D)"
    elif dim == "3D":
        interp = f"{interp}; first retained conformer"
    return fam, stat or "", param, wt, interp

def block(c):
    n = int(c[1:])
    if n <= 22: return "Graph theoretic"
    if n <= 73: return "1D"
    if 407 <= n <= 2358: return "2D fingerprint"
    if n <= 2358: return "2D"
    if n <= 3257: return "3D"
    return "4D"

# statistics
X = StandardScaler().fit_transform(df.values)
imps = np.array([RandomForestClassifier(n_estimators=500, random_state=s, n_jobs=-1).fit(X, y).feature_importances_ for s in range(10)])
imp_m = pd.Series(imps.mean(0), index=df.columns); imp_sd = pd.Series(imps.std(0, ddof=1), index=df.columns)
rank = imp_m.rank(ascending=False, method="min").astype(int)
pv, H = [], []
for c in df.columns:
    v = df[c].values
    try:
        r = kruskal(*[v[y == k] for k, _ in CLS]); pv.append(r.pvalue); H.append(r.statistic)
    except ValueError:
        pv.append(np.nan); H.append(np.nan)
pv = np.array(pv); n = np.sum(~np.isnan(pv))
order = np.argsort(np.where(np.isnan(pv), np.inf, pv)); q = np.full(len(pv), np.nan)
ranked = pv[order][:n] * n / np.arange(1, n + 1); ranked = np.minimum.accumulate(ranked[::-1])[::-1]
q[order[:n]] = np.minimum(ranked, 1)
mw = df["d23"]

recs = []
for i, c in enumerate(df.columns):
    nm = names[c]
    if block(c) == "Graph theoretic":
        fam, stat, param, wt, interp = "Graph-theoretic side-chain invariant", "", "", "", nm + " (hydrogen-suppressed side-chain + C-alpha graph)"
        rd = "Graph theoretic (prior study)"
    else:
        fam, stat, param, wt, interp = decode(nm); rd = nm
    s = df[c].rename(ONE).sort_values(ascending=False)
    const_info = s.nunique()
    rec = {"ID": c, "Descriptor_name": rd, "Block": block(c), "Family": fam, "Ensemble_statistic": stat,
           "Parameter": param, "Atomic_weighting": wt, "Interpretation": interp,
           "RF_Gini_importance_mean": imp_m[c], "RF_Gini_importance_SD": imp_sd[c], "RF_Gini_rank": rank[c],
           "KW_H": H[i], "KW_p_nominal": pv[i], "KW_p_BH_FDR": q[i],
           "Spearman_rho_MolWt": spearmanr(df[c], mw).correlation if const_info > 1 else np.nan,
           "Highest_3_residues": ",".join(s.index[:3]), "Lowest_3_residues": ",".join(s.index[::-1][:3]),
           "N_distinct_values": const_info}
    for k, lab in CLS:
        rec[f"Mean_{lab.replace(' ','_').replace('/','_')}"] = df[c][y == k].mean()
    recs.append(rec)
T = pd.DataFrame(recs)
T.to_csv(out, index=False, float_format="%.6g", encoding="utf-8")
print(T.shape, "min BH q", np.nanmin(q), "n nominal<0.05", int(np.nansum(pv < 0.05)))
