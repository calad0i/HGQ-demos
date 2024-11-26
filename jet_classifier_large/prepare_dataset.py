import h5py as h5
import numpy as np
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
import argparse


def load(path: Path):
    with h5.File(path, 'r') as f:
        _label = f['jets'][:, -6:-1]  # type: ignore
        assert np.all(np.sum(_label, axis=1) == 1)  # type: ignore
        label = np.argmax(_label, axis=1)  # type: ignore
        feature = np.array(f['jetConstituentList']).astype(np.float16)
    return feature, label


def main(inp: str, out: str, jobs: int):
    root = Path(inp)
    paths = list(root.glob('*.h5'))
    with Pool(jobs) as p:
        r = list(tqdm(p.imap(load, paths), total=len(paths)))
        features, labels = zip(*r)
        feature, label = np.concatenate(features), np.concatenate(labels)
    with h5.File(out, 'w') as f:
        f.create_dataset('feature', data=feature, compression='lzf')
        f.create_dataset('label', data=label, compression='lzf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inp', type=str, required=True)
    parser.add_argument('-o', '--out', type=str, required=True)
    parser.add_argument('-j', '--jobs', type=int, default=8)
    args = parser.parse_args()
    main(args.inp, args.out, args.jobs)

# ======= jets ========
# 0 ptfrac
# 1 pt
# 2 eta
# 3 mass
# 4 tau1_b1
# 5 tau2_b1
# 6 tau3_b1
# 7 tau1_b2
# 8 tau2_b2
# 9 tau3_b2
# 10 tau32_b1
# 11 tau32_b2
# 12 zlogz
# 13 c1_b0
# 14 c1_b1
# 15 c1_b2
# 16 c2_b1
# 17 c2_b2
# 18 d2_b1
# 19 d2_b2
# 20 d2_a1_b1
# 21 d2_a1_b2
# 22 m2_b1
# 23 m2_b2
# 24 n2_b1
# 25 n2_b2
# 26 tau1_b1_mmdt
# 27 tau2_b1_mmdt
# 28 tau3_b1_mmdt
# 29 tau1_b2_mmdt
# 30 tau2_b2_mmdt
# 31 tau3_b2_mmdt
# 32 tau32_b1_mmdt
# 33 tau32_b2_mmdt
# 34 c1_b0_mmdt
# 35 c1_b1_mmdt
# 36 c1_b2_mmdt
# 37 c2_b1_mmdt
# 38 c2_b2_mmdt
# 39 d2_b1_mmdt
# 40 d2_b2_mmdt
# 41 d2_a1_b1_mmdt
# 42 d2_a1_b2_mmdt
# 43 m2_b1_mmdt
# 44 m2_b2_mmdt
# 45 n2_b1_mmdt
# 46 n2_b2_mmdt
# 47 mass_trim
# 48 mass_mmdt
# 49 mass_prun
# 50 mass_sdb2
# 51 mass_sdm1
# 52 multiplicity
# 53 g
# 54 q
# 55 w
# 56 z
# 57 t
# 58 undef (all zeros)


# ======== jetConstituentList ========
# 0 px
# 1 py
# 2 pz
# 3 e
# 4 erel
# 5 pt
# 6 ptrel
# 7 eta
# 8 etarel
# 9 etarot
# 10 phi
# 11 phirel
# 12 phirot
# 13 deltaR
# 14 costheta
# 15 costhetarel
# 16 pdgid (in description but no data given)
