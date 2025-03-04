# Evaluate for rdkit mols generated by other methods

import torch
import argparse
import pickle


from evaluation import *
from rdkit import RDLogger


def rdmol_process(mols, dataset_info, only_2D=False):
    from rdkit.Chem.rdchem import BondType as BT
    bond_encoder = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}

    # if only_2D: return None, atom_type, edge_type, zero; else return pos, atom_type
    processed_list = []
    atom_encoder = dataset_info['atom_encoder']
    for mol in mols:
        N = mol.GetNumAtoms()
        atom_type = torch.tensor([atom_encoder[atom.GetSymbol()] for atom in mol.GetAtoms()])

        if not only_2D:
            pos = mol.GetConformer().GetPositions()
        else:
            pos = None

        if only_2D:
            edge_types = torch.zeros((N, N))
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bond_type = bond.GetBondType()
                order = bond_encoder[bond_type]
                edge_types[start, end] = order
                edge_types[end, start] = order
            fc = torch.tensor([atom.GetFormalCharge() for atom in mol.GetAtoms()])
            processed_list.append((pos, atom_type, edge_types, fc))
        else:
            processed_list.append((pos, atom_type))

    return processed_list


def evaluate(pkl_path, dataset_name, eval_type, sub_geometry, root_path):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    RDLogger.DisableLog('rdApp.error')
    RDLogger.DisableLog('rdApp.warning')

    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Dataset
    if dataset_name == 'qm9':
        dataset_root_path = root_path + 'QM9'
        dataset = QM9Dataset(dataset_root_path)
        dataset_info = get_dataset_info('qm9_with_h')
    elif dataset_name == 'Geom_Drugs':
        dataset_root_path = root_path + 'geom'
        dataset = GeomDrugDataset(dataset_root_path, 'data_geom_drug_1.pt')
        dataset_info = get_dataset_info('geom_with_h_1')
    else:
        raise ValueError("Invalid dataset name!")

    # Split dataset
    split_idx = dataset.get_idx_split()
    train_ds = dataset.index_select(split_idx['train'])
    test_ds = dataset.index_select(split_idx['test'])
    train_mols = [train_ds[i].rdmol for i in range(len(train_ds))]
    test_mols = [test_ds[i].rdmol for i in range(len(test_ds))]

    # Build Evaluation metrics
    EDM_metric = get_edm_metric(dataset_info, train_mols)
    EDM_metric_2D = get_2D_edm_metric(dataset_info, train_mols)
    #mose_metric = get_moses_metrics(test_mols, n_jobs=32, device=device)
    if sub_geometry:
        sub_geo_mmd_metric = get_sub_geometry_metric(
            test_mols, dataset_info, dataset_root_path)

    # Read pickles
    with open(pkl_path, 'rb') as f:
        samples = pickle.load(f)

    results = {}
    if eval_type == '3D' or eval_type == 'both':
        processed_mols = rdmol_process(samples, dataset_info, False)
        stability_res, rdkit_res, sample_rdmols = EDM_metric(processed_mols)
        results['3D'] = {
            'num_molecules': len(sample_rdmols),
            'atom_stability': stability_res['atom_stable'],
            'mol_stability': stability_res['mol_stable'],
            'validity': rdkit_res['Validity'],
            'complete': rdkit_res['Complete'],
        }

        if sub_geometry:
            sub_geo_mmd_res = sub_geo_mmd_metric(
                samples if eval_type == 'both' else sample_rdmols)
            results['3D'].update({
                'bond_length_MMD': sub_geo_mmd_res['bond_length_mean'],
                'bond_angle_MMD': sub_geo_mmd_res['bond_angle_mean'],
                'dihedral_angle_MMD': sub_geo_mmd_res['dihedral_angle_mean']
            })

    if eval_type == '2D' or eval_type == 'both':
        processed_mols = rdmol_process(samples, dataset_info, True)
        stability_res, rdkit_res, complete_rdmols = EDM_metric_2D(
            processed_mols)
        #mose_res = mose_metric(complete_rdmols)
        results['2D'] = {
            'atom_stability': stability_res['atom_stable'],
            'mol_stability': stability_res['mol_stable'],
            'validity': rdkit_res['Validity'],
            'complete': rdkit_res['Complete'],
            'valid_unique': rdkit_res['Unique'],
            'valid_unique_novel': rdkit_res['Novelty'],

        }

    return results
# if __name__ == "__main__":
#     from rdkit import RDLogger

#     seed = 42

#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     # random.seed(seed)

#     # Ignore info output by RDKit
#     RDLogger.DisableLog('rdApp.error')
#     RDLogger.DisableLog('rdApp.warning')

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--pkl_path', type=str, default='../generated_samples/qm9_gschnet.pkl')
#     parser.add_argument('--dataset_name', type=str, default='qm9', help="'qm9', 'Geom_Drugs'")
#     parser.add_argument('--type', type=str, default='3D', help="'3D', '2D', 'both'")
#     parser.add_argument('--sub_geometry', type=eval, default=False, help='Substructure Geometry Evaluation.')
#     parser.add_argument('--root_path', type=str, default='data/', help='Data path')

#     args, unparsed_args = parser.parse_known_args()
#     root_path = args.root_path
#     device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

#     # Dataset
#     if args.dataset_name == 'qm9':
#         dataset_root_path = root_path + 'QM9'
#         dataset = QM9Dataset(dataset_root_path)
#         dataset_info = get_dataset_info('qm9_with_h')
#     elif args.dataset_name == 'Geom_Drugs':
#         dataset_root_path = root_path + 'geom'
#         dataset = GeomDrugsDataset(dataset_root_path, 'data_geom_drug_1.pt')
#         dataset_info = get_dataset_info('geom_with_h_1')
#     else:
#         raise ValueError("Invalid dataset name!")

#     # Split dataset
#     split_idx = dataset.get_idx_split()
#     train_ds = dataset.index_select(split_idx['train'])
#     test_ds = dataset.index_select(split_idx['test'])
#     train_mols = [train_ds[i].rdmol for i in range(len(train_ds))]
#     test_mols = [test_ds[i].rdmol for i in range(len(test_ds))]

#     # Build Evaluation metrics
#     EDM_metric = get_edm_metric(dataset_info, train_mols)
#     EDM_metric_2D = get_2D_edm_metric(dataset_info, train_mols)
#     mose_metric = get_moses_metrics(test_mols, n_jobs=32, device=device)
#     if args.sub_geometry:
#         sub_geo_mmd_metric = get_sub_geometry_metric(test_mols, dataset_info, dataset_root_path)

#     # Read pickles
#     with open(args.pkl_path, 'rb') as f:
#         samples = pickle.load(f)

#     print(args)
#     if args.type == '3D' or args.type == 'both':
#         # convert samples to processed mols
#         processed_mols = rdmol_process(samples, dataset_info, False)

#         # EDM stability evaluation metrics
#         stability_res, rdkit_res, sample_rdmols = EDM_metric(processed_mols)
#         print('Number of molecules: %d' % len(sample_rdmols))
#         print("Metric-3D || atom stability: %.4f, mol stability: %.4f, validity: %.4f, complete: %.4f," % (
#             stability_res['atom_stable'], stability_res['mol_stable'], rdkit_res['Validity'], rdkit_res['Complete']))

#         # Mose evaluation metrics
#         mose_res = mose_metric(sample_rdmols)
#         print("Metric-3D || FCD: %.4f" % (mose_res['FCD']))

#         # 3D geometry
#         if args.sub_geometry:
#             if args.type == 'both':
#                 sub_geo_mmd_res = sub_geo_mmd_metric(samples)
#             else:
#                 sub_geo_mmd_res = sub_geo_mmd_metric(sample_rdmols)
#             print("Metric-Align || Bond Length MMD: %.4f, Bond Angle MMD: %.4f, Dihedral Angle MMD: %.6f" % (
#                 sub_geo_mmd_res['bond_length_mean'], sub_geo_mmd_res['bond_angle_mean'],
#                 sub_geo_mmd_res['dihedral_angle_mean']))
#             # ## bond length
#             # bond_length_str = ''
#             # for sym in dataset_info['top_bond_sym']:
#             #     bond_length_str += f"{sym}: %.4f " % sub_geo_mmd_res[sym]
#             # print(bond_length_str)
#             # ## bond angle
#             # bond_angle_str = ''
#             # for sym in dataset_info['top_angle_sym']:
#             #     bond_angle_str += f'{sym}: %.4f ' % sub_geo_mmd_res[sym]
#             # print(bond_angle_str)
#             # ## dihedral angle
#             # dihedral_angle_str = ''
#             # for sym in dataset_info['top_dihedral_sym']:
#             #     dihedral_angle_str += f'{sym}: %.6f ' % sub_geo_mmd_res[sym]
#             # print(dihedral_angle_str)

#     if args.type == '2D' or args.type == 'both':
#         # convert samples to processed mols
#         processed_mols = rdmol_process(samples, dataset_info, True)

#         stability_res, rdkit_res, complete_rdmols = EDM_metric_2D(processed_mols)
#         print("Metric-2D || atom stability: %.4f, mol stability: %.4f, validity: %.4f, complete: %.4f,"
#             " valid & unique: %.4f, valid & unique & novelty: %.4f" % (stability_res['atom_stable'], stability_res['mol_stable'],
#             rdkit_res['Validity'], rdkit_res['Complete'], rdkit_res['Unique'], rdkit_res['Novelty']))
#         mose_res = mose_metric(complete_rdmols)
#         print("Metric-2D || FCD: %.4f, SNN: %.4f, Frag: %.4f, Scaf: %.4f, IntDiv: %.4f" % (mose_res['FCD'],
#             mose_res['SNN'], mose_res['Frag'], mose_res['Scaf'], mose_res['IntDiv']))
