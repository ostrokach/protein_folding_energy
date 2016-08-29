import re
import numpy as np
import pandas as pd
from kmtools.pdb_tools import sifts


# #################################################################################################
# Parse ΔΔG datasets

def fix_mutations(
        mutations,
        mutation_pdb_re=re.compile('.*PDB: ([\w\d ]+).*'),
        mutation_pir_re=re.compile('.*PIR: ([\w\d ]+).*')):
    """Fix common malformating in mutation strings."""
    if pd.isnull(mutations):
        return np.nan

    mutations = mutations.replace('(S-H)', '').replace('(S-S)', '').strip(' ')
    if '(' in mutations and ')' in mutations:
        match = re.findall('.*\((.*)\).*', mutations)
        if len(match) != 1:
            raise Exception("{}::{}".format(mutations, match))
        mutations = match[0]
    if 'PDB: ' in mutations:
        match = mutation_pdb_re.findall(mutations)
        if len(match) != 1:
            raise Exception("{}::{}".format(mutations, match))
        mutations = match[0]
    elif 'PIR: ' in mutations:
        match = mutation_pir_re.findall(mutations)
        if len(match) != 1:
            raise Exception("{}::{}".format(mutations, match))
        mutations = match[0]

    fixed_mutations = []
    for mutation in mutations.split(','):
        if not mutation:
            continue
        mutation = mutation.strip(' ')
        fixed_mutations.append(mutation)
    return ','.join(fixed_mutations)


def get_uniprot_id_mutation_protherm(pdb_id, pdb_chains, pdb_mutations, uniprot_id):
    """Wrapper around `sifts.get_uniprot_id_mutation` for Protherm."""
    pdb_mutations = pdb_mutations.replace(' ', '')
    if pd.isnull(pdb_id) and pd.isnull(pdb_mutations):
        print('Not enough info: ({}, {}, {})'.format(pdb_id, pdb_mutations, uniprot_id))
        return np.nan, np.nan, np.nan, np.nan
    try:
        result = sifts.get_uniprot_id_mutation(pdb_id, pdb_chains, pdb_mutations, uniprot_id)
        return (
            result.get('uniprot_id_sifts', np.nan),
            result.get('uniprot_mutations_sifts', np.nan),
            result.get('pfam_id_sifts', np.nan),
            result.get('pdb_mutations_sifts', np.nan),
        )
    except sifts.SIFTSError as e:
        print(e)
        return np.nan, np.nan, np.nan, np.nan


def get_uniprot_id_mutation_rosetta_ddg(pdb_id, pdb_chains, pdb_mutations):
    """Wrapper around `sifts.get_uniprot_id_mutation` for Rosetta ddG."""
    if pd.isnull(pdb_id) and pd.isnull(pdb_mutations):
        print('Not enough info: ({}, {}, {})'.format(pdb_id, pdb_chains, pdb_mutations))
        return np.nan, np.nan, np.nan, np.nan
    try:
        result = sifts.get_uniprot_id_mutation(pdb_id, pdb_chains, pdb_mutations, None)
        return (
            result.get('uniprot_id_sifts', np.nan),
            result.get('uniprot_mutations_sifts', np.nan),
            result.get('pfam_id_sifts', np.nan),
            result.get('pdb_mutations_sifts', np.nan),
        )
    except sifts.SIFTSError as e:
        print(e)
        return np.nan, np.nan, np.nan, np.nan
