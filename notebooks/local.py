import re
import numpy as np
import pandas as pd
from ascommon.pdb_tools import sifts


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


def get_uniprot_id_mutation_protherm(x):
    """Wrapper around `sifts.get_uniprot_id_mutation` for Protherm."""
    pdb_id, pdb_mutations, uniprot_id = x
    if pd.isnull(pdb_id) and pd.isnull(pdb_mutations):
        print('Not enough info: {}'.format(dict(x.items())))
        return np.nan, np.nan, np.nan, pdb_mutations
    try:
        return sifts.get_uniprot_id_mutation(pdb_id, None, pdb_mutations, uniprot_id)
    except sifts.SIFTSError as e:
        print('{}: {}'.format(e, dict(x.items())))
        return np.nan, np.nan, np.nan, pdb_mutations


def get_uniprot_id_mutation_rosetta_ddg(x):
    """Wrapper around `sifts.get_uniprot_id_mutation` for Rosetta ddG."""
    pdb_id, pdb_chains, pdb_mutations = x
    if pd.isnull(pdb_id) and pd.isnull(pdb_mutations):
        print('Not enough info: {}'.format(dict(x.items())))
        return np.nan, np.nan, pdb_chains, pdb_mutations
    try:
        return sifts.get_uniprot_id_mutation(pdb_id, pdb_chains, pdb_mutations, None)
    except sifts.SIFTSError as e:
        print('{}: {}'.format(e, dict(x.items())))
        return np.nan, np.nan, pdb_chains, pdb_mutations
