"""
"""
import os.path as op
import re
from functools import lru_cache
import lxml.etree
import numpy as np
import pandas as pd
import Bio.SeqIO
from common import system_tools, pdb_tools


SIFTS_CACHE = {}
SIFTS_CACHE_DIR = None
MUTATION_RE = re.compile('([A-Z]+)([0-9]+)([A-Z].*)')
MUTATION_PDB_RE = re.compile('.*PDB: ([\w\d ]+).*')
MUTATION_PIR_RE = re.compile('.*PIR: ([\w\d ]+).*')
MUTATION_SPLIT_RE = re.compile(' +')


def unique(l):
    """Return a list of unique elements of `l`, preserving order."""
    seen = set()
    return [x for x in l if x not in seen or seen.add(x)]


@lru_cache(maxsize=512)
def get_uniprot_sequence(uniprot_id):
    """
    """
    output_file = op.join(SIFTS_CACHE_DIR, '{}.fasta'.format(uniprot_id))
    try:
        try:
            seqrecord = Bio.SeqIO.read(output_file, 'fasta')
        except FileNotFoundError:
            url = 'http://www.uniprot.org/uniprot/{}.fasta'.format(uniprot_id)
            system_tools.download(url, output_file)
            seqrecord = Bio.SeqIO.read(output_file, 'fasta')
    except ValueError:
        seqrecord = None
    return seqrecord


def fix_mutations(mutations):
    """Fix common malformating in mutation strings."""
    fixed_mutations = []
    mutations = mutations.replace('(S-H)', '').replace('(S-S)', '').strip(' ')
    if '(' in mutations and ')' in mutations:
        match = re.findall('.*\((.*)\).*', mutations)
        if len(match) != 1:
            raise Exception("{}::{}".format(mutations, match))
        mutations = match[0]
    if 'PDB: ' in mutations:
        match = MUTATION_PDB_RE.findall(mutations)
        if len(match) != 1:
            raise Exception("{}::{}".format(mutations, match))
        mutations = match[0]
    elif 'PIR: ' in mutations:
        match = MUTATION_PIR_RE.findall(mutations)
        if len(match) != 1:
            raise Exception("{}::{}".format(mutations, match))
        mutations = match[0]
    for mutation in mutations.split(','):
        if not mutation:
            continue
        mutation = mutation.strip(' ')
        fixed_mutations.append(mutation)
    return ','.join(fixed_mutations)


def split_mutation(mutation):
    """Split mutation where `wt`, `pos`, and `mut` are separated by spaces."""
    mutation_split = MUTATION_SPLIT_RE.split(mutation)
    if len(mutation_split) == 3:
        wt, pos, mut = mutation_split
    elif len(mutation_split) == 2:
        if any(c.isdigit() for c in mutation_split[0]):
            wt, pos, mut = '', *mutation_split
        elif any(c.isdigit() for c in mutation_split[1]):
            wt, pos, mut = *mutation_split, ''
        else:
            raise Exception(mutation)
    else:
        raise Exception(mutation)
    return wt, pos, mut


def validate_mutations(mutations, seqrecord):
    """Make sure that mutations match seqrecord.

    Returns
    -------
    validated_mutations : str
        Comma-separated list of mutations where each unmatched mutation has been replaced by a '?'.
    """
    validated_mutations = []
    for mutation in mutations.split(','):
        wt, pos, mut = split_mutation(mutation)
        wt_valid = ''
        for i, aa in enumerate(wt):
            try:
                mutations_match = str(seqrecord.seq)[int(pos) - 1 + i] == aa
            except (IndexError, ValueError):
                mutations_match = False
            if mutations_match:
                wt_valid += aa
            else:
                wt_valid += '?'
        validated_mutations.append(wt_valid)
    return ','.join(validated_mutations)


def convert_amino_acid(
        pdb_id, pdb_chain, pdb_aa, pdb_resnum, uniprot_id, sifts_df, pdb_resnum_offset=0):
    """Convert a single amino acid from PDB to UniProt coordinates.

    Parameters
    ----------
    pdb_id : str
        PDB ID.
    pdb_chain : str | None
        PDB chain of the residue to be converted.
    pdb_aa : str | ''
        PDB amino acid of the residue to be converted.
    pdb_resnum : str
        PDB RESNUM of the residue to be converted.
    uniprot_id : str
        UniProt ID that is expected.
    sifts_df : DataFrame
        SIFTS data to use for conversion.
    pdb_resnum_offset : int
        Move `pdb_resnum` forward or backward by a certain number of residues.

    Returns
    -------
    dict
    """
    if pdb_resnum_offset:
        pdb_resnum_idx = sifts_df[sifts_df['resnum'] == pdb_resnum].index[0]
        pdb_resnum = sifts_df.loc[pdb_resnum_idx + pdb_resnum_offset, 'resnum']

    pdb_residx = int(''.join(c for c in pdb_resnum if c.isdigit()))

    if pdb_aa:
        if 'uniprot_aa' in sifts_df:
            sifts_df_pdb_aa_match = (sifts_df['uniprot_aa'] == pdb_aa)
        else:
            sifts_df_pdb_aa_match = (sifts_df['pdb_aa'] == pdb_aa)
    else:
        sifts_df_pdb_aa_match = True

    if pdb_chain is None:
        sifts_df_pdb_chain_match = True
    else:
        sifts_df_pdb_chain_match = (sifts_df['pdb_chain'] == pdb_chain)

    try:
        # Get the subset of rows that we are interested in
        sifts_df_subset_0 = sifts_df[
            (sifts_df['pdb_id'] == pdb_id) &
            (sifts_df_pdb_chain_match) &
            (sifts_df['resnum'] == pdb_resnum) &
            (sifts_df_pdb_aa_match) &
            (sifts_df['uniprot_id'] == uniprot_id)]
        # Try using residx instead of resnum
        sifts_df_subset_2 = sifts_df[
            (sifts_df['pdb_id'] == pdb_id) &
            (sifts_df_pdb_chain_match) &
            (sifts_df['residx'] == pdb_residx) &
            (sifts_df_pdb_aa_match) &
            (sifts_df['uniprot_id'] == uniprot_id)]
    except KeyError as e:
        print(e)
        sifts_df_subset_0 = []
        sifts_df_subset_2 = []

    # Try mapping to a wildcard uniprot
    sifts_df_subset_1 = sifts_df[
        (sifts_df['pdb_id'] == pdb_id) &
        (sifts_df_pdb_chain_match) &
        (sifts_df['resnum'] == pdb_resnum) &
        (sifts_df_pdb_aa_match)]
    # Or residx and wildcard uniprot
    sifts_df_subset_3 = sifts_df[
        (sifts_df['pdb_id'] == pdb_id) &
        (sifts_df_pdb_chain_match) &
        (sifts_df['residx'] == pdb_residx) &
        (sifts_df_pdb_aa_match)]

    # Choose the best availible subset
    if len(sifts_df_subset_0):
        sifts_df_subset = sifts_df_subset_0
    if len(sifts_df_subset_1):
        sifts_df_subset = sifts_df_subset_1
    elif len(sifts_df_subset_2):
        sifts_df_subset = sifts_df_subset_2
    elif len(sifts_df_subset_3):
        sifts_df_subset = sifts_df_subset_3
    else:
        error_message = """\
SIFTS failed to match residue ({}, {}, {}, {})\
""".format(pdb_id, pdb_aa, pdb_resnum, uniprot_id)
        raise pdb_tools.SIFTSError(error_message)

    # Result
    uniprot_id = sifts_df_subset.iloc[0].get('uniprot_id', uniprot_id)
    uniprot_aa = sifts_df_subset.iloc[0]['uniprot_aa']
    uniprot_pos = int(sifts_df_subset.iloc[0]['uniprot_position'])
    pdb_chain = sifts_df_subset.iloc[0]['pdb_chain']

    uniprot_seqrecord = get_uniprot_sequence(uniprot_id)
    if str(uniprot_seqrecord.seq)[uniprot_pos - 1] != uniprot_aa:
        uniprot_aa = '?'
        uniprot_pos = np.nan

    return dict(
        uniprot_id=uniprot_id, uniprot_aa=uniprot_aa, uniprot_pos=uniprot_pos, pdb_chain=pdb_chain
    )


def convert_pdb_mutations_to_uniprot(pdb_id, pdb_chains, pdb_mutations, uniprot_id, sifts_df):
    """Convert mutation from PDB to UniProt coordinates.

    Works for a list of mutations joined with ','.

    Parameters
    ----------
    pdb_id : str
    pdb_chains : str | None
        Comma-separated list of PDB chain IDs, with one ID for every mutation

    Returns
    -------
    uniprot_id_out : str
        `uniprot_id` extracted from SIFTS.
    uniprot_mutations : str
        Comma-separated list of mutations in UniProt coordinates.
    """
    uniprot_mutations = []

    def get_pdb_chain_mutation():
        if ',' not in pdb_mutations:
            yield pdb_chains, pdb_mutations
        elif pd.isnull(pdb_chains) or ',' not in pdb_chains:
            for pdb_mutation in pdb_mutations.split(','):
                yield pdb_chains, pdb_mutation
        else:
            yield from zip(pdb_chains.split(','), pdb_mutations.split(','))

    for pdb_chain, pdb_mutation in get_pdb_chain_mutation():
        pdb_wt, pdb_resnum, pdb_mut = split_mutation(pdb_mutation)
        # Convert each of the mutant residues
        uniprot_aa_data = []
        for i, pdb_wt_aa in enumerate(pdb_wt if pdb_wt else ['']):
            uniprot_aa_data.append(
                convert_amino_acid(
                    pdb_id, pdb_chain, pdb_wt_aa, pdb_resnum, uniprot_id, sifts_df,
                    pdb_resnum_offset=i))
        uniprot_id = uniprot_aa_data[0]['uniprot_id']
        pdb_chains = ','.join(x['pdb_chain'] for x in uniprot_aa_data)
        uniprot_wt = ''.join(x['uniprot_aa'] for x in uniprot_aa_data)
        uniprot_pos = uniprot_aa_data[0]['uniprot_pos']
        uniprot_mutation = '{}{}{}'.format(uniprot_wt, uniprot_pos, pdb_mut)
        uniprot_mutations.append(uniprot_mutation)

    return uniprot_id, ','.join(uniprot_mutations), pdb_chains


def get_sifts_data(pdb_id, pdb_mutations):
    """Wrapper around `pdb_tools.get_sifts_data`."""
    if SIFTS_CACHE_DIR is None:
        raise Exception("You need to set 'SIFTS_CACHE_DIR' in order to use this function!")
    try:
        sifts_df = SIFTS_CACHE[pdb_id]
    except KeyError:
        sifts_df = pdb_tools.get_sifts_data(pdb_id, SIFTS_CACHE_DIR)
        SIFTS_CACHE[pdb_id] = sifts_df

    # Sometimes SIFTS does not contain any uniprot information for the protein in question
    subset_columns = ['uniprot_position', 'uniprot_aa', 'pdb_chain']
    missing_columns = sorted(set(subset_columns) - set(sifts_df.columns))
    if missing_columns == ['uniprot_aa', 'uniprot_position']:
        raise pdb_tools.SIFTSError('SIFTS has no UniProt annotation for this protein')
    elif missing_columns:
        raise pdb_tools.SIFTSError("SIFTS information missing: {}".format(missing_columns))
    sifts_df = sifts_df.dropna(subset=subset_columns)

    # residx counts the index of the residue in the pdb, starting from 1
    residx = []
    for pdb_chain in set(sifts_df['pdb_chain']):
        residx.extend(range(1, len(sifts_df[sifts_df['pdb_chain'] == pdb_chain]) + 1))
    sifts_df['residx'] = residx

    assert sifts_df['resnum'].dtype.char == 'O'
    assert sifts_df['residx'].dtype.char in ['l', 'd']
    return sifts_df


def get_uniprot_id_mutation(pdb_id, pdb_chains, pdb_mutations, uniprot_id):
    """Convert columns of PDB mutations to UniProt.

    Parameters
    ----------
    pdb_id : str
    pdb_mutations : str
    uniprot_id : str

    Returns
    -------
    uniprot_id : str
    uniprot_mutations : str
        Comma-separated list of mutations.
    pdb_mutations : str
        Comma-separated list of cleaned-up PDB mutations.
    """
    if pd.isnull(pdb_mutations) or pdb_mutations.startswith('WILD'):
        return uniprot_id, 'WILD', np.nan

    pdb_mutations = fix_mutations(pdb_mutations)

    def _uniprot_fallback(uniprot_id, pdb_mutations):
        """Try treating PDB mutations as though they are from UniProt."""
        if pd.isnull(uniprot_id):
            return np.nan, np.nan, np.nan
        else:
            uniprot_seqrecord = get_uniprot_sequence(uniprot_id)
            validated_mutations = validate_mutations(pdb_mutations, uniprot_seqrecord)
            return uniprot_id, validated_mutations, np.nan

    # If pdb_id is null, make sure pdb_mutations matches UniProt
    if pd.isnull(pdb_id):
        return _uniprot_fallback(uniprot_id, pdb_mutations)

    # Get SIFTS data
    try:
        sifts_df = get_sifts_data(pdb_id, pdb_mutations)
    except lxml.etree.XMLSyntaxError:
        return _uniprot_fallback(uniprot_id, pdb_mutations)

    uniprot_id, uniprot_mutations, pdb_chains = convert_pdb_mutations_to_uniprot(
        pdb_id, pdb_chains, pdb_mutations, uniprot_id, sifts_df)
    return uniprot_id, uniprot_mutations, pdb_chains, pdb_mutations


def get_uniprot_id_mutation_protherm(x):
    """Wrapper around `get_uniprot_id_mutation` which catches Exceptions."""
    pdb_id, pdb_mutations, uniprot_id = x
    if pd.isnull(pdb_id) and pd.isnull(pdb_mutations):
        print('Not enough info: {}'.format(dict(x.items())))
        return np.nan, np.nan, np.nan, pdb_mutations
    try:
        return get_uniprot_id_mutation(pdb_id, None, pdb_mutations, uniprot_id)
    except pdb_tools.SIFTSError as e:
        print('{}: {}'.format(e, dict(x.items())))
        return np.nan, np.nan, np.nan, pdb_mutations


def get_uniprot_id_mutation_rosetta_ddg(x):
    """Wrapper around `get_uniprot_id_mutation` which catches Exceptions."""
    pdb_id, pdb_chains, pdb_mutations = x
    if pd.isnull(pdb_id) and pd.isnull(pdb_mutations):
        print('Not enough info: {}'.format(dict(x.items())))
        return np.nan, np.nan, pdb_chains, pdb_mutations
    try:
        return get_uniprot_id_mutation(pdb_id, pdb_chains, pdb_mutations, None)
    except pdb_tools.SIFTSError as e:
        print('{}: {}'.format(e, dict(x.items())))
        return np.nan, np.nan, pdb_chains, pdb_mutations
