import os
import os.path as op
import logging
import atexit
import tempfile

import pandas as pd

from common import blast_tools, pdb_tools


# %%
logger = logging.getLogger(__name__)


# %% Functions used by ``make_ddg_training_set_core``

blast_db_path = '/home/kimlab1/strokach/databases/pdbfam/libraries_all_together_db/libraries_all'


def mutation_in_sequence(mutation, sequence):
    if mutation is None or sequence is None:
        return False
    try:
        mutation_aa = mutation[0]
        mutation_pos = int(mutation[1:-1])
        sequence_aa = sequence[mutation_pos - 1]
    except Exception as e:
        print(str(e))
        print(mutation, sequence)
        print()
        return False
    return sequence_aa == mutation_aa


def mutation_inside_domain(x):
    """

    Examples
    --------

    >>> mutation_inside_domain(['A10B', '10:20'])
    True
    >>> mutation_inside_domain(['A9B', '10:20'])
    False

    >>> mutation_inside_domain(['A10B', '1:10'])
    True
    >>> mutation_inside_domain(['A11B', '1:10'])
    False

    >>> mutation_inside_domain(['A10B,A100B', '100:200'])
    True
    >>> mutation_inside_domain(['A10B,A99B', '100:200'])
    False
    >>> mutation_inside_domain(['A10B,A99B', '10:20,100:200'])
    True

    """
    mutations, domain_def = x

    if isinstance(mutations, str):
        mutations = mutations.split(',')
    if type(mutations) not in [list, tuple]:
        raise Exception('Unsupported type for ``mutations``: {}'.format(type(mutations)))

    for i in range(len(mutations)):
        if mutations[i][0].isdigit() and mutations[i][-1].isdigit():
            mutations[i] = int(mutations[i])
        else:
            mutations[i] = int(mutations[i][1:-1])

    domain_counter = set()
    for x in domain_def.split(','):
        start_pos, end_pos = [int(i) for i in x.split(':')]
        domain_counter.update(range(start_pos, end_pos + 1))

    return any([mutation in domain_counter for mutation in mutations])


blast_cache = dict()


def run_blast(uniprot_id, uniprot_sequence, domain_def):
    blast_cache_key = (uniprot_id.split('_')[0], domain_def)
    if blast_cache_key in blast_cache:
        return blast_cache[blast_cache_key]
    domain_start, domain_end = [int(x) for x in domain_def.split(':')]
    domain_sequence = uniprot_sequence[domain_start - 1:domain_end]
    result_df, system_command = blast_tools.call_blast(domain_sequence, blast_db_path)
    blast_tools.annotate_blast_results(result_df, domain_start, len(domain_sequence))
    blast_cache[blast_cache_key] = (result_df, system_command)
    return result_df, system_command


def remove_domains_outside_mutation(result_df, mutation):
    result_df = (
        result_df[
            result_df['domain_def_new']
            .apply(lambda x: mutation_inside_domain([mutation, x]))
        ].copy()
    )
    return result_df


stratify_by_pc_identity = [
    (100, lambda x: x >= 80),
    (80, lambda x: (x >= 60) & (x < 80)),
    (60, lambda x: (x >= 40) & (x < 60)),
    (40, lambda x: x < 40),
]


def stratify_results_by_identity(result_df):
    templates = []
    for prefix, fun in stratify_by_pc_identity:
        rows = result_df[result_df['pc_identity'].apply(fun) == True]
        if not len(rows):
            continue
        row = rows[:1].copy()
        row['max_seq_identity'] = prefix
        templates.append(row)
    templates_df = pd.concat(templates, ignore_index=True)
    return templates_df


def get_templates(x):
    """Find a list of templates for a given domain spanning different sequnece indentity bins

    Parameters
    ----------
    x : list
        [unique_id, uniprot_id, uniprot_mutation, domain_def, uniprot_sequence]
    y : int
        hello world
    x : str
        good bye world

    Returns
    -------
    pandas.DataFrame
        Dataframe that includes only those domains that contain a mutation
    """
    unique_id, uniprot_id, uniprot_mutation, domain_def, uniprot_sequence = x

    try:
        result_df, system_command = run_blast(uniprot_id, uniprot_sequence, domain_def)
        if result_df is None or len(result_df) == 0:
            raise Exception('No templates were found in the PDB database!')
        result_df['unique_id'] = unique_id

        blast_results_mutdom = remove_domains_outside_mutation(result_df, uniprot_mutation)
        if blast_results_mutdom is None or len(blast_results_mutdom) == 0:
            raise Exception('Templates that were found do not cover the site of the mutation!')

    except Exception as e:
        print(
            'An error occured!\n',
            e, '\n',
            uniprot_id, ' ', domain_def, ' ', uniprot_mutation, '\n',
            # uniprot_sequence, '\n',
            sep='',
        )
        return None

    return blast_results_mutdom


def mutate_sequence(x):
    us, um = x
    um_pos = int(um[1:-1])
    if us[um_pos - 1] == um[0]:
        return us[:um_pos - 1] + um[-1] + us[um_pos:]
    else:
        raise Exception(us, um)


def get_seq_identity_df(uniprot_domain_df, domain_pair=False, reverse_mut=False):
    """
    .. note::
        I think this function is obsolete and not used anymore...
    """
    if not domain_pair:
        idx_column = 'uniprot_domain_id'
#        duplicate_subset = []
    else:
        idx_column = 'uniprot_domain_pair_id'
#        duplicate_subset = [
#            'uniprot_domain_id_1', 'uniprot_domain_id_2', 'uniprot_id_1', 'uniprot_id_2',
#            'uniprot_mutation'
#        ]

    seq_identity_40 = uniprot_domain_df[[idx_column]].copy()
    seq_identity_40['max_seq_identity'] = 40
    seq_identity_60 = uniprot_domain_df[[idx_column]].copy()
    seq_identity_60['max_seq_identity'] = 60
    seq_identity_80 = uniprot_domain_df[[idx_column]].copy()
    seq_identity_80['max_seq_identity'] = 80
    seq_identity_100 = uniprot_domain_df[[idx_column]].copy()
    seq_identity_100['max_seq_identity'] = 100
    seq_identity = pd.concat(
        [seq_identity_40, seq_identity_60, seq_identity_80, seq_identity_100],
        ignore_index=True)

    return seq_identity


# %% Parse protherm ddg training set data

class ParseProtherm():
    """
    """
    thermodynamic_parameters = [
        'dG_H2O', 'dG', 'Tm', 'ddG_H2O', 'ddG', 'dTm', 'dHvH'
    ]

    column_names = [
        'errors', 'protherm_no', 'pdb_id', 'protein_name', 'uniprot_name', 'uniprot_id',
        'mutated_pdb_chain', 'mutation', 'mutation_uniprot'
    ] + thermodynamic_parameters

    uniprot_name_conversion = {
        'MK10_HUMAN': 'BRCA1_HUMAN',  # BRCA1
    }

    uniprot_id_conversion = {
        'P69542': 'P69543',
        'P53779': 'P38398',  # BRCA1
    }

    pdb_id_conversion = {
        'érf5v': '3f5v',
        '1bgl': '4v40',
    }

    def __init__(self):
        self.data = []
        self.sifts_cache = {}
        self.sifts_cache_dir = (
            tempfile.TemporaryDirectory(dir=op.join(tempfile.gettempdir(), 'sifts'))
        )
        atexit.register(self.sifts_cache_dir.cleanup)

    def _parse(self, line, row_data, file_data, line_number):
        """
        Parameters
        ----------
        line : str
            ...
        row_data : list
            ...
        file_data : list
            Processed contents of the Protherm file
        line_number : int
            Line number that we are currently analysing
            (we need to look ahead sometimes...)
        """
        row = line.split()

        if row[0] == 'NO.':
            row_data['protherm_no'] = int(row[-1])

        elif row[0] == 'PROTEIN':
            row_data['protein_name'] = ' '.join(row[1:])

        elif row[0] == 'SWISSPROT_ID':
            if len(row) > 1:
                uniprot_name = row[1]
                row_data['uniprot_name'] = (
                    self.uniprot_name_conversion.get(uniprot_name, uniprot_name)
                )
            if len(row) > 2:
                uniprot_id = row[2].strip('()')
                row_data['uniprot_id'] = (
                    self.uniprot_id_conversion.get(uniprot_id, uniprot_id)
                )

        elif row[0] == 'PDB_wild' and len(row) > 1:
            pdb_id = row[-1].lower()
            row_data['pdb_id'] = self.pdb_id_conversion.get(pdb_id, pdb_id)

        elif row[0] == 'MUTATION' and len(row) == 2 and row[1] == 'wild':
            row_data['mutation'] = 'wild'

        elif row[0] == 'MUTATION' and len(row) > 1:
            row_data['mutation'] = ''.join(row[1:])

            # Skip strange rows
            if len(row) != 4:
                raise Exception("Keeping only single mutation variants.")
            if all(row_data[x] is None for x in ['uniprot_id', 'pdb_id', 'protein_name']):
                raise Exception(
                    "Skipping mutation with no protein_name, no uniprot_id and no pdb_id.")
            if len(row[-3]) > 1 or len(row[-1]) > 1:
                raise Exception("Only considering single amino acid substitutions.")

            # Convert mutation to uniprot coordinates if mutation is with respect to pdb
            if all(row_data[key] is not None for key in ['pdb_id', 'mutation']):
                self.__get_mutation_uniprot(row_data)

        elif row[0] == 'REMARKS' and len(row) > 1:
            row_data['remarks'] = ' '.join(row[1:])
            # Sometimes remarks stretch out over multiple lines
            i = 1
            next_line = file_data[line_number + i]
            while next_line[0] == ' ':
                row_data['remarks'] += '; ' + next_line.strip()
                i += 1
                next_line = file_data[line_number + i]

        elif (row[0] in self.thermodynamic_parameters and len(row) > 1 and line[0] != ' '):
            # Skip bad values
            value = ''.join(row[1:])
            if any(v in value for v in ['<', '>', 'Unknown', 'n.d.', 'NO_MOLECULE', 'dimer']):
                raise Exception(
                    'Cannot convert entry "{}" to float because it contains '
                    'a blacklisted character'.format(value)
                )
            # Clean value
            value = value.replace(',', '').replace('/K', '').lower()
            # Convert to float
            conversion = {
                '': 1,
                'kcal/mol': 1,
                'kcal/mole': 1,
                'cal/mol': 0.001,
                'cal/mole': 0.001,
                'kal/mol': 0.001,
                'kal/mole': 0.001,
                'kj/mol': 0.239001,
                'kj/mole': 0.239001,
            }
            value = value
            for suffix, cf in conversion.items():
                try:
                    row_data[row[0]] = cf * float(value.strip(suffix))
                    break
                except ValueError:
                    pass
            # Report errors
            if row_data[row[0]] is None:
                error_message = (
                    "Could not convert key {} value {} to float! Skipping..."
                    .format(row[0], value)
                )
                logger.warning(error_message)

    def parse(self, filename):
        with open(filename, 'r') as fh:
            file_data = fh.readlines()

        row_data = self.__get_empty_row_data()
        for line_number, line in enumerate(file_data):

            try:
                self._parse(line, row_data, file_data, line_number)
            except Exception as e:
                error_info = (
                    'Error type: "{}", Error Message: "{}", Line: "{}"'
                    .format(type(e), str(e), '{}: {}'.format(line_number, line.strip()))
                )
                error_message = error_info + ', row_data: "{}"\n\n'.format(row_data)
                if not str(e).startswith('Keeping only single mutation'):
                    logger.error(error_message)
                row_data['errors'].append(error_info)

            if line.startswith('//'):
                row_data = self.__flush_row_data(row_data)

    def __get_empty_row_data(self):
        row_data = {name: None for name in self.column_names}
        row_data['errors'] = []
        return row_data

    def __flush_row_data(self, row_data):
        row_data['errors'] = ';'.join(row_data['errors'])
        self.data.append(row_data)
        return self.__get_empty_row_data()

    def __get_mutation_uniprot(self, row_data):

        # try:
        # Get sifts data
        if row_data['pdb_id'] not in self.sifts_cache:
            sifts_df = pdb_tools.get_sifts_data(row_data['pdb_id'], self.sifts_cache_dir.name)
            self.sifts_cache[row_data['pdb_id']] = sifts_df
        else:
            sifts_df = self.sifts_cache[row_data['pdb_id']]
        # except Exception:
        #     raise Exception('Could not get SIFTS data!')

        # Sometimes sifts does not contain any uniprot information for the protein in question
        subset_columns = ['uniprot_position', 'uniprot_aa', 'pdb_chain']
        missing_columns = sorted(set(subset_columns) - set(sifts_df.columns))
        if missing_columns == ['uniprot_aa', 'uniprot_position']:
            raise Exception('SIFTS has no UniProt annotation for this protein')
        elif missing_columns:
            raise Exception("SIFTS information missing: {}".format(missing_columns))
        sifts_df = sifts_df.dropna(subset=subset_columns)

        # residx counts the index of the residue in the pdb, starting from 1
        residx = []
        for pdb_chain in set(sifts_df['pdb_chain']):
            residx.extend(range(1, len(sifts_df[sifts_df['pdb_chain'] == pdb_chain]) + 1))
        sifts_df['residx'] = residx

        if 'uniprot_id' in sifts_df.columns:
            # Get the subset of rows that we are interested in
            sifts_df_subset_0 = sifts_df[
                (sifts_df['pdb_id'] == row_data['pdb_id']) &
                (sifts_df['resnum'] == row_data['mutation'][1:-1]) &
                (sifts_df['pdb_aa'] == row_data['mutation'][0]) &
                (sifts_df['uniprot_id'] == row_data['uniprot_id'])]
            # Try using residx instead of resnum
            sifts_df_subset_2 = sifts_df[
                (sifts_df['pdb_id'] == row_data['pdb_id']) &
                (sifts_df['residx'] == int(
                    ''.join(c for c in row_data['mutation'][1:-1] if c.isdigit()))) &
                (sifts_df['pdb_aa'] == row_data['mutation'][0]) &
                (sifts_df['uniprot_id'] == row_data['uniprot_id'])]
        else:
            sifts_df_subset_0 = []
            sifts_df_subset_2 = []
        # Try mapping to a wildcard uniprot
        sifts_df_subset_1 = sifts_df[
            (sifts_df['pdb_id'] == row_data['pdb_id']) &
            (sifts_df['resnum'] == row_data['mutation'][1:-1]) &
            (sifts_df['pdb_aa'] == row_data['mutation'][0])]
        # Or residx and wildcard uniprot
        sifts_df_subset_3 = sifts_df[
            (sifts_df['pdb_id'] == row_data['pdb_id']) &
            (sifts_df['residx'] == int(
                ''.join(c for c in row_data['mutation'][1:-1] if c.isdigit()))) &
            (sifts_df['pdb_aa'] == row_data['mutation'][0])]
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
            raise Exception("SIFTS failed to match residue")

        # Save the calculated values
        row_data['pdb_aa'] = sifts_df_subset['pdb_aa'].iloc[0]
        row_data['uniprot_aa'] = sifts_df_subset['uniprot_aa'].iloc[0]
        row_data['mutation_uniprot'] = '{}{}{}'.format(
            sifts_df_subset['pdb_aa'].iloc[0],
            sifts_df_subset['uniprot_position'].iloc[0],
            row_data['mutation'][-1])
        if 'uniprot_id' in sifts_df_subset.columns:
            row_data['uniprot_id'] = sifts_df_subset['uniprot_id'].iloc[0]


# %%
if __name__ == '__main__':
    import doctest
    doctest.testmod()
