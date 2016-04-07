# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 17:04:30 2015

@author: Alexey Strokach
"""
#%%

from __future__ import print_function

import os
import gzip
import urllib
import xml.etree.ElementTree as ET

import pandas as pd

from elaspic.structure_tools import AAA_DICT
from elaspic.conf import get_temp_dir

from common import blast_tools


#%% Functions used by ``make_ddg_training_set_core``
###################################################################################################

#blastp_libraries_path = '/home/kimlab1/strokach/databases/pdbfam/DB_Libraries_nr/'
blast_db_path = '/home/kimlab1/strokach/databases/pdbfam/libraries_all_together_db/libraries_all'

def mutation_in_sequence(mutation, sequence):
    if mutation is None or sequence is None:
        return False
    try:
        mutation_aa = mutation[0]
        mutation_pos = int(mutation[1:-1])
        sequence_aa = sequence[mutation_pos-1]
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
        domain_counter.update(range(start_pos, end_pos+1))

    return any([mutation in domain_counter for mutation in mutations])



blast_cache = dict()
def run_blast(uniprot_id, uniprot_sequence, domain_def):
    blast_cache_key = (uniprot_id.split('_')[0], domain_def)
    if blast_cache_key in blast_cache:
        return blast_cache[blast_cache_key]
    domain_start, domain_end = [int(x) for x in domain_def.split(':')]
    domain_sequence = uniprot_sequence[domain_start-1:domain_end]
    result_df, system_command = blast_tools.call_blast(domain_sequence, blast_db_path)
    blast_tools.annotate_blast_results(result_df, domain_start, len(domain_sequence))
    blast_cache[blast_cache_key] = (result_df, system_command)
    return result_df, system_command



def remove_domains_outside_mutation(result_df, mutation):
    result_df = (
        result_df[result_df['domain_def_new']
        .apply(lambda x: mutation_inside_domain([mutation, x]))].copy()
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
    if us[um_pos-1] == um[0]:
        return us[:um_pos-1] + um[-1] + us[um_pos:]
    else:
        raise Exception(us, um)



def get_seq_identity_df(uniprot_domain_df, domain_pair=False, reverse_mut=False):
    """
    .. note::
        I think this function is obsolete and not used anymore...
    """
    if not domain_pair:
        idx_column = 'uniprot_domain_id'
        duplicate_subset = []
    else:
        idx_column = 'uniprot_domain_pair_id'
        duplicate_subset = [
            'uniprot_domain_id_1', 'uniprot_domain_id_2', 'uniprot_id_1', 'uniprot_id_2',
            'uniprot_mutation'
        ]

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





###############################################################################
### Functions

pdb_to_uniprot_cache = {}

def get_uniprot_from_pdb___deprecated(pdb_id, pdb_resnum, pdb_aa, uniprot_id, cache_dir=None):
    """ This function is deprecated. Use get_sifts_data instead.
    """
    if cache_dir is None:
        cache_dir = get_temp_dir() + 'sifts/'

    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)

    sifts_filename = pdb_id.lower() + '.xml.gz'
    if not os.path.isfile(cache_dir + sifts_filename):
        request = urllib.Request('ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/{}.xml.gz'.format(pdb_id.lower()))
        response = urllib.urlopen(request)
        with open(cache_dir + sifts_filename, 'wb') as ofh:
            ofh.write(response.read())

    # buf = StringIO(response.read())
    # ifh = gzip.GzipFile(fileobj=buf)
    with gzip.open(cache_dir + sifts_filename) as ifh:
        root = ET.fromstring(ifh.read())
    uniprot_position = None
    uniprot_aa = None
    pdb_chain = None
    for entity in root:
        if entity.tag.split('}')[-1] == 'entity':
            pdb_chain = entity.attrib.get('entityId')
            for segment in entity:
                if segment.tag.split('}')[-1] == 'segment' \
                and int(segment.attrib.get('start')) <= pdb_resnum \
                and int(segment.attrib.get('end')) >= pdb_resnum:
                    for listResidue in segment:
                        if listResidue.tag.split('}')[-1] == 'listResidue':
                            for residue in listResidue:
                                if residue.tag.split('}')[-1] == 'residue' \
                                and int(residue.attrib.get('dbResNum')) == pdb_resnum:
                                    for crossRefDb in residue:
                                        if crossRefDb.tag.split('}')[-1] == 'crossRefDb' \
                                        and crossRefDb.attrib.get('dbSource') == 'UniProt':
                                            if pdb_aa and pdb_aa != crossRefDb.attrib.get('dbResName'):
                                                continue
                                            if not uniprot_id:
                                                uniprot_id = crossRefDb.attrib.get('dbAccessionId') == uniprot_id
                                            uniprot_position = crossRefDb.attrib.get('dbResNum')
                                            uniprot_aa = crossRefDb.attrib.get('dbResName')
                                            break
    return uniprot_position, uniprot_aa, pdb_chain



def get_sifts_data(pdb_id, cache_dict={}, cache_dir=None):
    """Download the xml file for the pdb file with the pdb id pdb_id, parse that
    xml file, and return a dictionry which maps pdb resnumbing to uniprot
    numbering for the chain specified by pdb_chain and uniprot specified by
    uniprot_id.
    """
    if pdb_id in cache_dict:
        return cache_dict[pdb_id]

    # Set up a directory to store sifts xml files
    if cache_dir is None:
        cache_dir = get_temp_dir() + 'sifts/'
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)

    # Download the sifts file if it is not in cache
    sifts_filename = pdb_id.lower() + '.xml.gz'
    if not os.path.isfile(cache_dir + sifts_filename):
        request = urllib.Request('ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/' + sifts_filename)
        response = urllib.urlopen(request)
        with open(cache_dir + sifts_filename, 'wb') as ofh:
            ofh.write(response.read())
        # buf = StringIO(response.read())
        # ifh = gzip.GzipFile(fileobj=buf)
        # root = ET.fromstring(ifh.read())

    # Go over the xml file and find all cross-references to uniprot
    pdb_sifts_data = []
    with gzip.open(cache_dir + sifts_filename) as ifh:
        root = ET.fromstring(ifh.read())
    for entity in root:
        # Go over different entries in SIFTS
        if entity.tag.split('}')[-1] == 'entity':
                # Go over different chain segments
                for segment in entity:
                    if segment.tag.split('}')[-1] == 'segment':
                        # Go over different lists of residues
                        for listResidue in segment:
                            if listResidue.tag.split('}')[-1] == 'listResidue':
                                # Go over each residue
                                for residue in listResidue:
                                    if residue.tag.split('}')[-1] == 'residue':
                                        # Go over all database crossreferences keeping only those
                                        # that come from uniprot and match the given uniprot_id
                                        residue_data = _fill_residue_data(pdb_id, residue)
                                        if residue_data is None:
                                            continue
                                        pdb_sifts_data.append(residue_data)

    # Convert data to a DataFrame and make sure we have no duplicates
    pdb_sifts_data_df = pd.DataFrame(pdb_sifts_data)
    assert len(pdb_sifts_data_df) == len(pdb_sifts_data_df.drop_duplicates(subset=['pdb_chain', 'resnum']))

    # TODO: should optimise the code above instead of simply removing duplicates
    # pdb_sifts_data_df = pdb_sifts_data_df.drop_duplicates()

    cache_dict[pdb_id] = pdb_sifts_data_df
    return pdb_sifts_data_df



def _fill_residue_data(pdb_id, residue_xml):
#    residue_field_names = [
#        'pdb_id', 'pdb_chain', 'resnum', 'pdb_aa',
#        'uniprot_id', 'uniprot_position', 'uniprot_aa', 'pfam_id']
#    residue_data = {x: None for x in residue_field_names}
    residue_data = {'is_observed': True, 'comments': []}

    for crossRefDb in residue_xml:
        # Some more details about the residue
        if crossRefDb.tag.split('}')[-1] == 'residueDetail':
            residue_data['comments'].append(crossRefDb.text)
            if crossRefDb.text == 'Not_Observed':
                residue_data['is_observed'] = False
        # Mappings to other databases
        if crossRefDb.tag.split('}')[-1] == 'crossRefDb':
            if (crossRefDb.attrib.get('dbSource') == 'PDB'):
                    residue_data['pdb_id'] = crossRefDb.attrib.get('dbAccessionId')
                    residue_data['pdb_chain'] = crossRefDb.attrib.get('dbChainId')
                    residue_data['resnum'] = crossRefDb.attrib.get('dbResNum')
                    resname = crossRefDb.attrib.get('dbResName')
                    if resname in AAA_DICT:
                        residue_data['pdb_aa'] = AAA_DICT[resname]
                    else:
                        print('Could not convert amino acid {} to a one letter code!')
                        residue_data['pdb_aa'] = resname
            if (crossRefDb.attrib.get('dbSource') == 'UniProt'):
                    residue_data['uniprot_id'] = crossRefDb.attrib.get('dbAccessionId')
                    residue_data['uniprot_position'] = crossRefDb.attrib.get('dbResNum')
                    residue_data['uniprot_aa'] = crossRefDb.attrib.get('dbResName')
            if (crossRefDb.attrib.get('dbSource') == 'Pfam'):
                    residue_data['pfam_id'] = crossRefDb.attrib.get('dbAccessionId')

    residue_data['comments'] = ','.join(residue_data['comments'])

    return residue_data



###################################################################################################
#%% Parse protherm ddg training set data
###################################################################################################

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
        'MK10_HUMAN': 'BRCA1_HUMAN', # BRCA1
    }

    uniprot_id_conversion = {
        'P69542': 'P69543',
        'P53779': 'P38398', # BRCA1
    }

    pdb_id_conversion = {
        'rf5v': '3f5v'
    }

    def __init__(self):
        self.data = []
        self.sifts_cache = {}
        self.DEBUG = False


    def parse(self, filename):
        with open(filename, 'r') as fh:
            file_data = fh.readlines()

        row_data = self.__get_empty_row_data()
        for line_number, line in enumerate(file_data):

            row = line.split()
            try:

                if row[0] == 'NO.':
                    row_data['protherm_no'] = int(row[-1])

                if row[0] == 'PROTEIN':
                    row_data['protein_name'] = ' '.join(row[1:])

                if row[0] == 'SWISSPROT_ID':
                    if len(row) > 1:
                        uniprot_name = row[1]
                        row_data['uniprot_name'] = self.uniprot_name_conversion.get(uniprot_name, uniprot_name)
                    if len(row) > 2:
                        uniprot_id = row[2].strip('()')
                        row_data['uniprot_id'] = self.uniprot_id_conversion.get(uniprot_id, uniprot_id)

                if row[0] == 'PDB_wild' and len(row) > 1:
                    pdb_id = row[-1].lower().strip('\x82')
                    row_data['pdb_id'] = self.pdb_id_conversion.get(pdb_id, pdb_id)

                if row[0] == 'MUTATION' and len(row) == 2 and row[1] == 'wild':
                    row_data['mutation'] = 'wild'

                if row[0] == 'MUTATION' and len(row) > 1:
                    row_data['mutation'] = ''.join(row[1:])

                    # Skip strange rows
                    if len(row) != 4:
                        raise Exception("Keeping only single mutation variants")
                    if all(row_data[x] is None for x in ['uniprot_id', 'pdb_id', 'protein_name']):
                        raise Exception("Skipping mutation with no protein_name, no uniprot_id and no pdb_id")
                    if len(row[-3]) > 1 or len(row[-1]) > 1:
                        raise Exception("Only considering single amino acid substitutions")

                    # Convert mutation to uniprot coordinates if mutation is with respect to the pdb
                    if all(row_data[key] is not None for key in ['pdb_id', 'mutation']):
                        self.__get_mutation_uniprot(row_data)

                if row[0] == 'REMARKS' and len(row) > 1:
                    row_data['remarks'] = ' '.join(row[1:])
                    # Sometimes remarks stretch out over multiple lines
                    i = 1
                    next_line = file_data[line_number+i]
                    while next_line[0] == ' ':
                        row_data['remarks'] += '; ' + next_line.strip()
                        i += 1
                        next_line = file_data[line_number+i]

                if row[0] in self.thermodynamic_parameters and len(row) > 1 and line[0] != ' ':
                    value = ''.join(row[1:]).replace(',', '')
                    if any(x in value for x in {'<', '>', 'Unknown', 'n.d.', 'NO_MOLECULE'}):
                        raise Exception("Cannot convert entry '{}' to float!".format(value))

                    try:
                        if 'kcal/mol' in value:
                            row_data[row[0]] = float(value.strip('kcal/mol'))
                        elif 'kJ/mol' in value:
                            row_data[row[0]] = 0.239001 * float(value.strip('kJ/mol'))
                        elif 'cal/mol' in value:
                            row_data[row[0]] = 0.001 * float(value.strip('cal/mol'))
                        elif 'kal/mol' in value:
                            row_data[row[0]] = 0.001 * float(value.strip('kal/mol'))
                        elif 'K' in value:
                             row_data[row[0]] = float(value.strip('K'))
                        else:
                            row_data[row[0]] = float(value)
                    except ValueError as e:
                        print(e)
                        print('Could not convert key {} value {} to float! Skipping...'.format(row[0], value))


            except Exception as e:
                error_message = 'Line: {}. Key: {}. Error: {}'.format(line_number, row[0], str(e))
                print("error_message: {}".format(error_message))
                print("row_data: {}".format(row_data))
                print()
                row_data['errors'].append(error_message)

            if row[0] == '//':
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

        try:
            # Get sifts data
            if row_data['pdb_id'] not in self.sifts_cache:
                sifts_df = get_sifts_data(row_data['pdb_id'])
                self.sifts_cache[row_data['pdb_id']] = sifts_df
            else:
                sifts_df = self.sifts_cache[row_data['pdb_id']]
        except Exception:
            raise Exception('Could not get SIFTS data!')

        # Sometimes sifts does not contain any uniprot information for the protein in question
        subset_columns = ['uniprot_position', 'uniprot_aa', 'pdb_chain']
        if len(set(subset_columns) - set(sifts_df.columns)):
            raise Exception(
                "SIFTS information missing for some of the required columns: {}"
                .format(set(subset_columns) - set(sifts_df.columns)))
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
                (sifts_df['residx'] == int(''.join(c for c in row_data['mutation'][1:-1] if c.isdigit()))) &
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
            (sifts_df['residx'] == int(''.join(c for c in row_data['mutation'][1:-1] if c.isdigit()))) &
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
            raise Exception("No rows in the sifts df match the inclusion criteria!")

        # Save the calculated values
        row_data['pdb_aa'] = sifts_df_subset['pdb_aa'].iloc[0]
        row_data['uniprot_aa'] = sifts_df_subset['uniprot_aa'].iloc[0]
        row_data['mutation_uniprot'] = '{}{}{}'.format(
            sifts_df_subset['pdb_aa'].iloc[0],
            sifts_df_subset['uniprot_position'].iloc[0],
            row_data['mutation'][-1])
        if 'uniprot_id' in sifts_df_subset.columns:
            row_data['uniprot_id'] = sifts_df_subset['uniprot_id'].iloc[0]



#%%
if __name__ == '__main__':
    import doctest
    doctest.testmod()


