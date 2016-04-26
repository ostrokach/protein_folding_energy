# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:25:11 2014

@author: Alexey Strokach


TODO: There's some dirty business with the duplication of domains in the `uniprot_domain` table.
TODO: There's also some dirty business with some domains having templates with sequence identity 
        in range 0 <= x <= 1, and others having templates with sequence identity in range 1 <= x <= 100.
"""

#%%
from __future__ import print_function

import os
import re
import cPickle as pickle
import subprocess
import datetime

from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import sqlalchemy as sa

import matplotlib.pyplot as plt
import seaborn as sns

from common import constants
from elaspic_tools.mutation_sets import version_suffix, data_path, parsers



#%% Set parameters
MAKE_PLOTS = False
pd.options.mode.chained_assignment = None
sns.set_context("poster")
sns.set_style('whitegrid')

subprocess.check_call('mkdir -p ' + constants.protherm_data_path + 'parsed_data{}'.format(version_suffix), shell=True)



###################################################################################################
#%% PART 1 - LOAD DATA FROM SMALL-SCALE STUDIES
###################################################################################################

#%% Create the required database and tables if they don't exist already
#    create database elaspic_training;
#
#    create view elaspic_training.domain as select * from elaspic.domain;
#    create view elaspic_training.domain_contact as select * from elaspic.domain_contact;
#    create view elaspic_training.provean as select * from elaspic.provean;
#
#    create table elaspic_training.uniprot_domain like elaspic.uniprot_domain;
#    create table elaspic_training.uniprot_domain_template like elaspic.uniprot_domain_template;
#    create table elaspic_training.uniprot_domain_model like elaspic.uniprot_domain_model;
#    create table elaspic_training.uniprot_domain_mutation like elaspic.uniprot_domain_mutation;
#
#    create table elaspic_training.uniprot_domain_pair like elaspic.uniprot_domain_pair;
#    create table elaspic_training.uniprot_domain_pair_template like elaspic.uniprot_domain_pair_template;
#    create table elaspic_training.uniprot_domain_pair_model like elaspic.uniprot_domain_pair_model;
#    create table elaspic_training.uniprot_domain_pair_mutation like elaspic.uniprot_domain_pair_mutation;
#
#    ALTER TABLE `elaspic_training`.`uniprot_domain`
#    ADD COLUMN `max_seq_identity` VARCHAR(45) NULL AFTER `path_to_data`;



###################################################################################################
#%% Load data from small-scale studies

#%% Allali-Hassani et. al. 2009
#    A survey of proteins encoded by non-synonymous single nucleotide
#    polymorphisms reveals a significant fraction with altered stability
#    and activity
#    Abdellah ALLALI-HASSANI et al.


# Load required data from text files and the database
abdellah_et_al = pd.read_csv(constants.local_database_path + 'mutations/ddg/small-scale/abdellah_et_al.tsv', sep='\t')
abdellah_et_al['remarks'] = 'Allali-Hassani et.al. Biochem. J. 2009'
abdellah_et_al['gene_name'] = abdellah_et_al['Protein'].apply(lambda x: x.split('_')[0].lower())
abdellah_et_al_mut = abdellah_et_al.dropna(subset=['dbSNP']).copy()
abdellah_et_al_mut['uniprot_mutation'] = abdellah_et_al_mut['Protein'].apply(lambda x: x.split('_')[-1].upper())
abdellah_et_al_mut['dTm_median'] = abdellah_et_al_mut['ΔTagg']

engine = sa.create_engine('mysql://elaspic:elaspic@192.168.6.19/')
sql_query = """
select *
from mutation.ensembl_76_missense_variants_all_scores
join uniprot_kb.uniprot_sequence using (uniprot_id)
where variation_name in ('{}') ;
""".format("', '".join(abdellah_et_al_mut['dbSNP'].values))
variations = pd.read_sql_query(sql_query, engine)
del variations['gene_name']
del variations['uniprot_mutation']

sql_query = """
select *
from uniprot_kb.uniprot_identifier
join uniprot_kb.uniprot_sequence using (uniprot_id)
where identifier_id in ('{0}')
and db = 'sp' ;
""".format("', '".join(set(abdellah_et_al_mut['gene_name'].values)))
sequences_1 = pd.read_sql_query(sql_query, engine)
sequences_1['gene_name'] = sequences_1['identifier_id'].str.lower()

sql_query = """
select *
from uniprot_kb.uniprot_sequence
where gene_name in ('{0}')
and db = 'sp' ;
""".format("', '".join(set(abdellah_et_al_mut['gene_name'].values)))
sequences_2 = pd.read_sql_query(sql_query, engine)
sequences_2['gene_name'] = sequences_2['gene_name'].str.lower()



#%% Map data to uniprot id and mutation
abdellah_et_al_up_mut = pd.concat([
        abdellah_et_al_mut.merge(variations, left_on=['dbSNP'], right_on=['variation_name']),
        abdellah_et_al_mut.merge(sequences_1, on='gene_name'),
        abdellah_et_al_mut.merge(sequences_2, on='gene_name'),
    ], ignore_index=True)

abdellah_et_al_up_mut['mutation_in_sequence'] = [
    parsers.mutation_in_sequence(*x) for x in
    abdellah_et_al_up_mut[['uniprot_mutation', 'uniprot_sequence']].values]
abdellah_et_al_up_mut['is_splice_variant'] = abdellah_et_al_up_mut['uniprot_id'].str.contains('-')
abdellah_et_al_up_mut = abdellah_et_al_up_mut[
    (abdellah_et_al_up_mut['mutation_in_sequence']) &
    (~abdellah_et_al_up_mut['is_splice_variant'])]
abdellah_et_al_up_mut = abdellah_et_al_up_mut.drop_duplicates(subset=['uniprot_id', 'uniprot_mutation'])



#%% Save results
abdellah_et_al_up_mut.to_pickle(
    constants.protherm_data_path + 'parsed_data{}/small_studies_df.pickle'.format(version_suffix)
)



###################################################################################################
#%% PART 2 - LOAD PROTHERM DATA
###################################################################################################

protherm_filename_full = constants.local_database_path + 'mutations/ddg/protherm/ProTherm.dat'

cwd = os.getcwd()
os.chdir(constants.working_path + 'elaspic_tools/mutation_sets')


#%% Create a file that will parse the protherm database dump (WARNING: It may be memory-intensive!)
#    %%file _parse_protherm.py
#    import os
#    import sys
#    import cPickle as pickle
#    from tempfile import NamedTemporaryFile
#    import multiprocessing as mp
#
#    _ROOT = os.path.abspath(os.path.dirname(__file__))
#    sys.path.insert(0, os.path.join(_ROOT, "."))
#
#    from common import constants
#    import ddg_parsers
#
#    n_cores = 8
#    path_to_data = '/home/kimlab1/strokach/databases/'
#    protherm_filename_full = path_to_data + 'mutations/ddg/protherm/ProTherm.dat'
#
#    with open(protherm_filename_full) as ifh:
#        file_data = ifh.readlines()
#
#    chunk_size = len(file_data) / float(n_cores)
#    chunk_idxs = [0]
#    for i in range(n_cores - 1):
#        idx = int(chunk_size * (i + 1))
#        while not file_data[idx].startswith('//'):
#            idx += 1
#        chunk_idxs.append(idx+1)
#    chunk_idxs.append(len(file_data))
#
#    def worker(chunk_filename):
#        parse_protherm = parsers.ParseProtherm()
#        parse_protherm.parse(chunk_filename)
#        protherm_data = parse_protherm.data
#        print('Almost done {}'.format(i))
#        pickle.dump(protherm_data, open(chunk_filename + '.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
#        print('Done {}'.format(i))
#
#    chunk_filenames = []
#    for i in range(n_cores):
#        chunk_file = NamedTemporaryFile(delete=False)
#        chunk_file.writelines(file_data[chunk_idxs[i]:chunk_idxs[i+1]])
#        chunk_file.seek(0)
#        chunk_filenames.append(chunk_file.name)
#
#    jobs = []
#    for chunk_filename in chunk_filenames:
#        p = mp.Process(target=worker, args=(chunk_filename,))
#        p.start()
#        jobs.append(p)
#
#    for j in jobs:
#        j.join()
#
#    protherm_data = []
#    for chunk_filename in chunk_filenames:
#        protherm_data_chunk = pickle.load(open(chunk_filename + '.pickle'))
#        protherm_data.extend(protherm_data_chunk)
#        os.remove(chunk_filename)
#        os.remove(chunk_filename + '.pickle')
#
#    pickle.dump(protherm_data, open(protherm_filename_full + '.parsed.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
#


##%% Run the file created above, and delete it if everything finishes successfully
#    %run _parse_protherm.py
#    rm _parse_protherm.py


##%% Load results
#    protherm_dict = pickle.load(open(protherm_filename_full + '.parsed.pickle'))


###################################################################################################
#%% Parse ProTherm.dat in parallel

#%% Stuff for parallel processing
from IPython.parallel import Client
rc = Client()
dview = rc[:]
print('Connected to {} workers'.format(len(rc)))

async_result = dview.execute('%load_ext autoreload', silent=False, block=True)
print(''.join(async_result.stdout))
async_result = dview.execute('%autoreload 2', silent=False, block=True)
print(''.join(async_result.stdout))


# Divide one big file into many small ones, taking care to create breaks at appropriate positions
def divide_protherm_file(protherm_filename_full, num_chunks):
    with open(protherm_filename_full) as ifh:
        file_data = ifh.readlines()

    chunk_size = len(file_data) / float(num_chunks)
    chunk_idxs = [0]
    for i in range(num_chunks - 1):
        idx = int(chunk_size * (i + 1))
        while not file_data[idx].startswith('//'):
            idx += 1
        chunk_idxs.append(idx+1)
    chunk_idxs.append(len(file_data))

    chunk_filenames = []
    for i in range(num_chunks):
        chunk_file = NamedTemporaryFile(delete=False)
        chunk_file.writelines(file_data[chunk_idxs[i]:chunk_idxs[i+1]])
        chunk_file.seek(0)
        chunk_filenames.append(chunk_file.name)
    return chunk_filenames

# Define worker for parallel processing
def worker(chunk_filename):
    parse_protherm = parsers.ParseProtherm()
    parse_protherm.parse(chunk_filename)
    protherm_data = parse_protherm.data
    return protherm_data


# Magic
async_result = dview.execute('from elaspic_tools.mutation_sets import parsers', silent=False, block=True)
print(''.join(async_result.stdout))
chunk_filenames = divide_protherm_file(protherm_filename_full, len(rc))

protherm_dict = [x for xx in dview.map_sync(worker, chunk_filenames) for x in xx]
pickle.dump(protherm_dict, open(protherm_filename_full + '.parsed.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)


###################################################################################################
#%%% Load parsed Protherm data
protherm_dict = pickle.load(open(protherm_filename_full + '.parsed.pickle'))
protherm_df = pd.DataFrame(protherm_dict)
protherm_df = protherm_df.rename(columns={'mutation_uniprot': 'uniprot_mutation'})
protherm_df['uniprot_mutation'] = (
    protherm_df[['pdb_aa', 'uniprot_mutation']]
    .apply(lambda x: (x[0][0] + x[1][1:]) if (pd.notnull(x[0]) and pd.notnull(x[1])) else np.nan, axis=1)
)


#%% Manually correct some errors
# Acylphosphatase
protherm_df.loc[
    (pd.isnull(protherm_df['uniprot_name'])) &
    (protherm_df['protein_name'] == 'Acylphosphatase'),
    'uniprot_name'] = 'ACYP1_HUMAN'
protherm_df.loc[
    (pd.isnull(protherm_df['uniprot_id'])) &
    (protherm_df['protein_name'] == 'Acylphosphatase'),
    'uniprot_id'] = 'P07311'
mutations = protherm_df[
    (pd.notnull(protherm_df['mutation'])) &
    (protherm_df['protein_name'] == 'Acylphosphatase')]
protherm_df.loc[
    (pd.notnull(protherm_df['mutation'])) &
    (protherm_df['protein_name'] == 'Acylphosphatase'),
    'uniprot_mutation'] = mutations


# Alkaline phosphatase
protherm_df.loc[
    (pd.isnull(protherm_df['uniprot_name'])) &
    (protherm_df['protein_name'] == 'Alkaline phosphatase'),
    'uniprot_name'] = 'PPB_YEAST'
protherm_df.loc[
    (pd.isnull(protherm_df['uniprot_id'])) &
    (protherm_df['protein_name'] == 'Alkaline phosphatase'),
    'uniprot_id'] = 'P11491'


# Arginine kinase
protherm_df.loc[
    (pd.isnull(protherm_df['uniprot_name'])) &
    (protherm_df['protein_name'] == 'Arginine kinase'),
    'uniprot_name'] = 'KARG_DROME'
protherm_df.loc[
    (pd.isnull(protherm_df['uniprot_id'])) &
    (protherm_df['protein_name'] == 'Arginine kinase'),
    'uniprot_id'] = 'P48610'


# Eglin C
protherm_df.loc[
    (pd.isnull(protherm_df['uniprot_name'])) &
    (protherm_df['protein_name'] == 'Eglin C'),
    'uniprot_name'] = 'ICIC_HIRME'
protherm_df.loc[
    (pd.isnull(protherm_df['uniprot_id'])) &
    (protherm_df['protein_name'] == 'Eglin C'),
    'uniprot_id'] = 'P01051'



#%% Format error fields

# Select ddG_H2O if it is availible, else select ddG
protherm_df['ddg_best'] = [x[0] if pd.notnull(x[0]) else x[1] for x in protherm_df[['ddG_H2O', 'ddG']].values]

# Reverse the sign because protherm defines ddG and dTm as (mutant) - (wildtype)
# http://www.abren.net/protherm/protherm_knownproblems.php
change_columns = ['dTm', 'ddG', 'ddG_H2O', 'ddg_best']
for column in change_columns:
    protherm_df[column] = -protherm_df[column]


def format_errors(error_field):
    if not error_field:
        return ''
    error_messages = []
    error_subfields = [e.strip().strip(';') for e in error_field.split(':')]
    for i in range(len(error_subfields)-1):
        if error_subfields[i].endswith('Error'):
            error_message = error_subfields[i+1].rstrip('Line').strip(';').strip()
            if re.search('Cannot convert entry .* to float', error_message):
                error_message = 'Cannot convert entry to float'
            if error_message not in error_messages:
                error_messages.append(error_message)
    return '; ' + '; '.join(error_messages)


def classify_erros(error_messages):
    if not error_messages:
        return [0, 'Mapped successfully']
    if 'Wild-type protein' in error_messages:
        return [1, 'Wild-type']
    if 'No ddG and dTm score provided' in error_messages:
        return [1, 'No ddG and dTm scores']
    if ('Keeping only single mutation variants' in error_messages or
        'Only considering single amino acid substitutions' in error_messages):
            return [2, 'Multiple mutations']
    return [10, 'Mapping error']


def get_idx_where_better_exists(protherm_df):
    """Remove cases where the same uniprot_id-uniprot_mutation pair exists in a row with and without remarks
    """
    uniprot_mutation_no_remarks_ddg = set(
        protherm_df[
            pd.isnull(protherm_df['remarks']) &
            pd.notnull(protherm_df['ddg_best'])
        ][['uniprot_id', 'uniprot_mutation']].apply(tuple, axis=1)
    )
    uniprot_mutation_no_remarks_dtm = set(
        protherm_df[
            pd.isnull(protherm_df['remarks']) &
            pd.notnull(protherm_df['dTm'])
        ][['uniprot_id', 'uniprot_mutation']].apply(tuple, axis=1)
    )

    index_bad_ddg = protherm_df[
        (pd.notnull(protherm_df['remarks'])) &
        (pd.notnull(protherm_df['ddg_best'])) &
        (protherm_df[['uniprot_id', 'uniprot_mutation']]
            .apply(tuple, axis=1)
            .isin(uniprot_mutation_no_remarks_ddg))
    ].index

    index_bad_dtm = protherm_df[
        (pd.notnull(protherm_df['remarks'])) &
        (pd.isnull(protherm_df['ddg_best'])) &
        (pd.notnull(protherm_df['dTm'])) &
        (protherm_df[['uniprot_id', 'uniprot_mutation']]
            .apply(tuple, axis=1)
            .isin(uniprot_mutation_no_remarks_dtm))
    ].index

    return list(set(index_bad_ddg) | set(index_bad_dtm))


protherm_df['error_messages'] = protherm_df['errors'].apply(format_errors)

protherm_df['error_messages'] = (
    protherm_df['error_messages']
    .where(protherm_df['mutation'] != 'wild',
           protherm_df['error_messages'] + '; Wild-type protein')
)
protherm_df['error_messages'] = (
    protherm_df['error_messages']
    .where(pd.notnull(protherm_df['ddg_best']) | pd.notnull(protherm_df['dTm']),
           protherm_df['error_messages'] + '; No ddG and dTm score provided')
)
protherm_df['error_messages'] = (
    protherm_df['error_messages']
    .where(pd.notnull(protherm_df['uniprot_id']),
           protherm_df['error_messages'] + '; Uniprot id is missing')
)
protherm_df['error_messages'] = (
    protherm_df['error_messages']
    .where(pd.notnull(protherm_df['uniprot_mutation']),
           protherm_df['error_messages'] + '; Uniprot mutation is missing')
)

idxs_to_drop = get_idx_where_better_exists(protherm_df)
protherm_df.loc[idxs_to_drop, 'error_messages'] = (
    protherm_df.loc[idxs_to_drop, 'error_messages'] + '; A better version of this mutation exists'
)

# Check some of the `idxs_to_drop` mutations
#protherm_df.loc[16389, ['uniprot_id', 'uniprot_mutation', 'ddg_best', 'dTm', 'remarks', 'error_messages']]
#protherm_df[
#    (protherm_df['uniprot_id'] == 'P00720') &
#    (protherm_df['uniprot_mutation'] == 'V111A')
#][['uniprot_id', 'uniprot_mutation', 'ddg_best', 'dTm', 'remarks', 'error_messages']]

protherm_df['error_code'], protherm_df['error_category'] = zip(*protherm_df['error_messages'].apply(classify_erros))


#%% The same protein-mutation pair may occur multiple times
protherm_df_good = (
    protherm_df[
        (protherm_df['mutation'] != 'wild') &
        (protherm_df['error_messages'] == '')
    ]
)

# Average the thermidynamic parameters over all occurances of the same protein-mutation
data_columns = ['Tm', 'dG', 'dG_H2O', 'dHvH', 'dTm', 'ddG', 'ddG_H2O', 'ddg_best']
protherm_df_good_gp = protherm_df_good.groupby(['uniprot_id', 'uniprot_mutation'])
protherm_df_good_unique = (
    protherm_df_good_gp
    .agg(tuple)
    .merge(
        protherm_df_good_gp[data_columns]
        .agg(np.nanmean)
        .rename(columns=lambda c: c if c not in data_columns else c + '_mean'),
        left_index=True, right_index=True)
    .merge(
        protherm_df_good_gp[data_columns]
        .agg(np.nanstd)
        .rename(columns=lambda c: c if c not in data_columns else c + '_std'),
        left_index=True, right_index=True)
    .merge(
        protherm_df_good_gp[data_columns]
        .agg(np.nanmedian)
        .rename(columns=lambda c: c if c not in data_columns else c + '_median'),
        left_index=True, right_index=True)
    .merge(
        protherm_df_good_gp['mutation']
        .agg({'count': len}),
        left_index=True, right_index=True)
    .merge(
        protherm_df_good_gp
        .agg({'remarks': lambda x: tuple(x)}),
        left_index=True, right_index=True)
    .reset_index()
)

# Use ddG_H2O if it is availible, else ddG
def get_first_not_null(row):
    for value in row:
        if pd.notnull(value):
            return value
    return np.nan

protherm_df_good_unique['ddg_all_mean'] = (
    protherm_df_good_unique[['ddG_H2O_mean', 'ddG_mean']].apply(get_first_not_null, axis=1)
)
protherm_df_good_unique['ddg_all_median'] = (
    protherm_df_good_unique[['ddG_H2O_median', 'ddG_median']].apply(get_first_not_null, axis=1)
)
protherm_df_good_unique['ddg_all_std'] = (
    protherm_df_good_unique[['ddG_H2O_std', 'ddG_std']].apply(get_first_not_null, axis=1)
)

assert sum(protherm_df_good_unique.duplicated(subset=['uniprot_id', 'uniprot_mutation'])) == 0



###################################################################################################
#%% Add sequence information to each mutated protein
engine = sa.create_engine('mysql://elaspic:elaspic@192.168.6.19/uniprot_kb')
sql_query = """
select *
from uniprot_kb.uniprot_sequence
where uniprot_id in ('{}') ;
""".format("', '".join(protherm_df_good_unique['uniprot_id'].drop_duplicates()))
uniprot_sequences = pd.read_sql_query(sql_query, engine)

protherm_df_good_unique_wseq = protherm_df_good_unique.merge(uniprot_sequences, on=['uniprot_id'])
protherm_df_good_unique_wseq['sequence_match'] = [
    parsers.mutation_in_sequence(*x) for x
    in protherm_df_good_unique_wseq[['uniprot_mutation', 'uniprot_sequence']].values]
assert all(protherm_df_good_unique_wseq['sequence_match'])



#%%
protherm_df.to_pickle(
    constants.protherm_data_path + 'parsed_data{}/protherm_df.pickle'.format(version_suffix))
protherm_df_good.to_pickle(
    constants.protherm_data_path + 'parsed_data{}/protherm_df_good.pickle'.format(version_suffix))
protherm_df_good_unique.to_pickle(
    constants.protherm_data_path + 'parsed_data{}/protherm_df_good_unique.pickle'.format(version_suffix))
protherm_df_good_unique_wseq.to_pickle(
    constants.protherm_data_path + 'parsed_data{}/protherm_df_good_unique_wseq.pickle'.format(version_suffix))



###################################################################################################
#%% PART 3 - JOIN PROTHERM AND SMALL-SCALE STUDIES
###################################################################################################

#%%
abdellah_et_al_up_mut = pd.read_pickle(
    constants.protherm_data_path + 'parsed_data{}/small_studies_df.pickle'.format(version_suffix)
)
protherm_df_good_unique_wseq = pd.read_pickle(
    constants.protherm_data_path + 'parsed_data{}/protherm_df_good_unique_wseq.pickle'.format(version_suffix)
)


#%% Concatenate protherm data with data from small-scale studies
core_mut_wseq = pd.concat([protherm_df_good_unique_wseq, abdellah_et_al_up_mut], ignore_index=True)


#%% Add domain information to each mutated protein
engine = sa.create_engine('mysql://elaspic:elaspic@192.168.6.19/elaspic')
sql_query = """
select *
from elaspic.uniprot_domain ud
join elaspic.uniprot_domain_template udt using (uniprot_domain_id)
where uniprot_id in ('{}') ;
""".format("', '".join(protherm_df_good_unique_wseq['uniprot_id'].drop_duplicates()))
elaspic_domains = pd.read_sql_query(sql_query, engine)

core_mut_wseq_wdom = core_mut_wseq.merge(elaspic_domains, left_on=['uniprot_id'], right_on=['uniprot_id'])
core_mut_wseq_wdom['mutation_inside_domain'] = (
    core_mut_wseq_wdom[['uniprot_mutation', 'domain_def']]
    .apply(parsers.mutation_inside_domain, axis=1, raw=True)
)
core_mut_wseq_wdom = core_mut_wseq_wdom[core_mut_wseq_wdom['mutation_inside_domain']]



#%% Duplicate uniprot_domain columns for different sequence identity thresholds

### IMPORTANT!!! ORGANISM NAME MUST BE SET TO 'training' IN ORDER TO KEEP THE REAL AND TRAINING DATA SEGREGATED

# TODO: This should be done after fining templates!!!

### Mutation from wild-type to mutant
uniprot_domain_wt = core_mut_wseq_wdom.drop_duplicates(subset=['uniprot_domain_id', 'uniprot_id', 'uniprot_mutation']).copy()
uniprot_domain_wt['db'] = 'elaspic'
uniprot_domain_wt['uniprot_id'] = uniprot_domain_wt['uniprot_id'] + '_wt'
uniprot_domain_wt['path_to_data'] = None
uniprot_domain_wt['organism_name'] = 'training'
uniprot_domain_wt['uniprot_name'] = uniprot_domain_wt['uniprot_id'].apply(lambda x: x.split('_')[0]) + '_training'


### Mutation from mutant to wild-type
uniprot_domain_mut = core_mut_wseq_wdom.drop_duplicates(subset=['uniprot_domain_id', 'uniprot_id', 'uniprot_mutation']).copy()
uniprot_domain_mut['db'] = 'elaspic'
uniprot_domain_mut['uniprot_id'] = uniprot_domain_mut['uniprot_id'] + '_' + uniprot_domain_mut['uniprot_mutation']
uniprot_domain_mut['path_to_data'] = None
uniprot_domain_mut['organism_name'] = 'training'
uniprot_domain_mut['uniprot_name'] = uniprot_domain_mut['uniprot_id'].apply(lambda x: x.split('_')[0]) + '_training'

# Introduce the mutation into the uniprot sequence
uniprot_domain_mut['uniprot_sequence'] = (
    uniprot_domain_mut[['uniprot_sequence', 'uniprot_mutation']]
    .apply(parsers.mutate_sequence, axis=1, raw=True)
)
uniprot_domain_mut['uniprot_mutation'] = uniprot_domain_mut['uniprot_mutation'].apply(lambda x: x[-1] + x[1:-1] + x[0])
for column in ['dTm', 'ddG', 'ddG_H2O', 'ddg_best', 'ddg_all']:
    uniprot_domain_mut[column + '_mean'] = uniprot_domain_mut[column + '_mean'].apply(lambda x: -x)
    uniprot_domain_mut[column + '_median'] = uniprot_domain_mut[column + '_median'].apply(lambda x: -x)

print(len(uniprot_domain_wt))
print(len(uniprot_domain_mut))



#%% Add wild-type and mutant sequences to the MySQL database
engine = sa.create_engine('mysql://elaspic:elaspic@192.168.6.19/uniprot_kb')
sql_query = """
select uniprot_id
from uniprot_kb.uniprot_sequence
where db = 'elaspic' ;
"""
uniprots_uploaded_previously = pd.read_sql_query(sql_query, engine)
set_of_uniprots_uploaded_previously = set(uniprots_uploaded_previously['uniprot_id'].values)

uniprot_sequence_columns = [
    'db', 'uniprot_id', 'uniprot_name', 'protein_name', 'organism_name',
    'gene_name', 'protein_existence', 'sequence_version', 'uniprot_sequence']
uniprot_domain_wt[
        ~(uniprot_domain_wt['uniprot_id'].isin(set_of_uniprots_uploaded_previously))
    ][uniprot_sequence_columns].to_sql('uniprot_sequence', engine, if_exists='append', index=False)
uniprot_domain_mut[
        ~(uniprot_domain_mut['uniprot_id'].isin(set_of_uniprots_uploaded_previously))
    ][uniprot_sequence_columns].to_sql('uniprot_sequence', engine, if_exists='append', index=False)



#%% Save results
core_mut_wseq.to_pickle(constants.protherm_data_path + 'parsed_data{}/core_mut_wseq.pickle'.format(version_suffix))
core_mut_wseq_wdom.to_pickle(constants.protherm_data_path + 'parsed_data{}/core_mut_wseq_wdom.pickle'.format(version_suffix))
uniprot_domain_wt.to_pickle(constants.protherm_data_path + 'parsed_data{}/uniprot_domain_wt.pickle'.format(version_suffix))
uniprot_domain_mut.to_pickle(constants.protherm_data_path + 'parsed_data{}/uniprot_domain_mut.pickle'.format(version_suffix))



###################################################################################################
#%% PART 4 - PERFORM ALIGNMENTS TO FIND TEMPLATES AT DIFFERENT SEQUENCE IDENTITY THRESHOLDS
###################################################################################################

#%%
uniprot_domain_wt = pd.read_pickle(constants.protherm_data_path + 'parsed_data{}/uniprot_domain_wt.pickle'.format(version_suffix))
uniprot_domain_mut = pd.read_pickle(constants.protherm_data_path + 'parsed_data{}/uniprot_domain_mut.pickle'.format(version_suffix))


#%% Stuff for parallel processing
from IPython.parallel import Client
rc = Client()
dview = rc[:]
print('Connected to {} workers'.format(len(rc)))

async_result = dview.execute('%load_ext autoreload', silent=False, block=True)
print(''.join(async_result.stdout))
async_result = dview.execute('%autoreload 2', silent=False, block=True)
print(''.join(async_result.stdout))


#%%
uniprot_domain_wt['unique_id'] = range(len(uniprot_domain_wt))
uniprot_domain_mut['unique_id'] = range(len(uniprot_domain_wt), len(uniprot_domain_wt) + len(uniprot_domain_mut))

key_columns = ['unique_id', 'uniprot_id', 'uniprot_mutation', 'domain_def', 'uniprot_sequence']


def worker(df):
#    blast_results_mutdom_df = df.apply(parsers.get_templates, axis=1, raw=True)
#    templates_df = blast_results_mutdom_df.apply(parsers.stratify_results_by_identity, axis=1, raw=True)
    blast_results_mutdom_list = [
        parsers.get_templates(x) for x in df.values]

    templates_list = []
    failed_list = []
    for blast_results_mutdom in blast_results_mutdom_list:
        try:
            template = parsers.stratify_results_by_identity(blast_results_mutdom)
            templates_list.append(template)
        except:
            failed_list.append(blast_results_mutdom)
    return templates_list, failed_list

def get_df_chunks(df):
    return [x[1] for x in df.groupby(np.arange(len(df)) / (len(df) / len(rc) + 1))]

templates_and_failed_wt = dview.map_sync(worker, get_df_chunks(uniprot_domain_wt[key_columns]))
templates_and_failed_mut = dview.map_sync(worker, get_df_chunks(uniprot_domain_mut[key_columns]))

# TODO: Figure out why so many failed
templates_wt_df = pd.concat([x for xx in templates_and_failed_wt for x in xx[0]], ignore_index=True)
templates_mut_df = pd.concat([x for xx in templates_and_failed_mut for x in xx[0]], ignore_index=True)


del uniprot_domain_wt['ΔTagg']
del uniprot_domain_mut['ΔTagg']

uniprot_domain_wt_wtemplates = uniprot_domain_wt.merge(templates_wt_df, on=['unique_id'], suffixes=('_old', ''))
uniprot_domain_mut_wtemplates = uniprot_domain_mut.merge(templates_mut_df, on=['unique_id'], suffixes=('_old', ''))


#%%
uniprot_domain_wt_wtemplates['alignment_identity'] = uniprot_domain_wt_wtemplates['alignment_identity'] * 100
uniprot_domain_wt_wtemplates['alignment_coverage'] = uniprot_domain_wt_wtemplates['alignment_coverage'] * 100
uniprot_domain_mut_wtemplates['alignment_identity'] = uniprot_domain_mut_wtemplates['alignment_identity'] * 100
uniprot_domain_mut_wtemplates['alignment_coverage'] = uniprot_domain_mut_wtemplates['alignment_coverage'] * 100


#%% Correct sequence identity values

# Blast sometimes gives slightly different sequence coverage and sequence identity values,
# which screws up downstream analysis which depends on unique rows

# Here we assume that each ``unique_id_columns`` tuple should have a unque
# sequence identity / sequence coverage/ sequence score value

uniprot_domain_wt_wtemplates['t_date_modified'] = datetime.datetime.now()
uniprot_domain_mut_wtemplates['t_date_modified'] = datetime.datetime.now()

unique_id_columns = ['uniprot_id', 'alignment_def', 'max_seq_identity']
false_duplicate_columns = ['alignment_identity', 'alignment_coverage', 'alignment_score']

def remove_false_duplicates(df):
    df_base = df[[c for c in df.columns if c not in false_duplicate_columns]].drop_duplicates()
    df_duplicates = (
        df[unique_id_columns + false_duplicate_columns]
        .sort('alignment_score', ascending=False)
        .drop_duplicates(unique_id_columns)
    )
    df_combined = df_base.merge(df_duplicates, on=unique_id_columns)
    return df_combined

uniprot_domain_wt_wtemplates = remove_false_duplicates(uniprot_domain_wt_wtemplates)
uniprot_domain_mut_wtemplates = remove_false_duplicates(uniprot_domain_mut_wtemplates)


#%% Sanity check of the data

# The lengths should be the same because they are just the forward and backward mutations
assert len(uniprot_domain_wt_wtemplates) == len(uniprot_domain_mut_wtemplates)


#%% Export data that will be relied on by all other programs...
uniprot_domain_wt_mut_wtemplates = pd.concat([uniprot_domain_wt_wtemplates, uniprot_domain_mut_wtemplates], ignore_index=True)

uniprot_domain_wt_wtemplates.to_pickle(
    constants.protherm_data_path + 'parsed_data{}/uniprot_domain_wt_wtemplates.pickle'.format(version_suffix))
uniprot_domain_mut_wtemplates.to_pickle(
    constants.protherm_data_path + 'parsed_data{}/uniprot_domain_mut_wtemplates.pickle'.format(version_suffix))
uniprot_domain_wt_mut_wtemplates.to_pickle(data_path + 'core/uniprot_domain_wt_mut_wtemplates{}.pickle'.format(version_suffix))

# These are the columns that I used as final data previously, but it is wrong!!!
# ddG_mean
# dTm_mean

# These are the columns that should be used by all subsequent functions:
# ddg_all_median
# dTm_median


#%% Save the uniprot_id \t mutations data to a tsv file to use as input for elaspic
unique_uniprot_id_mutations = uniprot_domain_wt_mut_wtemplates[['uniprot_id', 'uniprot_mutation']].drop_duplicates()
assert (
    unique_uniprot_id_mutations[unique_uniprot_id_mutations['uniprot_id'].str.endswith('_wt')].shape[0] ==
    unique_uniprot_id_mutations[~unique_uniprot_id_mutations['uniprot_id'].str.endswith('_wt')].shape[0]
)
training_tsv = (
    unique_uniprot_id_mutations
    .groupby(['uniprot_id'])
    .agg({'uniprot_mutation': lambda x: ','.join(list(set(x)))})
    .reset_index()
)
training_tsv.to_csv(
    '/home/kimlab1/strokach/working/elaspic/input/training_core{}.tsv'.format(version_suffix),
    sep='\t', index=False, header=False)



#%% DEBUGGING PURPOSES ONLY!!!
# Make sure that mutations with very high ddG values are not errors
uniprot_domain_mut_wtemplates_strange_ddg = uniprot_domain_mut_wtemplates[
        (pd.notnull(uniprot_domain_mut_wtemplates['ddg_all_mean'])) &
        ((uniprot_domain_mut_wtemplates['ddg_all_mean'] > 10) |
        (uniprot_domain_mut_wtemplates['ddg_all_mean'] < -10))
    ]
print(uniprot_domain_mut_wtemplates_strange_ddg[
        ['protein_name', 'uniprot_id', 'uniprot_mutation', 'ddg_all_median']
    ].drop_duplicates().to_csv(sep='\t'))


# Make sure that we don't have fewer mutations than we did previously
uniprot_domain_wt_wtemplates_old = pd.read_pickle(
    constants.protherm_data_path + 'parsed_data{}/uniprot_domain_wt_wtemplates.pickle'.format('_v8'))

uniprot_domain_wt_wtemplates_new_upmut_set = set(uniprot_domain_wt_wtemplates[['uniprot_id', 'uniprot_mutation']].apply(tuple, axis=1))
uniprot_domain_wt_wtemplates_old_upmut_set = set(uniprot_domain_wt_wtemplates_old[['uniprot_id', 'uniprot_mutation']].apply(tuple, axis=1))

uniprot_domain_wt_wtemplates_new_extra = uniprot_domain_wt_wtemplates[
        ~(uniprot_domain_wt_wtemplates[['uniprot_id', 'uniprot_mutation']]
            .apply(tuple, axis=1).isin(uniprot_domain_wt_wtemplates_old_upmut_set))
    ]
assert len(uniprot_domain_wt_wtemplates_new_extra) == 0

uniprot_domain_wt_wtemplates_old_extra = uniprot_domain_wt_wtemplates_old[
        ~(uniprot_domain_wt_wtemplates_old[['uniprot_id', 'uniprot_mutation']]
            .apply(tuple, axis=1).isin(uniprot_domain_wt_wtemplates_new_upmut_set))
    ]
assert len(uniprot_domain_wt_wtemplates_old_extra) == 0



###################################################################################################
#%% PART 5 - UPLOAD DATA TO THE DATABASE
###################################################################################################

#%%
uniprot_domain_wt_mut_wtemplates = pd.read_pickle(
    data_path + 'core/uniprot_domain_wt_mut_wtemplates{}.pickle'.format(version_suffix))



#%% Divide the uniprot domain mutation data into uniprot_domain table and uniprot_domain_template table
key_columns = ['uniprot_id', 'alignment_def', 'max_seq_identity']

uniprot_domain_columns = [
    'uniprot_id', 'pdbfam_name', 'pdbfam_idx', 'pfam_clan',
    'alignment_def', 'pfam_names', 'alignment_subdefs', 'path_to_data', 'max_seq_identity']

uniprot_domain_template_columns = [
    'template_errors', 'cath_id', 'domain_start', 'domain_end',
    'domain_def', 'alignment_identity', 'alignment_coverage', 'alignment_score',
    't_date_modified']

uniprot_domain_wt_mut_final = (
    uniprot_domain_wt_mut_wtemplates
    .drop_duplicates(subset=uniprot_domain_columns + uniprot_domain_template_columns)
    .dropna(subset=['domain_def'])
)

assert uniprot_domain_wt_mut_final.shape[0] == uniprot_domain_wt_mut_wtemplates.drop_duplicates(key_columns).shape[0]



#%%
engine = sa.create_engine('mysql://elaspic:elaspic@192.168.6.19/elaspic_training')


#%% Fixing my fuckups!! Coment out when doing things for real

# Back up data, because you will truncate the entire table

path_to_temp_backup = '/home/kimlab1/strokach/tmp/'

uniprot_domain = pd.read_sql_table('uniprot_domain', engine)
uniprot_domain_template = pd.read_sql_table('uniprot_domain_template', engine)
uniprot_domain_model = pd.read_sql_table('uniprot_domain_model', engine)
uniprot_domain_mutation = pd.read_sql_table('uniprot_domain_mutation', engine)

uniprot_domain.to_pickle(path_to_temp_backup + 'uniprot_domain.pickle')
uniprot_domain_template.to_pickle(path_to_temp_backup + 'uniprot_domain.pickle')
uniprot_domain_model.to_pickle(path_to_temp_backup + 'uniprot_domain.pickle')
uniprot_domain_mutation.to_pickle(path_to_temp_backup + 'uniprot_domain.pickle')



#%% Save the uniprot_domain and uniprot_domain_template tables to the sql database
max_uniprot_domain_id = pd.read_sql_query('select max(uniprot_domain_id) from uniprot_domain', engine).values[0,0]
if max_uniprot_domain_id is None:
    max_uniprot_domain_id = 0

uniprot_domain_wt_mut_final['uniprot_domain_id'] = range(
    max_uniprot_domain_id + 1,
    max_uniprot_domain_id + 1 + len(uniprot_domain_wt_mut_final)
)



#%%
sql_query = """
select uniprot_id, alignment_def, max_seq_identity, cath_id
from uniprot_domain
join uniprot_domain_template using (uniprot_domain_id)
"""
precalculated_data = pd.read_sql_query(sql_query, engine)
precalculated_data['max_seq_identity'] = precalculated_data['max_seq_identity'].astype(int)
precalculated_data_tuples = set(precalculated_data.apply(tuple, axis=1))

uniprot_domain_wt_mut_final_new = uniprot_domain_wt_mut_final[
        ~(uniprot_domain_wt_mut_final[['uniprot_id', 'alignment_def', 'max_seq_identity', 'cath_id']]
            .apply(tuple, axis=1).isin(precalculated_data_tuples))
    ]



#%%
uniprot_domain_wt_mut_final_new[['uniprot_domain_id'] + uniprot_domain_columns].to_sql(
    'uniprot_domain', engine, if_exists='append', index=False)

uniprot_domain_wt_mut_final_new[['uniprot_domain_id'] + uniprot_domain_template_columns].to_sql(
    'uniprot_domain_template', engine, if_exists='append', index=False)


print('Done!')




