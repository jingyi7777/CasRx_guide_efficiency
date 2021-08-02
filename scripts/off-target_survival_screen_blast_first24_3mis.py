#!/usr/bin/env python
# coding: utf-8

# In[1]:

from Bio import SeqIO
from Bio.Seq import Seq
#from Bio.Alphabet import generic_dna
from Bio.SeqRecord import SeqRecord
import csv
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
from Bio.Blast.Applications import NcbiblastnCommandline
import random




input_guides = []
score_threshold = 27 #Blast score cutoff to be used to filter targets
match_cutoff = 20 #threshold for the number of bases that match
data_file = "../data/d14_plasmid_library_ratio_targeting_library.csv"
guide_input_file = "../data/off-target-info/blastin_essential_genes_first24.fasta" #input for blast (guides are rev comped to targets)
data_list = [] # a list version of the columns of data_file
#blast_result_file = "oak/stanford/groups/quake/weijingy/lincrna_candidates/blastresult/k562_coding_e1.xml" #file of blast results
targets = []
 


#generate a fasta file of guide targets
with open(data_file, 'r') as f:
	for i,row in enumerate(csv.reader(f.read().splitlines())):
		data_list.append(row) # if title exists, the first element is the title
		#guide = str(row[0]) # guide sequence
		guide = str(row[0])[0:24] # guide sequence,cut to first 24 bases
		guide_record = SeqRecord(Seq(guide), id = str(i))
		target = guide_record.reverse_complement()
		#if i != 0: # 0th row is the title
		targets.append(target)

SeqIO.write(targets, guide_input_file, "fasta") #create blast input file



#run blast 
blastn_cline = NcbiblastnCommandline(query=guide_input_file, db="../database/BLAST_Gencode/gencode.v33.transcripts.fa", 
										evalue=1, task="blastn", outfmt=5, out='../data/off-target-info/k562_essential_e1_f24.xml')
blastn_cline
print(blastn_cline)
stdout, stderr = blastn_cline()
blast_result_file = "../data/off-target-info/k562_essential_e1_f24.xml" #file of blast results

specificity_scores = [] # list of guide, ratio (from data_file) and number of blast matches
#parse Blast output
result_handle = open(blast_result_file)
blast_records =  NCBIXML.parse(result_handle)
for i,blast_record in enumerate(blast_records):
	match = 0
	match_dict = {}
	transcript_list=[] # store all aligned transcripts
	gene_list = []
	for alignment in blast_record.alignments:
		for hsp in alignment.hsps:
			if (hsp.score > score_threshold) and (hsp.strand == ('Plus', 'Plus')): #score_threshold = bit_score
				alignment = str(alignment)
				transcript_position_start = alignment.find("(")
				transcript_position_end = alignment.find(")")
				transcript_symbol = alignment[(transcript_position_start+1):transcript_position_end]
				gene_symbol = transcript_symbol.split('|')[5]
				query = str(hsp.query)
				match_seq = str(hsp.match) #calculate matches
				gaps = query.count("-") # gaps in the query
				gaps_ref = (str(hsp.sbjct)).count("-") # gaps in the reference transcripts
				matching_nt = len(query) - gaps #this may include mismatches
#				if (gaps == 0) and (gaps_ref == 0) and (len(match_seq) == match_cutoff): #perfect match                   
#				if (gaps + gaps_ref + match_cutoff - len(match_seq))==1: # 1 gap or mismatch
				#if (gaps + gaps_ref + match_cutoff - len(match_seq))==2: # 2 gap or mismatch
				if len(match_seq) > match_cutoff: # up to 3 mismatches
					transcript_list.append(transcript_symbol) # all aligned transcripts
					if gene_symbol in match_dict: # existing gene is not counted again in "match"
						continue
					else:
						match_dict[gene_symbol] = 1 #creates an entry in the dictionary for this (new) gene
						match += 1   #a new unique gene match was found
	gene_list = list(match_dict.keys())
	specificity_scores.append(data_list[i] + [match] + [gene_list] )

#write guide results to file: guide, ratio (from data_file) and number of blast matches
guide_result_file = "../data/off-target-info/essential_genes_blast_first_24_3mis_e1.csv" 

with open(guide_result_file, mode='w') as library:
	writer = csv.writer(library)
	writer.writerow(['guide','gene', 'gene_id', 'pos', 'raw ratio']+
						['blast_f24_mis3_e1_20_match_num']+['blast_gene_list_f24_mis3_e1_20'])
	#writer.writerow(data_list[0]+['blast_f24_mis3_e1_20_match_num']+['blast_gene_list_f24_mis3_e1_20'])
	for row in specificity_scores:
		#if row[4]==1: # only one gene match (specific to only one gene). I think maybe I need to change it to "0" for 1/2 gap/mismatch cases?
		writer.writerow(row)

