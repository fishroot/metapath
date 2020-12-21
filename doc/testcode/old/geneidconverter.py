#!/usr/bin/env python

#from src.mp_metapath import *

# check if python module 'src.mp_bioclite_wrapper' is available
try:
    from src.mp_bioclite_wrapper import bioconductor
    bioc = bioconductor()
except:
    print "could not import python module 'src.mp_bioclite_wrapper'"
    quit()

list = ['TFAP2A','Arnt','Arnt::Ahr','Ar','T','Pax5','NR2F1','Ddit3::Cebpa',
'E2F1','NFIL3','En1','ELK1','Evi1','FOXF2','FOXD1','FOXC1','FOXL1','GATA2',
'GATA3','Gfi','Foxq1','Foxd3','FOXI1','HLF','HNF1A','NHLH1','IRF1','IRF2',
'MEF2A','Myf','MZF1_1-4','MZF1_5-13','MAX','MYC::MAX','NFYA','NF-kappaB',
'Nkx2-5','PPARG','Pax2','Pax4','Pax6','PBX1','RORA_1','RORA_2','RREB1',
'RXRA::VDR','Prrx2','ELK4','SOX9','Sox17','SPIB','SRF','SRY','Sox5','znf143',
'NFE2L1::MafG','TEAD1','TAL1::TCF3','Hand1::Tcfe2a','USF1','YY1','ETS1','Myb',
'REL','ZEB1','NFKB1','TP53','RELA','TBP','Hltf','Spz1','NR3C1','HNF4A',
'NR1H2::RXRA','Zfp423','Mafb','TLX1::NFIC','Nkx3-2','NKX3-1','Nobox','ZNF354C',
'MIZF','Pdx1','BRCA1','Lhx3','ELF5','CTCF','Tal1::Gata1','Esrrb','Pou5f1','Sox2',
'Stat3','Tcfcp2l1 ','Zfx','Myc','FOXA1','EWSR1-FLI1','GABPA','Gata1','Klf4',
'REST','RUNX1','STAT1','Mycn','Foxa2','ESR1','PPARG::RXRA','NFE2L2','ARID3A',
'NFATC2','HNF1B','EBF1','INSM1','FEV','FOXO3','HOXA5','RXR::RAR_DR5',
'NR4A2','NFIC','Egr1','PLAG1','Nr2e3','SPI1','CREB1','AP1','SP1',
'CEBPA','ESR2','HIF1A::ARNT','SOX10']

# convert to EntrezIDs
list, black_list  = bioc.convert_geneids(
    input_file = "test.csv", input_format = 'entrezid',
    output_file = "E Symbol.txt", output_format = 'symbol')
    
print list
print black_list