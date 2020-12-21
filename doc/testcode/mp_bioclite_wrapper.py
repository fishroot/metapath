class bioconductor:

    def __init__(self):

        # initialize logger
        import logging
        self.logger = logging.getLogger('metapath')

        # check if python module 'rpy2' is available
        try:
            import rpy2.robjects as robjects
        except:
            self.logger.critical("could not import python module 'rpy2'")
            quit()
        
        self.robjects = robjects
        self.r = robjects.r

    def install(self, package_list = []):
        # check if python module 'rpy2' is available
        try:
            import rpy2.robjects as robjects
            from rpy2.robjects.packages import importr
        except:
            self.logger.critical("could not import python module 'rpy2'")
            quit()

        # evaluate bioconductor R script
        base = importr('base')
        base.source("http://www.bioconductor.org/biocLite.R")
        bioclite = self.robjects.globalenv['biocLite']

        # install bioconductor packages
        if package_list == []:
            bioclite()
        else:
            for package in package_list:
                bioclite(package)

    def csv_to_dict(self, file = None, header = False):
        # check if python module 'csv' is available
        try:
            import csv
        except:
            self.logger.critical("could not import python module 'csv'")
            quit()
        
        # check if file is readable
        try:
            csvfile = open(file, "rb")
        except:
            self.logger.critical("could not open file '%s'" % (file))
            quit()

        # try to detect csv dialect
        try:
            dialect = csv.Sniffer().sniff(csvfile.read(1024))
            csvfile.seek(0)
            reader = csv.DictReader(csvfile, dialect)
        except:
            csvfile.seek(0)
            reader = csv.DictReader(csvfile, delimiter='\t', quotechar='\'')
        
        return reader

    def convert_geneids(self, input_list = [],
        input_format = 'alias', output_format = 'entrezid',
        input_file = None, output_file = None,
        filter_results = False):

        import sys

        # make local copy of input list
        list = input_list[:]
        
        # prapare format strings
        input_format = input_format.lower().strip()
        output_format = output_format.lower().strip()
        
        if not list:
            if not input_file:
                self.logger.critical("you have to specify at least one of the parameters: 'list', 'input_file'")
                quit()
            
            self.csv_to_dict(input_file)

        #
        # convert using annotation packages in bioconductor
        #

        annotation_packages = [
            "hgu95a",
            "hgu95av2",
            "hgu95b",
            "hgu95c",
            "hgu95d",
            "hgu95e",
            "hgu133a",
            "hgu133a2",
            "hgu133b",
            "hgu133plus2",
            "hgug4100a",
            "hgug4101a",
            "hgug4110b",
            "hgug4111a",
            "hgug4112a",
            "hguqiagenv3"
        ]

        original_stdout = sys.stdout
        if input_format in annotation_packages:
            
            # load bioconductor annotation package
            self.logger.info("sending command to R: library('%s.db')" % (input_format))
            try:
                sys.stdout = NullDevice()
                self.r.library("%s.db" % (input_format))
                sys.stdout = original_stdout
            except:
                sys.stdout = original_stdout
                self.logger.critical("you have to install the R/bioconductor package: '%s.db'" % (input_format))
                quit()
            
            # get listvector
            self.logger.info("sending command to R: x <- %s%s" % (input_format, output_format.upper()))
            try:
                sys.stdout = NullDevice()
                self.r('x <- %s%s' % (input_format, output_format.upper()))
                sys.stdout = original_stdout
            except:
                sys.stdout = original_stdout
                self.logger.critical("output format '%s' is not supported by '%s.db'" % (output_format, input_format))
                quit()
            
            self.logger.info("sending command to R: mapped_genes <- mappedkeys(x)")
            self.r('mapped_genes <- mappedkeys(x)')
            self.logger.info("sending command to R: listmap <- as.list(x[mapped_genes])")
            self.r('listmap <- as.list(x[mapped_genes])')
            
            # prepare search list
            search_list = []
            for a in list:
                if a[0] == 'X':
                    a = a[1:]
                search_list.append(a)
            
        elif input_format in ['entrezgeneid', 'entrezgene', 'entrezid', 'entrez']:
            # load bioconductor annotation package
            self.logger.info("sending command to R: library('org.Hs.eg.db')")
            try:
                sys.stdout = NullDevice()
                self.r.library("org.Hs.eg.db")
                sys.stdout = original_stdout
            except:
                sys.stdout = original_stdout
                self.logger.critical("you have to install the R/bioconductor package: 'org.Hs.eg.db'")
                quit()

            # get listvector
            self.logger.info("sending command to R: x <- org.Hs.eg%s" % (output_format.upper()))
            try:
                self.r('x <- org.Hs.eg%s' % (output_format.upper()))
            except:
                self.logger.critical("output format '%s' is not supported by 'org.Hs.eg.db'" % (output_format))
                quit()
            
            self.logger.info("sending command to R: mapped_genes <- mappedkeys(x)")
            self.r('mapped_genes <- mappedkeys(x)')
            self.logger.info("sending command to R: listmap <- as.list(x[mapped_genes])")
            self.r('listmap <- as.list(x[mapped_genes])')
            
            # prepare search list
            search_list = list

        elif output_format in ['entrezgeneid', 'entrezgene', 'entrezid', 'entrez']:
            # load bioconductor annotation package
            self.logger.info("sending command to R: library('org.Hs.eg.db')")
            try:
                sys.stdout = NullDevice()
                self.r.library("org.Hs.eg.db")
                sys.stdout = original_stdout
            except:
                sys.stdout = original_stdout
                self.logger.critical("you have to install the R/bioconductor package: 'org.Hs.eg.db'")
                quit()

            # get listvector
            self.logger.info("sending command to R: x <- org.Hs.eg%s2EG" % (input_format.upper()))
            try:
                self.r('x <- org.Hs.eg%s2EG' % (input_format.upper()))
            except:
                self.logger.critical("input format '%s' is not supported by 'org.Hs.eg.db'" % (input_format))
                quit()

            self.logger.info("sending command to R: mapped_genes <- mappedkeys(x)")
            self.r('mapped_genes <- mappedkeys(x)')
            self.logger.info("sending command to R: listmap <- as.list(x[mapped_genes])")
            self.r('listmap <- as.list(x[mapped_genes])')
            
            # prepare search list
            search_list = list

        else:
            self.logger.critical("conversion from '%s' to '%s' is not supported" % \
                (input_format, output_format))
            quit()

        # search listvector
        black_list = []
        for i, a in enumerate(search_list):
            try:
                self.r("sym <- listmap['%s']" % (a))
                self.r("sym <- unlist(sym)")
                list[i] = self.robjects.globalenv["sym"][0]
                found = True
            except:
                black_list.append(list[i])

        # filter results
        if filter_results:
            list = [item for item in list if item not in black_list]

        return list, black_list
    
class NullDevice():
    def write(self, s):
        pass
    def flush(self, s):
        pass
