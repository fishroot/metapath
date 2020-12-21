import os


#
# metapath generic functions
#

def get_empty_file(file):
    file_dirname = os.path.dirname(file) + '/'
    file_name = os.path.basename(file)
    file_base_name, file_ext = os.path.splitext(file_name)
    file_base = file_dirname + file_base_name
    
    # search unused filename
    if os.path.exists(file):
        file_id = 2
        while os.path.exists('%s (%s)%s' % (file_base, file_id, file_ext)):
            file_id += 1
        file = '%s (%s)%s' % (file_base, file_id, file_ext)
        
    # create path if not available
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
        
    return file