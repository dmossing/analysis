#!/usr/bin/env python 
import sqlite3

def add_list_to_dict(this_hierarchy,dicti):
    parent_node = None
    if not parent_node in dicti:
        dicti[parent_node] = [None,[]]
    for ih,child_node in enumerate(this_hierarchy):
        if not child_node in dicti[parent_node][1]:
            dicti[parent_node][1] += [child_node]
            dicti[child_node] = [parent_node,[]]
        parent_node = child_node + 0
    return dicti

def get_hierarchy_strings(cur,name_list):
    found_vernaculars = []
    found_hierarchies = []
    for this_name in name_list:
#         print(this_name)
        these_tsns = []
        these_vernaculars = []
        for row in cur.execute('SELECT * FROM vernaculars WHERE LOWER(vernacular_name) LIKE LOWER("%s")'%this_name.replace('-','_')):
        # for row in cur.execute('SHOW COLUMNS FROM kingdoms'):
            these_tsns += [row[0]]
            these_vernaculars += [row[1]]
#             print(row)
        found_flag = False
        for ii,(this_tsn,this_vernacular) in enumerate(zip(these_tsns,these_vernaculars)):
            if not found_flag:
                for other_row in cur.execute('SELECT hierarchy_string FROM hierarchy WHERE TSN=%d'%this_tsn):
                    #print(this_vernacular)
                    found_vernaculars += [this_vernacular]
                    found_hierarchies += [other_row[0]]
                    #print(other_row[0])
                    found_flag = True
        if not found_flag:
            print('could not do %s'%this_name)
    return found_vernaculars,found_hierarchies

def string_lists_to_dicts(cur,vernaculars,hierarchies):
    tsn_dicti = {}
    tree_dicti = {}
    for this_name,this_hierarchy_str in zip(vernaculars,hierarchies):
        this_hierarchy = [int(s) for s in this_hierarchy_str.split('-')]
        for th in this_hierarchy[:-1]:
#             print(th)
            for row in cur.execute('SELECT complete_name FROM taxonomic_units WHERE tsn=%d'%th):
                tsn_dicti[th] = row[0]
#                 print(row[0])
        th = this_hierarchy[-1]
        for row in cur.execute('SELECT complete_name FROM taxonomic_units WHERE tsn=%d'%th):
            tsn_dicti[th] = '%s\n%s'%(row[0],this_name)
#             print(row[0])

        add_list_to_dict(this_hierarchy,tree_dicti)
    return tsn_dicti,tree_dicti

def common_names_to_dicts(sqlite_filename,common_names):
    con = sqlite3.connect(sqlite_filename)
    cur = con.cursor()
    found_vernaculars,found_hierarchies = get_hierarchy_strings(cur,common_names)
    tsn_dicti,tree_dicti = string_lists_to_dicts(cur,found_vernaculars,found_hierarchies)
    return found_vernaculars,found_hierarchies,tsn_dicti,tree_dicti

def dict_to_newick(dicti,root=None):
    if len(dicti[root][1])==0:
        return '%d'%root
    else:
        child_newick_list = [dict_to_newick(dicti,root=child) for child in dicti[root][1]]
        child_newick = '(%s)'%(','.join(child_newick_list))
        if root is None:
            return '%s;'%child_newick
        else:
            return '%s%d'%(child_newick,root)

def dicts_to_qnewick(tree_dicti,tsn_dicti,root=None):
    if len(tree_dicti[root][1])==0:
        return '"%s"'%tsn_dicti[root]
    else:
        child_newick_list = [dicts_to_qnewick(tree_dicti,tsn_dicti,root=child) for child in tree_dicti[root][1]]
        child_newick = '(%s)'%(','.join(child_newick_list))
        if root is None:
            return '%s;'%child_newick
        else:
            return '%s"%s"'%(child_newick,tsn_dicti[root])
