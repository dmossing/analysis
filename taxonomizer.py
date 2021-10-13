#!/usr/bin/env python 
import sqlite3
import numpy as np

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

def sci_plus_common_string_lists_to_dicts(cur,sci_names,vernaculars,hierarchies):
    tsn_dicti = {}
    tree_dicti = {}
    for this_sn,this_name,this_hierarchy_str in zip(sci_names,vernaculars,hierarchies):
        this_hierarchy = [int(s) for s in this_hierarchy_str.split('-')]
        for th in this_hierarchy[:-1]:
            for row in cur.execute('SELECT complete_name FROM taxonomic_units WHERE tsn=%d'%th):
                tsn_dicti[th] = row[0]
        th = this_hierarchy[-1]
        tsn_dicti[th] = '%s\n%s'%(this_sn,this_name)
        add_list_to_dict(this_hierarchy,tree_dicti)
    return tsn_dicti,tree_dicti

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

def gen_name_dict(sci_names,com_names):
    name_dict = {}
    for sn,cn in zip(sci_names,com_names):
        if not ' x ' in sn and not 'sp.' in sn:
            sn = ' '.join(sn.split(' ')[:2])
            if not sn in name_dict:
                name_dict[sn] = [cn]
            else:
                name_dict[sn] += [cn]
    for sn in name_dict:
        name_dict[sn] = list(set(name_dict[sn]))
    return name_dict

def look_up_tsns_and_hierarchies(cur,name_dict):
    simple_list = list(name_dict.keys())
    simple_list.sort()
    found_sci_names,found_tsns,found_hierarchies = [[] for _ in range(3)]
    for iname,name in enumerate(simple_list):
        candidates = []
        for row in cur.execute('SELECT tsn FROM taxonomic_units WHERE complete_name="%s"'%name):
            candidates += [row[0]]
        if not candidates:
            for row in cur.execute('SELECT tsn FROM vernaculars WHERE vernacular_name="%s" AND language="English"'%name_dict[name][0].replace("'","\'")):
                candidates += [row[0]]
        if candidates:
            for cand in candidates:
                found_vernacular = False
                found_hierarchy = False
                for row in cur.execute('SELECT vernacular_name FROM vernaculars WHERE tsn=%d AND language="English"'%cand):
                    found_vernacular = True
                for row in cur.execute('SELECT hierarchy_string FROM hierarchy WHERE tsn=%d'%cand):
                    found_hierarchy = True
                    this_hierarchy = row[0]
                found_both = (found_vernacular and found_hierarchy)
                if found_both:
                    found_tsns += [cand]
                    found_sci_names += [name]
                    found_hierarchies += [this_hierarchy]
                    # add to found_hierarchy
                    break
            if not found_both:
                print('could not find in vernacular and hierarchy: %s'%name)
        else:
            print('could not find in tax: %s'%name)
    return found_sci_names,found_tsns,found_hierarchies

def look_up_vernaculars(cur,name_dict,found_sci_names,found_tsns):
    found_vernaculars = []
    for name,tsn in zip(found_sci_names,found_tsns):
        candidates = []
        for row in cur.execute('SELECT vernacular_name FROM vernaculars WHERE tsn=%d AND language="English"'%tsn):
            candidates += [row[0]]
        if len(candidates)>1:
            print('found extra for %s:'%name)
            for c in candidates:
                print(c)
            in_dict = [c for c in candidates if c in name_dict[name]]
            if in_dict:
                pick = in_dict[0]
            else:
                pick = candidates[0]
            print('picked '+str(pick))
            found_vernaculars += [pick]
            print('\n')
        elif len(candidates)==1:
            found_vernaculars += candidates
        elif not candidates:
            found_vernaculars += [None]
            print('could not do '+name)
            print('\n')
    return found_vernaculars

def get_sqlite_tax_info(sqlite_filename,name_dict):
    con = sqlite3.connect(sqlite_filename)
    cur = con.cursor()
    found_sci_names,found_tsns,found_hierarchies = look_up_tsns_and_hierarchies(cur,name_dict)
    found_vernaculars = look_up_vernaculars(cur,name_dict,found_sci_names,found_tsns)

    found_sci_names = [fsn for fsn,fv in zip(found_sci_names,found_vernaculars) if not fv is None]
    found_tsns = [fsn for fsn,fv in zip(found_tsns,found_vernaculars) if not fv is None]
    found_vernaculars = [fv for fv in found_vernaculars if not fv is None]

    return found_sci_names,found_vernaculars,found_tsns,found_hierarchies,cur

def sci_plus_common_names_to_dicts(sqlite_filename,sci_names,com_names):
    name_dict = gen_name_dict(sci_names,com_names)
    found_sci_names,found_vernaculars,found_tsns,found_hierarchies,cur = get_sqlite_tax_info(sqlite_filename,name_dict)
    tsn_dicti,tree_dicti = sci_plus_common_string_lists_to_dicts(cur,found_sci_names,found_vernaculars,found_hierarchies)
    return found_sci_names,found_vernaculars,found_hierarchies,tsn_dicti,tree_dicti

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
    # starting from node root, form a qnewick format tree
    if len(tree_dicti[root][1])==0:
        return '"%s"'%tsn_dicti[root]
    else:
        child_newick_list = [dicts_to_qnewick(tree_dicti,tsn_dicti,root=child) for child in tree_dicti[root][1]]
        child_newick = '(%s)'%(','.join(child_newick_list))
        if root is None:
            return '%s;'%child_newick
        else:
            return '%s"%s"'%(child_newick,tsn_dicti[root])

def dicts_to_ordered_qnewick(tree_dicti,tsn_dicti,order_dicti,root=None):
    # starting from node root, form a qnewick format tree
    if len(tree_dicti[root][1])==0:
        common_name = tsn_dicti[root].split(', ')[-1]
        if common_name in order_dicti:
            order_val = order_dicti[common_name]
        else:
            order_val = np.nan
        return '"%s"'%tsn_dicti[root],order_val
    else:
        qstrs = []
        order_vals = []
        for child in tree_dicti[root][1]:
            qstr,order_val = dicts_to_ordered_qnewick(tree_dicti,tsn_dicti,order_dicti,root=child)
            qstrs += [qstr]
            order_vals += [order_val]
        child_newick_list = [x for _,x in sorted(zip(order_vals,qstrs))]
        mean_order_val = np.nanmean(order_vals)
        child_newick = '(%s)'%(','.join(child_newick_list))
        if root is None:
            return '%s;'%child_newick,mean_order_val
        else:
            return '%s"%s"'%(child_newick,tsn_dicti[root]),mean_order_val
