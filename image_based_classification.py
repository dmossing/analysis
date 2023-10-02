#!/usr/bin/env python

def gen_classifier(image_data,tuning_data,look_for = lambda x: x>np.percentile(x,95),red_image='r',feature_list=None):
    reload(opto_utils)
    tuning_data['sc'] = tuning_data['tuning'][:,:,:,:,:,8:-8].mean(-1).mean(-2) #.mean(-1)
    
    slope,score = opto_utils.roiwise_regression_slope(tuning_data['sc'])
    
    tuning_data['slope'] = slope
    tuning_data['score'] = score
    
    feature_dict = opto_utils.compute_image_features(image_data[red_image],image_data['g'],image_data['msk'],image_data['depth'])
    
    def gen_features(feature_dict,feature_list=None):
        if feature_list is None:
            feature_list = [feature_dict[key] for key in ['rednucleus','redaround','greenlvl','redgreencorr','redmembrane','redcv']]
            feature_list = feature_list + [feature_dict['redlvl']/feature_dict['greenlvl']]
            feature_list = feature_list + [feature_dict['redlvl']/feature_dict['redaround']]
        features = np.concatenate([f[tuning_data['cell_criteria']][:,np.newaxis] for f in feature_list],axis=1)
        return features
    
    image_data['feature_dict'] = feature_dict
    image_data['features_fn'] = gen_features
    features = image_data['features_fn'](image_data['feature_dict'])
    image_data['features'] = features

    criterion = gen_criterion(tuning_data,image_data) #(score>0) & ~np.isnan(features.sum(1))

#     features = sst.zscore(features[criterion],axis=0)
#     these_features = features[criterion]

    train = (np.random.randn(features.shape[0])>0) & criterion
    if not np.all(np.isnan(slope[train])):
        outcome = look_for(slope[train])
        classifier = {}
        classifier['logreg'] = sklearn.linear_model.LogisticRegression()
        classifier['logreg'].fit(features[train],outcome)
        classifier['transform'] = (classifier['logreg'].coef_ * image_data['features']).sum(1)
        classifier['training_set'] = train
        classifier['prediction'] = safe_predict_(classifier['logreg'],features)
    else:
        classifier = None
    return image_data, tuning_data, classifier

def safe_predict_(logreg,features):
    output = np.zeros((features.shape[0],),dtype='bool')
    lkat = ~np.isnan(features.sum(1)) & np.all(np.abs(features)<np.inf,axis=1)
    output[~lkat] = np.nan
    output[lkat] = logreg.predict(features[lkat])
    return output

def safe_predict_proba_(logreg,features):
    output = np.zeros((features.shape[0],))
    lkat = ~np.isnan(features.sum(1)) & np.all(np.abs(features)<np.inf,axis=1)
    output[~lkat] = np.nan
    output[lkat] = logreg.predict_proba(features[lkat])[:,1]
    return output

def extract_and_classify(dsname,iexpt,look_for = lambda x: x>np.percentile(x,95), running=False):
    tuning_data = extract_tuning_data(dsname,iexpt,running=False)
    image_data = extract_image_data(dsname,iexpt)
    if not np.any(np.isnan(image_data['r'])): # and not np.all(np.isnan(tuning_data['slope'])):
        image_data, tuning_data, classifier = gen_classifier(image_data,tuning_data,look_for=look_for)
    else:
        classifier = None
    return image_data,tuning_data,classifier