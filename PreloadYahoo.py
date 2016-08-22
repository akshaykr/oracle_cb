import settings
import numpy as np

flist = [settings.DATA_DIR+"yahoo/set1.train.txt",
         settings.DATA_DIR+"yahoo/set1.valid.txt",
         settings.DATA_DIR+"yahoo/set1.test.txt",
         settings.DATA_DIR+"yahoo/set2.train.txt",
         settings.DATA_DIR+"yahoo/set2.valid.txt",
         settings.DATA_DIR+"yahoo/set2.test.txt"]

def preprocess():
    fidx = 0
    fname = flist[fidx]

    queryIds = {}
    queryDocs = {}
    order = []

    set1_features = set([])
    set2_features = set([])

    maxDocsPerQuery = 0
    print("Opening new file: %s" % (fname))
    f = open(fname,'r')
    num_lines = 0
    while True:
        curr_line = None
        done = False
        while curr_line == None:
            try:
                curr_line = f.__next__()
            except Exception:
                fidx += 1
                curr_line = None
                if fidx == len(flist):
                    done = True
                    break
                fname = flist[fidx]
                print("Opening new file: %s" % (fname))
                f = open(fname, 'r')
        if done:
            break
        num_lines += 1
        data = curr_line.split()
        curr_q = int(data[1].split(":")[-1])
        rel = int(data[0])
        features = dict([[int(x.split(":")[0]), float(x.split(":")[-1])] for x in data[2:]])

        if fname.find("set1") != -1:
            set1_features.update(set(features.keys()))
        if fname.find("set2") != -1:
            set2_features.update(set(features.keys()))

        if curr_q not in queryIds:
            queryIds[curr_q] = len(queryIds)
            queryDocs[queryIds[curr_q]] = 0

        queryid = queryIds[curr_q]
        order.append(queryid)
        doc_id = queryDocs[queryid]
        queryDocs[queryid] += 1
        if queryDocs[queryid] > maxDocsPerQuery:
            maxDocsPerQuery = queryDocs[queryid]

    numQueries = len(queryIds)
    numDocs = np.minimum(maxDocsPerQuery, 6)

    ## TODO: update
    feature_set = set1_features & set2_features
    numFeatures = len(feature_set)
    feature_map = dict(zip(list(feature_set), range(len(feature_set))))

    print("Datasets:loadTxt [INFO] Compiled statistics",
          " NumQueries, NumUnique, MaxNumDocs, MaxNumFeatures: ", numQueries, len(order), numDocs, numFeatures, flush = True)

    docsPerQuery = np.zeros(numQueries, dtype = np.int32)
    processed = {}
    for qid in queryIds.keys():
        docsPerQuery[queryIds[qid]] = queryDocs[queryIds[qid]]
        processed[queryIds[qid]] = 0
    relevances = -np.ones((numQueries, numDocs), dtype = np.int32)
    features = np.nan * np.ones((numQueries, numDocs, numFeatures), dtype = np.float32)

    fidx = 0
    fname = flist[fidx]
    print("Opening new file: %s" % (fname))
    f = open(fname,'r')
    num_lines = 0
    while True:
        curr_line = None
        done = False
        while curr_line == None:
            try:
                curr_line = f.__next__()
            except Exception:
                fidx += 1
                curr_line = None
                if fidx == len(flist):
                    done = True
                    break
                fname = flist[fidx]
                print("Opening new file: %s" % (fname))
                f = open(fname, 'r')
        if done:
            break

        num_lines += 1
        # if num_lines % 10000 == 0:
        #     print("Processed %d lines" % (num_lines))

        data = curr_line.split()
        relevance = int(data[0])
        qid = int(data[1].split(":")[-1])
        queryid = queryIds[qid]
        fvec = dict([[int(x.split(":")[0]), float(x.split(":")[-1])] for x in data[2:]])
        arr = np.array(np.zeros(numFeatures))
        for (k,v) in fvec.items():
            if k in feature_map.keys():
                arr[feature_map[k]] = v

        docIndex = processed[queryid]
        if docIndex >= numDocs:
            docsPerQuery[queryid] = numDocs
            continue
        processed[queryid] += 1
        relevances[queryid,docIndex] = relevance
        features[queryid,docIndex,:] = arr

    np.savez_compressed(settings.DATA_DIR+'yahoo/yahoo_big.npz', relevances=relevances,
                        features = features, docsPerQuery = docsPerQuery, queryOrder = order)
    print("Datasets:loadTxt [INFO] Loaded"
          " [Min,Max]NumDocs: ", np.min(docsPerQuery), np.max(docsPerQuery), flush = True)
    return (docsPerQuery, relevances, features)
    
if __name__=='__main__':
    preprocess()
