import settings
import numpy as np

flist = [settings.DATA_DIR+"mslr/mslr30k_train1.txt",
         settings.DATA_DIR+"mslr/mslr30k_train2.txt",
         settings.DATA_DIR+"mslr/mslr30k_train3.txt",
         settings.DATA_DIR+"mslr/mslr30k_train4.txt",
         settings.DATA_DIR+"mslr/mslr30k_train5.txt"]

def preprocess():
    fidx = 0
    fname = flist[fidx] ## settings.DATA_DIR+"mslr/mslr30k_vali%d.txt" % (fnum)
    queryIds = {}
    queryDocs = {}
    order = []
    maxDocsPerQuery = 0
    maxFeatures = 136
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
                fname = flist[fidx] ## settings.DATA_DIR+"mslr/mslr30k_vali%d.txt" % (fnum)
                print("Opening new file: %s" % (fname))
                f = open(fname, 'r')
        if done:
            break
        num_lines += 1
        data = curr_line.split()
        curr_q = int(data[1].split(":")[-1])
        rel = int(data[0])
        features = np.array([float(x.split(":")[-1]) for x in data[2:]])

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
    numDocs = min(maxDocsPerQuery, 50)
    numFeatures = maxFeatures

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
    fname = flist[fidx] ## settings.DATA_DIR+"mslr/mslr30k_vali%d.txt" % (fnum)
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
                fname = flist[fidx] ## settings.DATA_DIR+"mslr/mslr30k_vali%d.txt" % (fnum)
                print("Opening new file: %s" % (fname))
                f = open(fname, 'r')
        if done:
            break

        num_lines += 1
        data = curr_line.split()
        relevance = int(data[0])
        qid = int(data[1].split(":")[-1])
        queryid = queryIds[qid]
        fvec = np.array([float(x.split(":")[-1]) for x in data[2:]])

        docIndex = processed[queryid]
        if docIndex >= numDocs:
            docsPerQuery[queryid] = numDocs
            continue
        processed[queryid] += 1
        relevances[queryid,docIndex] = relevance
        features[queryid,docIndex,:] = fvec
    np.savez_compressed(settings.DATA_DIR+'mslr/mslr30k_tv.npz', relevances=relevances,
                        features = features, docsPerQuery = docsPerQuery, queryOrder = order)
    print("Datasets:loadTxt [INFO] Loaded"
          " [Min,Max]NumDocs: ", np.min(docsPerQuery), np.max(docsPerQuery), flush = True)
    return (docsPerQuery, relevances, features)
