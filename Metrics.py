###Classes that define different metrics for semi-synthetic experiments
import numpy


class Metric:
    #ranking_size: (int) Maximum size of slate across contexts, l
    def __init__(self, ranking_size):
        self.rankingSize = ranking_size
        self.name = None
        ###All sub-classes of Metric should supply a computeMetric method
        ###Requires: relevance_list of size ranking_size, query_id (only used by NDCG)
        ###Returns: (float, clickList, dwellTimes) indicating value, clicked docs
        ###             and dwell times on clicked docs.

class ConstantMetric(Metric):
    def __init__(self, ranking_size, constant):
        Metric.__init__(self, ranking_size)
        self.constant = constant
        self.name = 'Constant'
        print("ConstantMetric:init [INFO] RankingSize: %d Constant: %f" % 
                (ranking_size, constant), flush = True)
    #relevance_list ([int],length=ranking_size): Relevance of document
    #in each slot of the slate.
    #query_id: (int) Index of the query (unused)
    def computeMetric(self, relevance_list, num_ranks, query_id):
        clickedDocs = (relevance_list > 0)
        return self.constant, clickedDocs, clickedDocs.astype(numpy.float32)
        
class DCG(Metric):
    def __init__(self, ranking_size):
        Metric.__init__(self, ranking_size)
        self.discountParams = 2.0 + numpy.array(range(self.rankingSize), dtype = numpy.float32)
        self.discountParams = numpy.reciprocal(numpy.log2(self.discountParams))
        self.name = 'DCG'
        print("DCG:init [INFO] RankingSize:", ranking_size, flush = True)
    #relevance_list ([int],length=ranking_size): Relevance of document
    #in each slot of the slate.
    #query_id: (int) Index of the query (unused)
    def computeMetric(self, relevance_list, num_ranks, query_id):
        gain = numpy.exp2(relevance_list) - 1.0
        dcg = numpy.dot(self.discountParams, gain)
        clickedDocs = (relevance_list > 0)
        return dcg, clickedDocs, clickedDocs.astype(dtype = numpy.float32)
        
class NDCG(Metric):
    #relevances: All query-document relevances, to compute max gain
    #allow_repetitions: (bool) If True, max gain is computed as if repetitions are allowed in the ranking
    def __init__(self, ranking_size, relevances, allow_repetitions):
        Metric.__init__(self, ranking_size)
        self.discountParams = 2.0 + numpy.array(range(self.rankingSize), dtype = numpy.float32)
        self.discountParams = numpy.reciprocal(numpy.log2(self.discountParams))
        #Make this mimic the Letor 4.0 Eval script
        #self.discountParams[0] = 1.0
        #self.discountParams[1] = 1.0
        self.name = 'NDCG'
        currentRelevances = relevances.copy()
        currentRelevances[currentRelevances == -1] = 0
        numQueries, numDocs = numpy.shape(currentRelevances)
        validDocs = min(numDocs, ranking_size)
        if not allow_repetitions:
            self.sortedRelevances = numpy.zeros((numQueries, ranking_size), dtype = numpy.int32)
            self.sortedRelevances[:,0:validDocs] = (-numpy.sort(-currentRelevances, axis = 1))[:, 0:validDocs]
        else:
            self.sortedRelevances = numpy.tile((currentRelevances.max(axis = 1))[:, None], ranking_size)
        print("NDCG:init [INFO] RankingSize: %d [numQueries, numDocs]: %d,%d AllowRepetitions" % 
                    (ranking_size, numpy.shape(relevances)[0], numpy.shape(relevances)[1]),
                    allow_repetitions, flush = True)
    def computeMetric(self, relevance_list, num_ranks, query_id):
        gain = numpy.exp2(relevance_list) - 1.0
        dcg = numpy.dot(self.discountParams, gain)
        maxGain = numpy.exp2(self.sortedRelevances[query_id, :]) - 1.0
        maxDCG = numpy.dot(self.discountParams, maxGain)
        nDCG = 0.0
        if maxDCG != 0:
            nDCG = dcg * 1.0 / maxDCG
        clickedDocs = (relevance_list > 0)
        return nDCG, clickedDocs, clickedDocs.astype(dtype = numpy.float32)
        
class MaxRelevance(Metric):
    def __init__(self, ranking_size):
        Metric.__init__(self, ranking_size)
        self.name = 'MaxRelevance'
        print("MaxRelevance:init [INFO] RankingSize:", ranking_size, flush = True)
    def computeMetric(self, relevance_list, num_ranks, query_id):
        currRelevance = relevance_list
        if num_ranks < self.rankingSize:
            currRelevance = relevance_list[0:num_ranks]
        maxRelevance = currRelevance.max()
        maxRelevanceDoc = numpy.random.choice(numpy.where(currRelevance == maxRelevance)[0])
        clickedDocs = numpy.zeros(self.rankingSize, dtype = numpy.bool)
        clickedDocs[maxRelevanceDoc] = 1
        return maxRelevance, clickedDocs, clickedDocs.astype(numpy.float32)
        
class SumRelevance(Metric):
    def __init__(self, ranking_size):
        Metric.__init__(self, ranking_size)
        self.name = 'SumRelevance'
        print("SumRelevance:init [INFO] RankingSize:", ranking_size, flush = True)
    def computeMetric(self, relevance_list, num_ranks, query_id):
        sumRelevance = relevance_list.sum(dtype = numpy.float32)
        clickedDocs = (relevance_list > 0)
        return sumRelevance, clickedDocs, clickedDocs.astype(dtype = numpy.float32)
        
class UserModelMetrics(Metric):
    #abandon_penalty: (int) If no sat-click, TTS is set to this value
    #click_probabilities: ([int],length=relevance_grades) probability of clicking 
    #                       conditioned on how relevant the document is.
    #stop_probabilities: ([int],length=relevance_grades) probability of stopping 
    #                       conditioned on how relevant the document is.
    def __init__(self, ranking_size, abandon_penalty,
                    click_probabilities, stop_probabilities):
        Metric.__init__(self, ranking_size)
        self.abandonPenalty = abandon_penalty
        self.clickProbabilities = click_probabilities
        self.stopProbabilities = stop_probabilities
        print("UserModelMetrics:init [INFO] RankingSize: %d AbandonPenalty: %f" % (ranking_size, abandon_penalty), flush = True)
        print("UserModelMetrics:init [INFO] ClickProbabilities:", click_probabilities, flush = True)
        print("UserModelMetrics:init [INFO] StopProbabilities:", stop_probabilities, flush = True)
    def computeMetric(self, relevance_list, num_ranks, query_id):
        currRelevance = relevance_list
        if num_ranks < self.rankingSize:
            currRelevance = relevance_list[0:num_ranks]
        clickProb = self.clickProbabilities[currRelevance]
        clickVals = numpy.random.random(num_ranks)
        clickEvents = (clickVals < clickProb)
        stopProb = self.stopProbabilities[currRelevance]
        stopVals = numpy.random.random(num_ranks)
        stopEvents = (stopVals < stopProb)
        satClicks = numpy.logical_and(clickEvents, stopEvents)
        unsatClicks = numpy.logical_xor(clickEvents, satClicks)
        satClickIndex = -1
        if numpy.any(satClicks):
            satClickIndex = satClicks.nonzero()[0][0]
            if satClickIndex < (num_ranks - 1):
                clickEvents[satClickIndex+1:] = 0
                stopEvents[satClickIndex+1:] = 0
                unsatClicks[satClickIndex+1:] = 0
                satClicks[satClickIndex+1:] = 0
        numUnsatClicks = unsatClicks.sum(dtype = numpy.int32)
        dwells = numpy.random.gamma(shape = 1, scale = 4, size = numUnsatClicks)
        numExamined = num_ranks
        if satClickIndex >= 0:
            numExamined = satClickIndex + 1
        timeTaken = 1.0 * numExamined + dwells.sum(dtype = numpy.float32)
        if (satClickIndex < 0) or (timeTaken > self.abandonPenalty):
            timeTaken = self.abandonPenalty
        dwellTimes = numpy.zeros(self.rankingSize, dtype = numpy.float32)
        dwellTimes[numpy.where(unsatClicks)] += dwells
        clickedDocs = numpy.zeros(self.rankingSize, dtype = numpy.int32)
        clickedDocs[numpy.where(clickEvents)] = 1
        if timeTaken < 8:
            return 1, clickedDocs.astype(int), dwellTimes
        else:
            return 0, clickedDocs.astype(int), dwellTimes
        return timeTaken, clickedDocs.astype(int), dwellTimes
        
class NavigationalTTS(UserModelMetrics):
    #relevance_grades: (int) Number of unique relevance ratings.
    def __init__(self, ranking_size, abandon_penalty, relevance_grades):
        clickProbabilities = numpy.linspace(start = 0.05, stop = 0.95,
                                                num = relevance_grades, dtype = numpy.float32)
        stopProbabilities = numpy.linspace(start = 0.05, stop = 0.5,
                                               num = relevance_grades, dtype = numpy.float32)
        UserModelMetrics.__init__(self, ranking_size, abandon_penalty,
                                      clickProbabilities, stopProbabilities)
        self.name = 'NavigationalTTS'
        
class InformationalTTS(UserModelMetrics):
    def __init__(self, ranking_size, abandon_penalty, relevance_grades):
        clickProbabilities = numpy.linspace(start = 0.4, stop = 0.9,
                                                num = relevance_grades, dtype = numpy.float32)
        stopProbabilities = numpy.linspace(start = 0.05, stop = 0.5,
                                               num = relevance_grades, dtype = numpy.float32)
        UserModelMetrics.__init__(self, ranking_size, abandon_penalty,
                                      clickProbabilities, stopProbabilities)
        self.name = 'InformationalTTS'
        
if __name__ == "__main__":
    allRelevances = numpy.random.random_integers(0, high = 2, size = 20)
    relevances = numpy.random.choice(allRelevances, size = 10, replace = False)
    print("All Relevances", allRelevances, flush = True)
    print("Relevances", relevances, flush = True)
    dcg = DCG(10)
    print("DCG", dcg.computeMetric(relevances, 5, 0), flush = True)
    ndcg = NDCG(10, allRelevances[None,:], False)
    print ("NDCG NoRep", ndcg.computeMetric(relevances, 5, 0), flush = True)
    ndcg = NDCG(10, allRelevances[None,:], True)
    print ("NDCG YesRep", ndcg.computeMetric(relevances, 5, 0), flush = True)
    maxrel = MaxRelevance(10)
    print("MaxRelevance", maxrel.computeMetric(relevances, 5, 0), flush = True)
    sumrel = SumRelevance(10)
    print("SumRelevance", sumrel.computeMetric(relevances, 5, 0), flush = True)
    navigator = NavigationalTTS(10, 60, 3)
    print("NavigationalTTS", navigator.computeMetric(relevances, 5, 0), flush = True)
    informer = InformationalTTS(10, 60, 3)
    print("InformationalTTS", informer.computeMetric(relevances, 5, 0), flush = True)
