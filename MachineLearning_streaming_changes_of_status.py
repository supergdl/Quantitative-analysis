import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
import pandas as pd
from matplotlib import style
import statistics

from collections import Counter

style.use("ggplot")


how_much_better = 5


FEATURES =  ['DE Ratio',
             'Trailing P/E',
             'Price/Sales',
             'Price/Book',
             'Profit Margin',
             'Operating Margin',
             'Return on Assets',
             'Return on Equity',
             'Revenue Per Share',
             'Market Cap',
             'Enterprise Value',
             'Forward P/E',
             'PEG Ratio',
             'Enterprise Value/Revenue',
             'Enterprise Value/EBITDA',
             'Revenue',
             'Gross Profit',
             'EBITDA',
             'Net Income Avl to Common ',
             'Diluted EPS',
             'Earnings Growth',
             'Revenue Growth',
             'Total Cash',
             'Total Cash Per Share',
             'Total Debt',
             'Current Ratio',
             'Book Value Per Share',
             'Cash Flow',
             'Beta',
             'Held by Insiders',
             'Held by Institutions',
             'Shares Short (as of',
             'Short Ratio',
             'Short % of Float',
             'Shares Short (prior ']


def Status_Calc(stock, sp500):
    difference = stock - sp500

    if difference > how_much_better:
        return 1
    else:
        return 0

# build the data set to feed it through sk learn
def Build_Data_Set():
    data_df = pd.read_csv("key_stats_acc_perf_WITH_NA_enhanced.csv")

    #data_df = data_df[:100]

    # shuffle data by index
    data_df = data_df.reindex(np.random.permutation(data_df.index))
    data_df = data_df.replace("NaN",0).replace("N/A",0)

    data_df["Status2"] = list(map(Status_Calc, data_df["stock_p_change"], data_df["sp500_p_change"]))

    # X is coordinates
    X = np.array(data_df[FEATURES].values)
    # converts nan to 0, to get over the error of
    # "ValueError: Input contains NaN, infinity or a value too large for dtype('float64')"
    X = np.nan_to_num(X)

    # y is target or the actual number whatever the classification is
    # The return of DataFrame.replace() is a DataFrame with value replaced,
    # so it can call replace() again in a chain
    y = (data_df["Status2"]
         .replace("underperform",0)
         .replace("outperform",1)
         .values.tolist())

    # normalize features
    X = preprocessing.scale(X)
    

    Z = np.array(data_df[["stock_p_change","sp500_p_change"]])
    # to get over the error
    #"UserWarning: Numerical issues were encountered when centering the data and might not be solved.
    #Dataset may contain too large values. You may need to prescale your features."
    Z = np.nan_to_num(Z)

    
    return X,y,Z



# Run ml algo and graph it
def Analysis():

    test_size = 1

    invest_amount = 10000
    total_invests = 0
    if_market = 0
    if_strat = 0
    
    X, y, Z = Build_Data_Set()
    print(len(X))

    clf = svm.SVC(kernel="linear", C=1.0)
    #clf.fit(X,y)
    clf.fit(X[:-test_size],y[:-test_size])

    correct_count = 0

    for x in range(1, test_size+1):
        # fit() Expected 2D array
        if clf.predict([X[-x]])[0] == y[-x]:
            correct_count += 1

        if clf.predict([X[-x]])[0] == 1:
            invest_return = invest_amount + (invest_amount * (Z[-x][0]/100))
            market_return = invest_amount + (invest_amount * (Z[-x][1]/100))
            total_invests += 1
            if_market += market_return
            if_strat += invest_return

##    print("Accuracy:", (correct_count/test_size) * 100.00)
##
##    print("Total Trades:", total_invests)
##    print("Ending with Strategy:",if_strat)
##    print("Ending with Market:",if_market)
##
##    compared = ((if_strat - if_market) / if_market) * 100.0
##    do_nothing = total_invests * invest_amount
##
##    avg_market = ((if_market - do_nothing) / do_nothing) * 100.0
##    avg_strat = ((if_strat - do_nothing) / do_nothing) * 100.0
##
##
##    
##    print("Compared to market, we earn",str(compared)+"% more")
##    print("Average investment return:", str(avg_strat)+"%")
##    print("Average market return:", str(avg_market)+"%")
##    

    

    


    data_df = pd.read_csv("forward_sample_WITH_NA.csv")

    data_df = data_df.replace("N/A",0).replace("NaN",0)

    X = np.array(data_df[FEATURES].values)
    X = np.nan_to_num(X)

    X = preprocessing.scale(X)
    

    Z = data_df["Ticker"].values.tolist()

    invest_list = []

    for i in range(len(X)):
        p = clf.predict([X[i]])[0]
        if p == 1:
            #print(Z[i])
            invest_list.append(Z[i])

    #print(len(invest_list))
    #print(invest_list)
    return invest_list

    


# shuffle data example
def Randomizing():
    df = pd.DataFrame({"D1":range(5), "D2":range(5)})
    print(df)
    # shuffle df.index and then reindex the index after shuffling
    df2 = df.reindex(np.random.permutation(df.index))
    print(df2)  
    return invest_list


# excute 8 times and find the best portfolio
final_list = []

loops = 8

for x in range(loops):
    stock_list = Analysis()
    for e in stock_list:
        final_list.append(e)

x = Counter(final_list)

print(15*"_")
for each in x:
    if x[each] > loops - (loops/3):
        print(each)
