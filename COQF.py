import pandas as pd
from itertools import combinations
from math import log
from math import sqrt
from math import floor
import numpy as np
from functools import reduce
from sklearn.metrics.pairwise import cosine_similarity
import json


def binarize(df):
  return df.applymap(lambda x: 1 if x > 0 else 0)

def standard_qf(donation_df):
  projects = donation_df.columns
  funding = {p: (donation_df[p].apply(lambda x: sqrt(x)).sum() ** 2) - donation_df[p].sum() for p in projects}

  return funding


def extract_info_from_json(json_fpath):
  with open('test-votes.json') as f:
    votes_json = json.load(f)

  # get all the user ids
  user_ids = [x['user_id'] for x in votes_json]

  issues = list(votes_json[0]['credits'].keys())
  #(could assert that this should be the same no matter what index we pick)

  #split issues into "for" and "against" versions
  issues_fa = [i + '-for' for i in issues] + [i + '-against' for i in issues]

  # "refactoring" the json object like this will make it easier to
  # build all this data into an array for pandas
  votes_json_2 = {x['user_id']: x['credits'] for x in votes_json}

  votematrix_fa = [[votes_json_2[u][i]['for'] for i in issues] + [votes_json_2[u][i]['against'] for i in issues] for u in user_ids]
  votematrix = [[votes_json_2[u][i]['for'] - votes_json_2[u][i]['against'] for i in issues] for u in user_ids]

  vote_fa = pd.DataFrame(index=user_ids, columns=issues_fa, data=votematrix_fa)
  vote_comb = pd.DataFrame(index=user_ids, columns=issues, data=votematrix)

  return user_ids, issues, issues_fa, vote_fa, vote_comb


def COQF_sp26(json_fpath, calcstyle='cosine', harsh=False):

  user_ids, issues_comb, issues_fa, vote_fa, vote_comb = extract_info_from_json(json_fpath)

  # issues_comb is a list of issues ("combined"), issues_fa is split into "X-for" and "X-against" for each issue X
  # same for vote_comb and vote_fa, but these are vote matrices (DataFrames technically)


  # calculate the cosine similarity between each pair of users
  # For this we'll use the "combined" dataframe where an against vote is a negative number and a for vote is a positive number
  # this makes more sense to me because "for" and "against" votes are not merely orthogonal but instead represent opposite opinions

  cosine_sim = pd.DataFrame(index = user_ids, columns = user_ids, data = (cosine_similarity(vote_comb, vote_comb) + 1)/2)
  # (add 1 and divide by two to sacale from a (-1,1) range to a (0,1) range)

  # calculate ow similar each user is to the average voter in each issue (for or against)
  k_indicators = pd.DataFrame(index=issues_fa, columns=user_ids, data=[cosine_sim.apply(lambda r: r * vote_fa[i] / vote_fa[i].sum()).sum() for i in issues_fa]).T
  # digging into the above line:
  # for a particular issue i, cosine_sim.apply(lambda r: r * votedf_fa[i] / votedf_fa[i].sum() scales each row by that user's relative contribution to issue i
  # summing that matrix along the columns then gives a row vector where entry (0,j) is user j's cosine similarity to the average supporter of issue i (where the average is weighted by relative contribution to issue i)
  # the matrix is made up of those row vectors, and then transformed so that ultimately entry in row u, column i is user u's average cosine similarity to the supporters of issue i

  # this line is kind of a holdover from the old days, not sure if neccesary
  # we're setting a user's distance to an issue to 1 if they directly contributed to that issue
  binary_votes_fa = binarize(vote_fa)
  k_indicators = k_indicators.apply(lambda r: np.maximum(r, binary_votes_fa[r.name]))


  # Create a dictionary to store funding amounts for each project.
  funding = {i: vote_fa[i].sum() for i in issues_fa}


  for i in issues_fa:

    # get the actual k values for this project using contributions and indicators.

    # C will be used to build the matrix of k values.
    # It is a matrix where rows are users, columns are issues, and the uth row of the matrix just has user u's contribution to the issue i (what we're focusing on in this loop iteration) in every entry.
    a = pd.DataFrame(index=user_ids, columns = ['_'], data = vote_fa[i].values)
    b = pd.DataFrame(index= ['_'], columns = issues_fa, data=1)
    C = a.dot(b)
    # C is attained by taking the matrix multiplication of the column vector vote_fa[i] (which is every agent's donation to issue i) and a row vector with as many columns as projects, and a 1 in every entry
    # the above line requires casting vectors to dataframes, which accounts for most of the length


    # now, K is a matrix where rows are users, columns are issues, and entry u,i ranges between c_i and sqrt(c_i) depending on i's relationship with cluster g
    K = (k_indicators * C.pow(1/2)) + ((1 - k_indicators) * C)

    if harsh == True:
      K = (1 - k_indicators) * C



    # so, the matrix K holds each user's contribution to issue i, scaled by their relationship with some other issue j


    # normalize the cluster dataframe so that rows sum to 1. Now, an entry tells us the "weight" that a particular cluster has for a particular user.
    # but take out donations to issue i first. That way, a user's weight is spread among only the non-i issues
    votes_minus_i = vote_fa.copy()
    votes_minus_i[i].values[:] = 0
    normalized_votes_minus_i = votes_minus_i.apply(lambda row: row / row.sum() if any(row) else 0, axis=1)



    # Now we have all the k values, which are one of the items inside the innermost sum expressed in COCM.
    # the other component of these sums is a division of each k value by the number of groups that user is in.
    # P_prime is a matrix that combines k values and total group memberships to attain the value inside the aforementioned innermost sum.
    # In other words, entry g,h of P_prime is:
    #
    #       sum_{i in g} K(i,h) / T_i
    #
    # where T_i is the total number of groups that i is in
    P_prime = K.transpose().dot(normalized_votes_minus_i)

    # Now, we can create P, whose non-diagonal entries g,h represent the pairwise subsidy given to the pair of groups g and h.
    P = (P_prime * P_prime.transpose()).pow(1/2)

    # The diagonal entries of P are not relevant, so get rid of them. We only care about the pairwise subsidies between distinct groups.
    np.fill_diagonal(P.values, 0)

    # Now the sum of every entry in P is the amount of subsidy funding CO-QF awards to the project.
    funding[i] += P.sum().sum()

    # print(K)
    # print()
  funding_final = {i: sqrt(funding[i+'-for']) - sqrt(funding[i+'-against']) for i in issues_comb}
  return funding_final
