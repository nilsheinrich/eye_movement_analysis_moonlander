from visualize_gaming_sequence import visualize_sequence


# data parameters (to identify input_ and eye_data files)

id_code = 'EU29TT1'  #'MO07LN1' #'AE07EM1'
done_ = 'done'
n_run = '46' #5
arg_comb = '6FW' #'5TW'
start_time = 35
end_time = 40

# call visualize function
visualize_sequence(id_code=id_code, done_=done_, n_run=n_run, arg_comb=arg_comb, start_time=start_time, end_time=end_time, safe_ani=False)
#visualize_sequence(id_code=id_code, done_=done_, n_run=n_run, arg_comb=arg_comb, safe_ani=True)
