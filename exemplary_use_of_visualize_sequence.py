from visualize_gaming_sequence import visualize_sequence


# data parameters (to identify input_ and eye_data files)

id_code = 'MO07LN1' #'AE07EM1'
done_ = 'done' #'crashed'
n_run = '43' #38
arg_comb = '3TS' #'5TW'
start_time = 5.237867832183838
end_time = 10.237867832183838

# call visualize function
visualize_sequence(id_code=id_code, done_=done_, n_run=n_run, arg_comb=arg_comb, start_time=start_time, end_time=end_time, safe_ani=True)
#visualize_sequence(id_code=id_code, done_=done_, n_run=n_run, arg_comb=arg_comb, safe_ani=True)
