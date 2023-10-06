from visualize_gaming_sequence import visualize_sequence


# data parameters (to identify input_ and eye_data files)

id_code = 'AR02AA'  #'AR02AA' #'EU29TT1' #'AE07EM1'
done_ = 'done'  #'done'
n_run = '22'  # '46'
arg_comb = '5TS'  # '6FW'
start_time = 3.478442
end_time = 16.478442

# call visualize function
visualize_sequence(id_code=id_code, done_=done_, n_run=n_run, arg_comb=arg_comb, start_time=start_time, end_time=end_time, safe_ani=False)
# visualize_sequence(id_code=id_code, done_=done_, n_run=n_run, arg_comb=arg_comb, safe_ani=True)
