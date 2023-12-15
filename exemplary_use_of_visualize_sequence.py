from visualize_gaming_sequence import visualize_sequence


# data parameters (to identify input_ and eye_data files)

id_code = 'UD06AD'
done_ = 'crashed'  #'done'
n_run = '30'  #'34'
arg_comb = '6TW'
start_time = 4.5
end_time = 14.5

# call visualize function
visualize_sequence(id_code=id_code, done_=done_, n_run=n_run, arg_comb=arg_comb, start_time=start_time, end_time=end_time, safe_ani=True)
# visualize_sequence(id_code=id_code, done_=done_, n_run=n_run, arg_comb=arg_comb, safe_ani=True)
