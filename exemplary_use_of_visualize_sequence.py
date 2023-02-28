from visualize_gaming_sequence import visualize_sequence


# data parameters (to identify input_ and eye_data files)

id_code = 'AE07EM1'
done_ = 'crashed'
n_run = 39
arg_comb = '5TW'
start_time = 5
end_time = 15

# call visualize function
visualize_sequence(id_code=id_code, done_=done_, n_run=n_run, arg_comb=arg_comb, start_time=5, end_time=15, safe_ani=False)
