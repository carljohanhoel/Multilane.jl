# ZZZ Removed precompilation of Multilane, not sure what that means

push!(LOAD_PATH,joinpath("./src"))

using Revise #To allow recompiling of modules withhout restarting julia

parallel_version = true   #Test code in parallel mode
# parallel_version = false

# simple_run = true
simple_run = false

tree_in_info = false



include("simulation_setup.jl")




## Save files to log to be able to check parameters
if !ispath(log_path)
   mkdir(log_path)
end
mkdir(log_path*"/code")
cp(pwd()*"/test/",log_path*"/code/test/")
cp(pwd()*"/src/",log_path*"/code/src/")
cp(pwd()*"/scratch_cj/",log_path*"/code/scratch_cj/")
cp(estimator_path*".py",log_path*"/neural_net.py")

##
trainer = Trainer(rng=rng_trainer, rng_eval=rng_evaluator, training_steps=training_steps,
                  n_network_updates_per_sample=n_network_updates_per_sample, save_freq=save_freq,
                  eval_freq=eval_freq, eval_eps=eval_eps, fix_eval_eps=true, remove_end_samples=remove_end_samples,
                  stash_factor=stash_factor, save_evaluation_history=save_evaluation_history, show_progress=true, log_dir=log_path)
if parallel_version
   if sim_problem isa POMDP
      processes = train_parallel(trainer, hr, problem, policy, updater)
   else
      processes = train_parallel(trainer, hr, problem, policy)
   end
   for proc in processes #This make Julia wait with terminating until all processes are done. However, all processes will never finish when stash size is bigger than 1. Fine for now...
      fetch(proc)
   end
else
   if sim_problem isa POMDP
      train(trainer, hr, problem, policy, updater)
   else
      train(trainer, hr, problem, policy)
   end
end
##
