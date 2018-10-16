using JLD

# JLD.save("./Logs/states_run_i.jld", "state_hist", hist_ref.state_hist, "action_hist", hist_ref.action_hist, "reward_hist", hist_ref.reward_hist)#, "ainfo_hist", hist.ainfo_hist)
JLD.save("./Logs/saved_states_one_vehicles.jld", "state_hist", hist_idle.state_hist)

saved_data = JLD.load("./Logs/states_run_i.jld")

using Plots

a = []
for i in 1:length(saved_data["action_hist"])
    push!(a,saved_data["action_hist"][i].acc)
end
plot(a,linewidth=2,title="My Plot")

write_to_png(visualize(mdp,s,0.0),"./Figs/tmp_dbg.png")

write_to_png(visualize(ll["mdp"],ll["s"],0.0),"./Figs/tmp_dbg.png")





#Code for loading saved evaluation history, visualizing MCTS tree and producing video
hist_loaded = JLD.load("./Logs/181015_170852_/eval_hist_process_1_step_37.jld")
hist_loaded = hist_loaded["hist"]


#Visualization
#Set time t used for showing tree. Use video to find interesting situations.
t = 3.0
step = convert(Int, t / pp.dt) + 1
write_to_png(visualize(sim_problem,hist_loaded.state_hist[step],hist_loaded.reward_hist[step]),"./Figs/state_at_t.png")
print(hist_loaded.action_hist[step])
inchromium(D3Tree(hist_loaded.ainfo_hist[step][:tree],init_expand=1))

#Produce video
frames = Frames(MIME("image/png"), fps=10/pp.dt)
@showprogress for (s, ai, r, sp) in eachstep(hist_loaded, "s, ai, r, sp")
    push!(frames, visualize(problem, s, r))
end
gifname = "./Figs/test_loaded_hist.ogv"
write(gifname, frames)
