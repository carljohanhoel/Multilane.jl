using JLD

JLD.save("./Logs/myfile.jld", "state_hist", hist.state_hist, "action_hist", hist.action_hist, "reward_hist", hist.reward_hist)#, "ainfo_hist", hist.ainfo_hist)

saved_data = JLD.load("./Logs/myfile.jld")

using Plots

a = []
for i in 1:length(saved_data["action_hist"])
    push!(a,saved_data["action_hist"][i].acc)
end
plot(a,linewidth=2,title="My Plot")

write_to_png(visualize(mdp,s,0.0),"./Figs/tmp_dbg.png")

write_to_png(visualize(ll["mdp"],ll["s"],0.0),"./Figs/tmp_dbg.png")
