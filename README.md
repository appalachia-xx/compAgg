# romeo
ReinfOrceMent lEarning cOmpressor

This is a minimal readme which explains how to recreate graphs in the paper.
Notice you might need to install dependencies (liebre, kafka) and also install python packages to create plots
Input files available here: https://drive.google.com/file/d/1-MVVS-RNej3ct6t54bTI4FdMpEyQs5ds/view?usp=sharing 

# Linear Road Experiments (Fig. 7) / Fig. 9 can be done similarly

- To begin with, run the baselines for different X values
- `./scripts/start_all_evaluation_CCR.sh`
- `./scripts/create_plots_for_exp.sh data/agentperformance/linearroad True 0/200/0.5/1 1/200/0.5/1 2/200/0.5/1 3/200/0.5/1 4/200/0.5/1 5/200/0.5/1 6/200/0.5/1 7/200/0.5/1 8/200/0.5/1 9/200/0.5/1 10/200/0.5/1`
- `python plotting/create_summary_data.py data/agentperformance/linearroad/0/200/0.5/1:0 data/agentperformance/linearroad/1/200/0.5/1:1 data/agentperformance/linearroad/2/200/0.5/1:2 data/agentperformance/linearroad/3/200/0.5/1:3 data/agentperformance/linearroad/4/200/0.5/1:4 data/agentperformance/linearroad/5/200/0.5/1:5 data/agentperformance/linearroad/6/200/0.5/1:6 data/agentperformance/linearroad/7/200/0.5/1:7 data/agentperformance/linearroad/8/200/0.5/1:8 data/agentperformance/linearroad/9/200/0.5/1:9 data/agentperformance/linearroad/10/200/0.5/1:10 data/agentperformance/linearroad/baselines_data.csv`
- Then run the agent-based version for the different policies, using start_all.sh
  - `./scripts/start_all.sh LinearRoad WELOB`
  - `./scripts/start_all.sh LinearRoad ELOB`
  - `./scripts/start_all.sh LinearRoad LOB`
  - `./scripts/start_all.sh LinearRoad WELAW`
  - `./scripts/create_plots_for_exp.sh data/agentperformance_dqn True WELOB/linear ELOB/linear LOB/linear WELAW/linear`
  - `python plotting/create_summary_data.py data/agentperformance_dqn/WELOB/linear:welob_linear data/agentperformance_dqn/ELOB/linear:elob_linear data/agentperformance_dqn/LOB/linear:lob_linear data/agentperformance_dqn/WELAW/linear:welaw_linear data/agentperformance_dqn/baselines_data.csv`
  - `python plotting/paper_plot_baselines_vs_multiple_agents_icpe_single_column.py data/agentperformance/linearroad data/lr_rate.csv data/agentperformance_dqn/baselines_data.csv data/agentperformance_dqn/lr_baseline_vs_multiple_agents.pdf data/agentperformance_dqn/lr_baseline_vs_multiple_agents.png linearroad welob_linear,elob_linear,lob_linear,welaw_linear WEL-OB,EL-OB,L-OB,WEL-AW`

# Actions (Figure 8)
- To plot such a graph, extrac the actions from an episode (make an excerpt of the actions.csv file in the corresponding folder) and then run:
- `python plotting/plot_episode_actions.py data/agentperformance_dqn/WELAW/linear/actions.excerpt.csv data/agentperformance_dqn/actions.pdf`

# State creation times (Figure 10)
- To plot such graph, extract the required data for both Linear and synthetic like this
- `./scripts/extract_state_initilization_csv.sh data/agentperformance_dqn/WELAW/linear/spe.log data/agentperformance_dqn/WELAW/linear/statecreationtimes.csv`
- and then run
- `python plotting/plot_state_initilization_times.py data/agentperformance_dqn/WELAW/linear/statecreationtimes.csv data/agentperformance_dqn/WELAW/linear/statecreationtimes.csv data/agentperformance_dqn/statecreationtimes.pdf`
  - Note in this case I am passing twice the linear road data, pass the synthetic one once you create it

# Scalability Experiments
- These are started using the start_all_evaluation_CCR.sh script.
  - The experiment config needs to be added in the beginning. The script contains a sample setup to compare no agent vs. agent for a given compression for Linear Road
- To then create the plots, you can use the following scripts (these are using the folders given in the sample `start_all_evaluation_CCR.sh` script)
  - `./scripts/create_plots_for_exp.sh data/scalability/linearroad/3/80/0.5 True 1 2 3 4 5`
  - `./scripts/create_plots_for_exp.sh data/scalability/linearroad/3/2/80 True 1 2 3 4 5`
  - `python plotting/merge_overhead_data.py data/scalability/linearroad/3/2/80 data/scalability/linearroad/3/80/0.5 1 2 3 4 5 data/scalability/linearroad/merged.csv`
  - `python plotting/compute_overheads_from_merged_data.py data/scalability/linearroad/merged.csv data/scalability/linearroad/diffs.csv`
  - `python plotting/plot_overheads.py data/scalability/linearroad/diffs.csv data/scalability/linearroad/plot.pdf`
- To create the same plot you find in the paper, you can then use (notice in the paper the experiments are for both linear road and synthetic. In the command below just reusing twice the linear road data for simplicity)
  - `python plotting/plot_overheads_combined.py data/scalability/linearroad/diffs.csv data/scalability/linearroad/diffs.csv data/scalability/plotcombined.pdf`


If you want to measure the CPU and Memory consumption of the Agent, then:
- uncomment the last command in start_python_agent.sh
- start an experiment with the agent running, for instance `./scripts/start_all.sh LinearRoad WELAW`  