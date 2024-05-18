
def graph_result(optimizer,path):
    optimizer.history.save_global_objectives_chart(filename=f"Graphs/{path}/goc")
    optimizer.history.save_local_objectives_chart(filename=f"Graphs/{path}/loc")
    optimizer.history.save_global_best_fitness_chart(filename=f"Graphs/{path}/gbfc")
    optimizer.history.save_local_best_fitness_chart(filename=f"Graphs/{path}/lbfc")
    optimizer.history.save_runtime_chart(filename=f"Graphs/{path}/rtc")
    optimizer.history.save_exploration_exploitation_chart(filename=f"Graphs/{path}/eec")
    optimizer.history.save_diversity_chart(filename=f"Graphs/{path}/dc")