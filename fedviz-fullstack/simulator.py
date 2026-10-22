import math, random

def clamp(x, lo, hi): return max(lo, min(hi, x))

def run_experiment(c):
    clients = int(clamp(float(c.get("clients", 8)), 2, 50))
    rounds = int(clamp(float(c.get("rounds", 20)), 1, 200))
    part = clamp(float(c.get("participation", 75)), 10, 100)
    noise = clamp(float(c.get("noise", .8)), 0, 3)
    clip = clamp(float(c.get("clip", 1)), .1, 5)
    threshold = clamp(float(c.get("threshold", .15)), 0, .9)
    dataset = str(c.get("dataset", "UNSW-NB15"))
    distribution = str(c.get("distribution", "Non-IID"))
    random.seed(int(c.get("seed", 42)) + clients*3 + rounds + int(noise*100))
    difficulty = {"UNSW-NB15": 0, "TON-IoT": -1.5, "Balanced synthetic": 3}.get(dataset, 0)
    heterogeneity = 2.5 if distribution == "Non-IID" else .5
    base = 76.5 + difficulty + min(9, math.log1p(rounds)*1.9) + min(3.5, math.log1p(clients)*.95) + part/100*2 - heterogeneity
    fed = clamp(base+.2, 60, 97)
    rows = [
      ("Centralized MLP", clamp(base+1.5,65,98),False,False,False),
      ("FedAvg",fed,False,False,False),
      ("FedAvg + DP",fed-(1.8+noise*2+max(0,1-clip)*.4),True,False,False),
      ("FedAvg + Secure Aggregation",fed-.3,False,True,False),
      ("FedAvg + DP + Secure Aggregation",fed-(2+noise*2+max(0,1-clip)*.4),True,True,False),
      ("Full + Adaptive Communication",fed-(2+noise*2+max(0,1-clip)*.4)+threshold*1.7+(1-part/100*(1-threshold*.62))*1.2,True,True,True)]
    strategies=[]
    for name, acc, dp, sec, adapt in rows:
        acc=round(clamp(acc+random.uniform(-.4,.4),45,99),2)
        p=round(clamp(acc-random.uniform(.3,1.8),0,100),2)
        r=round(clamp(acc-random.uniform(.4,2),0,100),2)
        strategies.append({"strategy":name,"accuracy":acc,"precision":p,"recall":r,"f1":round(2*p*r/max(p+r,.01),2),"dp":dp,"secure_aggregation":sec,"adaptive_communication":adapt})
    comm_ratio=(part/100)*(1-threshold*.62)
    saved=round(clamp((1-comm_ratio)*100,0,95),1)
    selected=max(1,round(clients*part/100))
    return {"mode":"synthetic-demo","notice":"Illustrative metrics only; no real training, cryptographic secure aggregation, or formal DP accounting.","config":{"clients":clients,"rounds":rounds,"participation":part,"noise":noise,"clip":clip,"threshold":threshold,"dataset":dataset,"distribution":distribution},"metrics":{"best_accuracy":strategies[-1]["accuracy"],"communication_saved_pct":saved,"data_kept_local_pct":100,"rounds":rounds,"selected_clients_per_round":selected},"strategies":strategies,"round_history":[{"round":i,"accuracy":round(54+(strategies[-1]["accuracy"]-54)*(1-math.exp(-i/max(rounds/4,1))),2)} for i in range(1,rounds+1)],"logs":[f"Initialized {clients} synthetic client organizations.",f"Dataset: {dataset}; distribution: {distribution}.",f"Selected {selected}/{clients} clients per round.",f"Simulated FedAvg for {rounds} rounds.",f"DP concept configured with noise={noise:.2f}, clip={clip:.2f}.","Secure-aggregation concept enabled; raw records stay local in the simulation.",f"Adaptive threshold={threshold:.2f}; estimated communication saving={saved:.1f}%.","Synthetic calculation complete; no real model training or privacy guarantee."]}
