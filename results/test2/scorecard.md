| Objective | Experiments | Comparisons | GUARD better | n.s. | GUARD worse | Baselines significantly better |
|---|---|---|---|---|---|---|
| timely delivery ratio TDR(1 s) (in E4 the denominators differ between methods) | E1,E2,E3,E4,E8 | 160 | 159 | 1 | 0 | - |
| timely delivery ratio TDR(0.5 s) | E1 | 48 | 48 | 0 | 0 | - |
| timely delivery ratio TDR(2 s) | E1 | 48 | 48 | 0 | 0 | - |
| packet delivery ratio | E1,E2,E3,E4,E8 | 160 | 158 | 1 | 1 | bcp |
| throughput | E1 | 48 | 48 | 0 | 0 | - |
| fairness (Jain index) | E1,E2,E3 | 128 | 128 | 0 | 0 | - |
| packet loss ratio | E1 | 48 | 48 | 0 | 0 | - |
| mean delay | E1 | 48 | 21 | 8 | 19 | acsro, cetx, fsfg, ganaco, rlcc, twinloop |
| 95th-percentile delay | E1,E2,E3,E8 | 152 | 47 | 21 | 84 | acsro, cbpr, cetx, fsfg, ganaco, rlcc |
| 99th-percentile delay | E1 | 48 | 15 | 10 | 23 | acsro, cetx, fsfg, ganaco, rlcc |
| energy per delivered reading | E1,E2,E3,E4,E8 | 160 | 145 | 1 | 14 | bcp |
| collided frames (all receivers) | E1,E2 | 80 | 78 | 1 | 1 | cbpr |
| collided frames (sink) | E1 | 48 | 44 | 4 | 0 | - |
| frames per delivered reading | E1 | 48 | 38 | 2 | 8 | bcp, twinloop |
| control bytes | E1 | 48 | 0 | 0 | 48 | acsro, bcp, cbpr, cetx, fsfg, ganaco, rlcc, twinloop |
| total protocol overhead bytes | E1 | 48 | 15 | 0 | 33 | acsro, bcp, cetx, fsfg, ganaco, rlcc |
| lifetime: first node dead | E4 | 8 | 8 | 0 | 0 | - |
| lifetime: alive fraction at 900 s | E4 | 8 | 6 | 2 | 0 | - |
| lifetime: timely readings delivered | E4 | 8 | 7 | 1 | 0 | - |
| lifetime: readings delivered | E4 | 8 | 6 | 1 | 1 | bcp |
| lifetime: time with >= 90 % delivery | E4 | 8 | 8 | 0 | 0 | - |
| lifetime: time with >= 50 % delivery | E4 | 8 | 6 | 1 | 1 | bcp |
| critical-reading delivery ratio (events) | E3 | 8 | 8 | 0 | 0 | - |
| critical-reading 95th-percentile delay (events) | E3 | 8 | 3 | 0 | 5 | acsro, cetx, fsfg, ganaco, rlcc |
