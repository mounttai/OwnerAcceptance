The code and simulated data are for the purpose of replicating the main results in the paper:

Yao, Dai and Tang, Chuang and Chu, Junhong, A Dynamic Model of Owner Acceptance in Peer-to-Peer Sharing Markets (March 8, 2022). *Marketing Science*, Forthcoming, Available at SSRN: [https://ssrn.com/abstract=4052540](https://ssrn.com/abstract=4052540).

Feel free to contact Dai via [email](mailto:dai@yaod.ai), or visit [http://wwww.yaod.ai](http://wwww.yaod.ai) for more contact information.

Due to a non-disclore agreement with the platform, we cannot make the data public. Hence, we share the implementation of the estimation algorithm in Matlab, as well as the simulation data and estimation results on the simulated data reported in Table A10 and A11. Interested users could easily simulate any data under other values of the parameters in the model than the two sets of values we used in the paper, and estimate different models (myopic owners only, forward-looking owners only, mixture of owners) with the code. Extensive comments were provided inside all the code files to facilitate understanding and execution of the code files.


# code files for the implementation of the estimation algorithm

* EstMyopicModel.m - estimate the model assuming only myopic owners
* EstStrategicModel.m - estimate the model assuming only strategic owners
* EstMixedModel.m - estimate the model assuming a mixture of both types of owners

* utils/ObjMyopicModel.m - objective function of the model assuming only myopic owners
* utils/ObjStrategicModel.m - objective function of the model assuming only strategic owners
* utils/ObjSegmembership.m - objective function of the segment memberships
* utils/ComputeLLMyopic.m - compute the logliklihood for myopic owners
* utils/ComputeLLStrategic.m - compute the logliklihood for strategic owners
* utils/ComputeU.m - compute the current period utility
* utils/ComputeW.m - compute the W(*) values using successive approximation
* utils/SetupCoreEnv.m - setup all relevant variables for the estimation of the model

notes: (1) to run estimation on different data, change "data_type" to other values at line 13 in the first three files starting with "Est"; (2) all the estimation results are stored under "results" folder.


# code files for generating simulated data

* SimData_SameParas.m - simulate data where the utility parameters take the same values for both myopic and strategic owners
* SimData_DiffParas.m - simulate data where the utility parameters take different values

notes: (1) the two files are exactly the same except lines 30-31 where the utility parameters are defined, as well as line 33 where the parameters for the segment memberships are defined; (2) lines 16-18 in both files determine whether the owners are all myopic, all strategic, or mixed; (3) all the simulated data are stored under "data" folder.


# simulated data used in the paper (i.e., Table A10 and A11)

* data-results-same-paras/ - in this folder, all "data-*.mat" files are simulated, while all "est-*.mat" files are estimation results on these simulated data
* data-results-diff-paras/ - this folder has the same structure as the one above.


**Please see the paper for more details on the model. If you use these codes files or data, please CITE this paper as the source.**

