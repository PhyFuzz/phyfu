=====
Usage
=====

Fuzzing
------------
To run fuzzing, use the following command:

.. code-block:: console

    phyfu.fuzz [pse] [scenario] --test_times 10000 --seed_getter [art/random]
    
- pse: the name of the PSE
- scenario: the name of the physical scenario.
- [art/random]: the type of seed generator. "art" means using seed scheduling. "random" means without seed scheduling.

Valid combinations of pse and scenario:

- [taichi/brax/warp/nimble] two_balls
- brax ur5e
- warp snake
- nimble catapult

The command above would run the fuzzing for 10000 times. You can change the number by changing the value of "--test_times". The seed generator is set to "art" by default. You can change it to "random" by changing the value of "--seed_getter".

Upon the completion of the fuzzing, the console would print the following messages:

.. code-block:: text

    #loss_too_large: xxx
    #deviated_init_state: xxx

The ``#loss_too_large`` means the number of backward errors discovered from the fuzzing campaign. The ``#deviated_init_state`` means the number of forward errors.

The results and the log information about each fuzzing iteration will be saved in the directory ``phyfu/results/[pse]/[scenario]/[time_stamp]``, where the ``time_stamp`` is the start time of the fuzzing formated as ``MMdd_hhmm``. The log information includes the seed used in each iteration, the loss, the gradient, and parameter value under optimization of each iteration. Also, information about all the errors would be saved in the file ``data_analysis.txt`` under the same directory.

Running fuzzing for taichi DiffMPM requires special handling, as it can crash the GPUs. Use the following command to run the fuzzing for taichi DiffMPM:

.. code-block:: console

    phyfu.fuzz_mpm taichi mpm --operation fuzzing --test_times 10000 --seed_getter [art/random]

To obtain the number of errors found by the fuzzing campaign of taichi DiffMPM, use the following command:

.. code-block:: console

    phyfu.fuzz_mpm taichi mpm --operation find_errors --time_stamp [time_stamp]

, where ``time_stamp`` is the start time of the fuzzing formated as ``MMdd_hhmm`` (``MM`` means month, ``dd`` means day, ``hh`` means hours, ``mm`` means minutes). After that, you will see the total number of forward and backward errors found by in the DiffMPM fuzzing campaign.


Data Analysis
----------------
After running the fuzzing, you can use the following command to analyze the data:

.. code-block:: console

    phyfu.analyze [pse] [scenario]

Valid combinations of pse and scenario are the same as the fuzzing command.

The analysis process may take dozens of minutes to several hours, depending on the pse, scenario, and the powerfulness of your machines. The console would print the following messages when the analysis is done:

.. code-block:: text
 
    Module: [pse], model_name: [scenario]
    Total number of errors found by art: xxx
    Total number of crashes found by art: xxx
    Total number of errors found by random: xxx
    Total number of crashes found by random: xxx
    Total execution time: xxhr xxmin xxsec
    Overhead of seed scheduling for [pse] [scenario]: xx minutes
    #Position errors: xxx
    #Velocity errors: xxx
    #Gradient direction errors: xxx
    #Gradient extent errors: xxx
    #Unapparent errors: xxx

The meaning of each line is self-explanatory. Note that the lines with "random" may not appear if you have not run the fuzzing with random seed generator.
The last five lines are the distribution of errors found by the fuzzing campaign. The "overhead of seed scheduling" is the time spent on seed scheduling. The total execution time is the time spent on running the fuzzing. The overhead of seed scheduling is usually less than 1% of the total execution time.
