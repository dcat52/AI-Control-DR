## Adding parallel config ##
Create this file: `~/.parallel/config`

Add the following line:
```C++
--rpl '{0#} 1 $f=1+int((log(total_jobs())/log(10))); $_=sprintf("%0${f}d",seq())'
```

## Generating a CSV ##
To generate a csv to run jobs, do something like this (Note: the `{000#}` requires the configuration above or aditional arguments):
```C++
echo "JN, ARGS" > params.csv
parallel echo {0#}, --update_freq {1} --tau {2} ::: {1..9} ::: 0.{1..9..2} 1.0 >> params.csv
```

It can then be read like so:
```C++
parallel --header : --colsep ', ' echo JOB: {JN}    ARGS: {ARGS} :::: params.csv
```

With this method, the parameters are defined at dataset generation time, not at script runtime. This means any parameter not mentioned will take the default (except for a select few that we want to change regardless).