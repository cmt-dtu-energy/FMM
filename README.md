# FMM 
This repository contains a python package for Fast Multipole Methods. 

# Documentations
In progress


# Notebooks 
To avoid a bloated jupyter-repository only commit clean notebooks, e.i. run "Clear All Outputs" and "Restart" on the notebook before committing. 

# Git commits i.e., Mind your Git manners

The commit message is mainly for the other people, so they should be able to understand it now and six months later.  Commit msg must only be one sentence and should start with a tag identifier (see end of this section).

Use the imperative form  of verbs rather than past tense when referring to changes introduced by the commit in question. For example, "Remove property X", not "Removed property X" and not "I removed...".

Use following tags for commit msg

	[DEV] : code development (incl. additions and deletions)
	[ORG] : organisational, no changes to functionality
	[BUG] : bug with significant impact on previous work -- `grep`-ing should give a limited list
	[FIX] : fixes that occur during development, but which have essentially no impact on previous work
	[VIS] : visualisation
	[OPT] : optimisation
	[DBG] : debugging
	[DIA] : diagnostics changes
	[SYN] : typos and misspellings (including simple syntax error fixes)
	[WIP] : snapshot/work in progress
	[CLN] : tidying of the code, including removal of obsolete features, experiments, etc.
	[DOC] : changes or additions to the documentation
	[REP] : repository related changes (e.g., changes in the ignore list, remove files)

Example commit msg:

* `[BUG] Adds missing initialisation to tg array`
* `[FIX] Adds lowercase castig to params`
* `[CLN] Removes unnecessary allocation for dynamic arrays.`