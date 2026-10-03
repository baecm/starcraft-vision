"""One script per figure of the Director-CenterNet paper and the thesis.

Shared style, data access and drawing helpers live in `figures.common`; each
other module draws one figure and runs on its own (see its docstring), and
`scripts/qualitative_figures.py` dispatches to them under the old subcommand
names. `render_all.sh` holds the commands that produced the published figures.
"""
