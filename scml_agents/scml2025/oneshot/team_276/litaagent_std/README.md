# SCML 2024 Agent Skeleton (Std)

This skeleton contains the following folders/files:

1. _myagent.py_ : A skeleton for an agent for the Std track.
1. _helpers_: A folder containing a script to try an agent.
1. report: A folder with latex and docx files that you can use to write
   your 2-4 pages report. Please remember to submit a `pdf` version of the
   report. A sample PDF is also provided for reference.

# Using the Skeleton

To develop your agent, the only required steps are the following:

1. \[recommended\] create a virtual environment

- Install venv

  > python3 -m venv .venv

- Activate the virtual environment:

  - On linux

    > source .venv/bin/activate

  - On windows (power shell)

    > call .venv\\bin\\activate.bat

2. \[required\] **Install scml**

   > pip install scml

1. \[recommended\] Test the installation by running a simple simulation

   > scml run2024 --std --steps=10

1. \[recommended\] Change the name of the agent class from \`MyAgent' to
   your-agent-name.

1. Change the implementation of whatever functions you need in the provided
   factory manager

1. \[recommended\] Modify the name of either `../report/myagent.tex` or
   `../report/myagent.docx` to `../report/your-agent-name.tex` or
   `../report/your-agent-name.docx` as appropriate and use it to write your
   report.

1. \[recommended\] You can run a simple tournament of your agent against basic
   strategies by either running `myagent.py` from the command line

   > python -m myagent.myagent

   or using the CLI as described in the documentation

1. \[required\] **Submit your agent**: After developing your agent,
   zip `your-agent-name` folder into `your-team-name_your-agent-name.zip`
   (with the pdf or the report included) and submit it along with
   `your-agent-name.pdf` (after generating it from the tex or docx file).
   This is the only file you need to submit.

_Submissions start on January 1st, 2024 at <https://scml.cs.brown.edu>_

## Agent Information

Fill this section with your agent information

- Agent Name: my-agent-name
- Team Name: my-team-name
- Contact Email: <my-email@some-server.xyz>
- Affiliation: Institute, Department
- Country: country-name
- Team Members:
  1. First Name <first.email@institute.xyz>
  1. Second Name <first.email@institute.xyz>
  1. ...
