"""PC tools. They run in the toolhost process (the only one that touches the PC on the agent's behalf),
each gated by the task's capability scope and the safety policy; core keeps a few local tools."""
