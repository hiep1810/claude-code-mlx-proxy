---
trigger: always_on
---

# Objective
Eliminate hallucinations and wasted compute by forcing explicit alignment before execution.

# Protocol
Before providing a full solution, the agent MUST output the following structure:

## 🛠️ Execution Status: [Pending / Draft / Blocked]
State exactly what stage the response is in.

### 🔍 Points of Uncertainty
* Identify any missing variables, ambiguous terms, or technical preferences.
* List assumptions that *would* have been made.

### ⚠️ Potential Effects of Uncertainty
* Explain how the final output changes based on the unknown variables.
* Use a table if there are more than two variables.

# Constraints
* Never "fill in the blanks" for the user without stating it is a placeholder.
* If a task is 100% clear, the status should be [Executing] and the uncertainty section should state "None identified."