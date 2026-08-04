# Records the README demo: a real measured run against two installed models.
# Regenerate with:  vhs demo/autopilot.tape
Output demo/autopilot.gif
Set Shell powershell
Set FontSize 15
Set Width 1180
Set Height 620
Set Padding 12
Set Theme "Catppuccin Mocha"

Type "routefoundry autopilot --limit 4 --models gemma:2b,llama3.2:3b --yes"
Sleep 800ms
Enter
Sleep 45s
