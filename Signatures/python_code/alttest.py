import altair as alt
import pandas as pd
import numpy as np
import altair_viewer
import matplotlib.pyplot as plt

 

data = pd.read_csv( '/Users/hagitbenshoshan/Documents/DHTA/DHTA/Signatures/results_cod/jensenshannon.csv')
"""print (data)
# Create the heatmap using Altair
heatmap = alt.Chart(data).mark_rect().encode(
    x='Column2:O',  # Nominal data on the x-axis
    y='ATT:O',  # Nominal data on the y-axis
    color='VAL:Q'  # Quantitative data mapped to color
).properties(
    width=400,
    height=300,
    title='Heatmap Example using Altair'
)

# Show the heatmap
heatmap.display()
heatmap.save('heatmap.png')
""" 
# Create heatmap
plt.imshow(data, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.title("Matplotlib Heatmap Example")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the heatmap using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(data, annot=True, cmap='coolwarm', cbar=True)
plt.title('Sample Heatmap')
plt.show()