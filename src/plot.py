#this file contains functions that help with plotting and data visualization

import os
import glob

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import seaborn as sns


from src.timestamps import check_sunup
from src.parameters import load_json
from src.spectrogramming import specs_from_wavs
from src.rfid import get_quadrant_from_box, get_boxes_from_quadrant
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates



def recorded_boxes(data):
    counts = {}

    # Calculate total occurrences for each unique number across audiomoths
    for audiomoth, values in data.items():
        for count in values.values():
            if count in counts:
                counts[count] += 1
            else:
                counts[count] = 1

    # Sort the counts dictionary by keys (unique numbers) in ascending order
    sorted_counts = dict(sorted(counts.items()))

    # Plot the sorted counts
    plt.barh(range(len(sorted_counts)), list(sorted_counts.values()))
    plt.ylabel("Unique Number")
    plt.xlabel("Count")
    plt.title("Total times recorded by box")
    plt.yticks(range(len(sorted_counts)), list(sorted_counts.keys()))
    plt.show()
    
    # Now iterate over each audiomoth and create a separate plot
    for audiomoth, values in data.items():
        counts = {}

        # Calculate total occurrences for each unique number
        for count in values.values():
            if count in counts:
                counts[count] += 1
            else:
                counts[count] = 1

        # Sort the counts dictionary by keys (unique numbers) in descending order
        sorted_counts = dict(sorted(counts.items(), reverse=True))

        # Filter and plot only even values between 2 and 40
        x = []
        y = []
        for num in range(2, 41, 2):
            if num in sorted_counts:
                x.append(sorted_counts[num])
                y.append(num)

        # Plot the filtered counts
        plt.figure()  # Create a new figure for each audiomoth
        plt.barh(y, x)
        plt.xlabel("Count")
        plt.ylabel("Unique Number")
        plt.title(f"Time box recorded - {audiomoth}")
        plt.xticks(range(0, max(x) + 1))
        plt.yticks(range(2, 41, 2))

    # Show all the plots
    plt.show()

    
def get_box_colors(save=False):
    
    unique_boxes = list(np.arange(2,42,2))
    
    # Define the color maps for each quadrant
    colormaps = ['Blues', 'Oranges', 'Greens', 'RdPu']

    # Create the dictionary to store box-color mappings
    box_color_map = {}

    # Assign colors to boxes based on their quadrant
    for box in unique_boxes:
        # Find the quadrant of the box
        quadrant = get_quadrant_from_box(box)
        
        # Get the colormap corresponding to the quadrant
        colormap = colormaps[quadrant - 1]
        
        # Get the boxes in this quadrant
        boxes_in_quadrant = get_boxes_from_quadrant(quadrant)

        # Get the position of the box within the unique boxes in the quadrant
        box_position = boxes_in_quadrant.index(box)

        # Calculate the normalized value based on the box position
        normalized_value = (box_position / (len(unique_boxes) - 1))*2+.2

        # Get the color for the box based on the adjusted value in the colormap
        color = plt.get_cmap(colormap)(normalized_value)

        # Store the box-color mapping in the dictionary
        box_color_map[box] = color

    # Sort the keys in the dictionary from smallest to largest
    box_color_map = {k: box_color_map[k] for k in sorted(box_color_map)}

    if save:
        parameters.save_json(box_color_map, save_dir='/Volumes/LaCie_barn/mouse_barn_audiomoth/parameters/', save_name='box_colors.json')
        return box_color_map
    
    return box_color_map

    
def barn_layout(barriers, boxes, passages, fig, ax, box_colors, to_scale=False, aspect=0.79, color_boxes=False):
    
	# Set the x-axis and y-axis limits
	ax.set_xlim(0, barriers['x'].max())
	ax.set_ylim(barriers['y'].max(), 0)

	# Set the aspect ratio for a scale
	if to_scale:
		ax.set_aspect(barriers['y'].max() / barriers['x'].max())
	else:
		#or set custom
		ax.set_aspect(aspect)

	# Draw lines for the barriers (red)
	for i in barriers.loc[barriers['type'] == 3, 'name'].unique():
		ax.plot(barriers.loc[barriers['name'] == i, 'x'], barriers.loc[barriers['name'] == i, 'y'],
				 linewidth=1, color="black")

	for i in barriers.loc[barriers['type'] == 4, 'name'].unique():
		ax.plot(barriers.loc[barriers['name'] == i, 'x'], barriers.loc[barriers['name'] == i, 'y'],
				 linewidth=1, color="black", zorder=1)

	# Draw polygons for the entry way and back shelf
	for i in barriers.loc[barriers['type'] == 2, 'name'].unique():
		if i == 'entrance':
			ax.fill(barriers.loc[barriers['name'] == i, 'x'], barriers.loc[barriers['name'] == i, 'y'],
					 color="white", zorder=1)

			ax.plot(barriers.loc[barriers['name'] == i, 'x'], barriers.loc[barriers['name'] == i, 'y'],
				 linewidth=1, linestyle='--', color="black")

	#drop the odd numbered boxes that don't exist anymore
	#boxes = boxes[boxes['box'].isin(np.arange(2,42,2))]

	# Plot the location of the passages
	ax.scatter(passages['x'], passages['y'], marker="s", color="white", s=4**2, zorder=10)

	# Plot points for box positions (colored circles)
	for x, y, box in zip(boxes['x'], boxes['y'], boxes['box']):
		
		#nudge boxes so they don't overlap
		if box == 4:
			y -= 0.1
		elif box == 2:
			y += 0.1
		elif box == 20:
			x -= 0.15
		elif box == 26:
			x -= 0.1
			y -= 0.2
		elif box == 22:
			y -= 0.1
#		elif box == 29:
#			y += 0.1
#		elif box == 23:
#			y -= 0.1
#		elif box == 39:
#			y -= 0.2
#			x += 0.2
		
		if color_boxes:
			
			color = box_colors[box]  # Get the color from the box_color_map

			# Check if box is odd or even for linestyle
			if box % 2 != 0:  # Odd numbered box
				scatter = ax.scatter(x, y, marker='o', facecolor=color, edgecolor='black', linewidth=0.5, s=13**2, zorder=10, linestyle='--')
			else:  # Even numbered box
				scatter = ax.scatter(x, y, marker='o', facecolor=color, edgecolor='black', linewidth=0.5, s=13**2, zorder=10)

		else:

			# Check if box is odd or even for linestyle
			if box % 2 != 0:  # Odd numbered box
				scatter = ax.scatter(x, y, marker='o', facecolor=color, edgecolor='black', linewidth=0.5, s=13**2, zorder=10, linestyle='--')
			else:  # Even numbered box
				scatter = ax.scatter(x, y, marker='o', facecolor=color, edgecolor='black', linewidth=0.5, s=13**2, zorder=10)

	# Add text labels for box numbers
	for x, y, box in zip(boxes['x'], boxes['y'], boxes['box']):

		#nudge boxes so they don't overlap
		if box == 4:
			y -= 0.1
		elif box == 2:
			y += 0.1
		elif box == 20:
			x -= 0.15
		elif box == 26:
			x -= 0.1
			y -= 0.2
		elif box == 22:
			y -= 0.1

		ax.text(x, y, str(box), horizontalalignment='center', verticalalignment='center', zorder=20, fontsize=9)

	# Add text
	ax.text(5.0, .775, 'nest box', fontsize=9, zorder=20, color="darkgrey")
	ax.text(5.1, 5, 'door/entry area', fontsize=9, zorder=20, color="darkgrey")
	ax.text(3.6, 2.2, 'internal wall', fontsize=9, zorder=20, color="darkgrey")


	# Show the plot
	# Remove ticks
	ax.set_xticks([])
	ax.set_yticks([])

	# Remove tick labels
	ax.set_xticklabels([])
	ax.set_yticklabels([])
    

def season_confusion_matrices(squeak_axis, USV_axis, squeak_model, USV_model, squeak_data, USV_data, training_params, normalize_over = 'columns'):

	# reproduce the USV data split (to generate the confusion matrices)               
	USV_X = np.array(USV_data[training_params['features']])
	USV_y = np.array(USV_data[training_params['target']])
	USV_X_train, USV_X_test, USV_y_train, USV_y_test = train_test_split(USV_X, 
																		USV_y, 
																		test_size = float(training_params['test_size']), 
																		random_state = int(training_params['split_random_state']))

	# reproduce the squeak data split (to generate the confusion matrices)    
	squeak_X = np.array(squeak_data[training_params['features']])
	squeak_y = np.array(squeak_data[training_params['target']])
	squeak_X_train, squeak_X_test, squeak_y_train, squeak_y_test = train_test_split(squeak_X, 
																		squeak_y, 
																		test_size = float(training_params['test_size']), 
																		random_state = int(training_params['split_random_state']))


	# Evaluate
	labels = ['summer_22', 'autumn_22', 'winter', 'spring', 'summer_23']
	squeak_y_pred = squeak_model.predict(squeak_X_test)
	USV_y_pred = USV_model.predict(USV_X_test)

	if normalize_over == 'rows':
		squeak_cm = confusion_matrix(squeak_y_test, squeak_y_pred, labels=labels, normalize='true')
		squeak_cm_df = pd.DataFrame(squeak_cm, columns=labels, index=labels)
		USV_cm = confusion_matrix(USV_y_test, USV_y_pred, labels=labels, normalize='true')
		USV_cm_df = pd.DataFrame(USV_cm, columns=labels, index=labels)

	elif normalize_over == 'columns':
		squeak_cm = confusion_matrix(squeak_y_test, squeak_y_pred, labels=labels, normalize='pred')
		squeak_cm_df = pd.DataFrame(squeak_cm, columns=labels, index=labels)

		USV_cm = confusion_matrix(USV_y_test, USV_y_pred, labels=labels, normalize='pred')
		USV_cm_df = pd.DataFrame(USV_cm, columns=labels, index=labels)

	# Plot squeak confusion matrix
	sns.heatmap(squeak_cm_df,
				annot=True,
				annot_kws={"size": 6, "color": 'lightgrey'},
				fmt=".2f",
				cmap='viridis',
				xticklabels=True,
				yticklabels=True,
				vmin=0,
				vmax=1,
				square=True,
				ax=squeak_axis,
				cbar=False)

	# Plot USV confusion matrix
	sns.heatmap(USV_cm_df,
				annot=True,
				annot_kws={"size": 6, "color": 'lightgrey'},
				fmt=".2f",
				cmap='viridis',
				xticklabels=True,
				yticklabels=True,
				vmin=0,
				vmax=1,
				square=True,
				ax=USV_axis,
				cbar=False)

	for ax in [squeak_axis, USV_axis]:
		ax.set_xticklabels(["sum'22", 'fall', 'winter', 'spring', "sum'23"])
		
	squeak_axis.set_yticklabels(["sum'22", 'fall', 'winter', 'spring', "sum'23"])
	USV_axis.set_yticklabels(["", "", "", "", ""])
	USV_axis.set_ylabel("")
	

def animate_barn(df, start, stop, coords, framerate=1, interval=10):

    """
    Animate the barn floor with nest boxes based on the provided dataframe.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        start (pd.Timestamp): The starting time for the animation.
        stop (pd.Timestamp): The stopping time for the animation.
        coords (dict): The coordinates of the nest boxes as {box_name: (x, y)} dictionary.
        framerate (float, optional): The framerate of the animation. Defaults to 1.
        inetrval (float, optional): How many minutes between each frame. Defaults to 10.

    Returns:
        matplotlib.animation.FuncAnimation: The animation object.
    """
    
    #get max count
    min_count = 0
    max_count = df['occupants_count'].max()
    
    # Filter dataframe based on start and stop times
    df['minute'] = pd.to_datetime(df['minute'])  # Convert to datetime type
    df = df[(df['minute'] >= pd.to_datetime(start)) & (df['minute'] <= pd.to_datetime(stop))]
    
    # Set frames argument
    frames = sorted(df['minute'].unique())[::interval]

   # Create figure and axes
    fig, ax = plt.subplots(figsize=(12.74, 5.77))

    # Set the aspect ratio of the plot
    plot_width = 13
    plot_height = 6
    ax.set_xlim(-0.5, plot_width)
    ax.set_ylim(-0.5, plot_height)
    ax.set_aspect('equal')

    # Initialize the plot elements
    boxes = {}
    texts = {}

    # Iterate over the box coordinates and create circle patches
    for box_name, (x, y) in coords.items():
        # Create the circle patch with specified coordinates
        box = plt.Circle((x, y), 0.3, color='lightgray', edgecolor='black', linewidth=0)
        text = ax.text(x-0.35, y+0.3, box_name, ha='center', va='center', fontsize=4)

        # Add the box patch to the plot
        ax.add_patch(box)
        boxes[box_name] = box
        texts[box_name] = text

    # Update function for animation
    def update(frame):
        # Clear the previous plot
        ax.clear()

        # Set the limits of the plot
        ax.set_xlim(-0.5, plot_width)
        ax.set_ylim(-0.5, plot_height)
        ax.set_aspect('equal')

        # Add the boxes and texts to the plot
        for box in boxes.values():
            ax.add_patch(box)
        for text in texts.values():
            ax.add_artist(text)

        # Filter dataframe for the current frame
        frame_df = df[df['minute'] == frame]

        for _, row in frame_df.iterrows():
            box_name = (' ').join(['box',str(row['box'])])
            x, y = coords[box_name]

            # Get the count value
            count = row['occupants_count']

            # Set the color based on the count using the viridis colormap
            color = plt.cm.GnBu(count / max_count)  # Adjust max_count as needed

            # Create the circle patch with the specified coordinates and color
            box = plt.Circle((x, y), 0.3, edgecolor='black', linewidth=0, facecolor=color)
            
            #add text to the circle indicating how many mice are in the box
            text = ax.text(x, y, str(count), ha='center', va='center', fontsize=8)
            
            # Add the box patch and text to the plot
            ax.add_patch(box)
            ax.add_artist(text)

        # Display the timestamp
        timestamp = frame_df.iloc[0]['minute']
        ax.text(0.9, -.25, str(timestamp), ha='center', va='center', fontsize=10)
        
        # Check sunup value and print NIGHT or DAY
        sunup_value = frame_df.iloc[0]['sunup']
        if sunup_value == 0:
            ax.text(0.025, 0.95, "NIGHT", ha='left', va='top', transform=ax.transAxes, fontsize=16, color='dimgrey')
        else:
            ax.text(0.025, 0.95, "DAY", ha='left', va='top', transform=ax.transAxes, fontsize=18, color='darkorange')
    
    # Add colorbar
    cmap = plt.cm.ScalarMappable(cmap='GnBu', norm=plt.Normalize(vmin=min_count, vmax=max_count))
    cbar = plt.colorbar(cmap, ax=ax)

    # Set the ticks and tick labels
    cbar_ticks = np.arange(min_count, max_count + 1, 2)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_ticks)
    
    # Set the label of the colorbar
    cbar.set_label('Number of Mice')

    # Adjust the height of the colorbar to match the plot's y-axis
    fig.canvas.draw()
    cbar_pos = cbar.ax.get_position()
    plot_pos = ax.get_position()
    cbar_height = plot_pos.height
    cbar_pos.y0 = plot_pos.y0
    cbar_pos.y1 = plot_pos.y0 + cbar_height
    cbar.ax.set_position(cbar_pos)
    
    # Create the animation
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=framerate)

    # Show the plot
    plt.show()

    # Return the animation object
    return anim

    
def add_nan(plot, xdim):

    fill_to = int(xdim*np.ceil(plot.shape[0]/xdim))
    add_num = int(fill_to - plot.shape[0])
    filler = np.empty((1, add_num)) * np.nan
    plot = np.append(plot,filler)
    plot = plot.reshape(int(plot.shape[0]/xdim),xdim)

    return plot

def get_times_with_interval(times,interval):
    """
    Take a list of times in isoformat and return a subset of those times starting with the first
    and skipping by the value of interval in hours. Useful for labeling rows in heatmaps.
    """
    
    if type(times[0]) is str:
        first_stamp = datetime.fromisoformat(times[1]) #ignore the first time stamp since it might end in seconds that are not 00
    else:
        first_stamp = times[1]
        
    current_stamp = first_stamp
    time_stamps_by_interval = [str(first_stamp)]

    for i in range(int(len(times)//(interval*60))) :
        current_stamp = current_stamp + timedelta(hours=interval) 
        time_stamps_by_interval.append(str(current_stamp))
        
    return time_stamps_by_interval


def heatmaps(df, mouse):
    """
    take a vocalization/rfid dataframe for a given deployment and audiomoth 
    make three heatmaps: squeaks, USVs, and occupance
    if mouse, also include a plot for how many mice were in box (df must be merged vocs/rfid in this case)

    """
    
    if mouse:
        assert 'occupants_count' in df.columns, "To plot occupants you must use a merged dataframe"
    
    #for each time, check whether it was during daylight
    df['daylight'] = [check_sunup(i) for i in df['minute']]

    # subset by hour for plotting
    times = list(df['minute'])
    interval = 1 #interval in hours (amount of time you want in each bin)
    times_by_hour = get_times_with_interval(times,interval)
    df_by_hour = df.loc[df['minute'].isin(times_by_hour)].reset_index()
    
    #get info about which times are during daylight and which not
    night_starts = df_by_hour.index[df_by_hour['daylight'].diff() == -1]
    day_starts = df_by_hour.index[df_by_hour['daylight'].diff() == 1]
    durations = []
    for night_start, day_start in zip(night_starts, day_starts):
        duration =day_start - night_start
        durations.append((night_start, duration))

    #get a list of plots by date
    cry_plots = [df['cry_count']]
    USV_plots = [df['USV_count']]
    if mouse:
        mouseplots = [df['occupants_count']]

    #reshape by adding nan
    cry_plots = [add_nan(i, int(interval*60)) for i in cry_plots]
    USV_plots = [add_nan(i, int(interval*60)) for i in USV_plots]
    if mouse:
        mouseplots = [add_nan(i, int(interval*60)) for i in mouseplots]

    from scipy.ndimage import gaussian_filter
    from matplotlib.collections import LineCollection
    import matplotlib.patches as patches

    #number of periods of night to draw
    num_rects = len(durations)

    #get the data
    marginal_energy_cry = np.nanmean(cry_plots[0], axis=1)
    marginal_energy_USV = np.nanmean(USV_plots[0], axis=1)
    if mouse:
        marginal_mice = np.nanmean(mouseplots[0], axis=1)

    #minmax scale it
    marginal_energy_minmaxscale_cry = [(i - np.nanmin(marginal_energy_cry))/(np.nanmax(marginal_energy_cry) - np.nanmin(marginal_energy_cry)) for i in marginal_energy_cry]
    marginal_energy_minmaxscale_USV = [(i - np.nanmin(marginal_energy_USV))/(np.nanmax(marginal_energy_USV) - np.nanmin(marginal_energy_USV)) for i in marginal_energy_USV]
    if mouse:
        marginal_mice_minmaxscale = [(i - np.nanmin(marginal_mice))/(np.nanmax(marginal_mice) - np.nanmin(marginal_mice)) for i in marginal_mice]

    #gaussian on non-scaled data
    cry_sound_filtered = gaussian_filter(marginal_energy_cry, sigma=1.75)
    USV_sound_filtered = gaussian_filter(marginal_energy_USV, sigma=1.75)
    if mouse:
        mice_filtered = gaussian_filter(marginal_mice, sigma=1.75)

    #gaussion of scaled data
    sound_filtered_minmaxscale_cry = gaussian_filter(marginal_energy_minmaxscale_cry, sigma=1.75)
    sound_filtered_minmaxscale_USV = gaussian_filter(marginal_energy_minmaxscale_USV, sigma=1.75)
    if mouse:
        mice_filtered_minmaxscale = gaussian_filter(marginal_mice_minmaxscale, sigma=1.75)

    #-------
    #make heatmaps and save to save_dir

    # plot https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
    plt.style.use('default')
    
    if not mouse:
        fig, axes =plt.subplots(nrows =1, 
                                ncols= 4, 
                                figsize=[30,10], 
                                sharey=True, 
                                gridspec_kw={'width_ratios': [0.025, 1,0.025, 1],'wspace':0.01, 'hspace':0.01})
    else:
        fig, axes =plt.subplots(nrows =1, 
                                ncols= 6, 
                                figsize=[40,10], 
                                sharey=True, 
                                gridspec_kw={'width_ratios': [0.025, 1,0.025, 1,0.025, 1],'wspace':0.01, 'hspace':0.01})
    

    axes[1].set_title("cries")
    sns.heatmap(cry_plots[0], 
                ax=axes[1], 
                mask = np.isnan(cry_plots[0]), 
                cmap='viridis', 
                square = False, 
                cbar=True, 
                yticklabels=times_by_hour, 
                xticklabels=True)

    axes[3].set_title("USVs")
    sns.heatmap(USV_plots[0], 
                ax=axes[3], 
                mask = np.isnan(USV_plots[0]), 
                cmap='viridis', 
                square = False, 
                cbar=True, 
                yticklabels=times_by_hour, 
                xticklabels=True)

    if mouse:
        axes[5].set_title("number of mice in the box")
        sns.heatmap(mouseplots[0], 
                    ax=axes[5], 
                    mask = np.isnan(mouseplots[0]), 
                    cmap='plasma', 
                    square = False, 
                    cbar=True, 
                    yticklabels=times_by_hour, 
                    xticklabels=True)
    
    #draw day and night
    for rect_id in range(num_rects):
        chunk = durations[rect_id]
        x_start = chunk[0]
        x_end = chunk[0]+chunk[1]
        y_pos = axes[2].get_ylim()[1]
        duration = chunk[1]
        axes[0].set_yticklabels(times_by_hour)
        axes[0].set_xlim([0,0.1])
        rect = patches.Rectangle((y_pos, x_start), 0.1, duration, linewidth=10, color = 'black')
        axes[0].add_patch(rect)
        axes[0].axvline(x=x_start-0.5, color = 'gray', linestyle = '--', linewidth = 1)
        axes[0].axvline(x=x_end+0.5, color = 'gray', linestyle = '--', linewidth = 1)
        axes[0].get_xaxis().set_visible(False)
        #axes[0].get_yaxis().set_visible(False)
        #axes[0].axis('off')

    for rect_id in range(num_rects):
        chunk = durations[rect_id]
        x_start = chunk[0]
        x_end = chunk[0]+chunk[1]
        y_pos = axes[2].get_ylim()[1]
        duration = chunk[1]
        axes[2].set_yticklabels(times_by_hour)
        axes[2].set_xlim([0,0.1])
        rect = patches.Rectangle((y_pos, x_start), 0.1, duration, linewidth=10, color = 'black')
        axes[2].add_patch(rect)
        axes[2].axvline(x=x_start-0.5, color = 'gray', linestyle = '--', linewidth = 1)
        axes[2].axvline(x=x_end+0.5, color = 'gray', linestyle = '--', linewidth = 1)
        axes[2].get_xaxis().set_visible(False)
        axes[2].get_yaxis().set_visible(False)
        #axes[2].axis('off')

    if mouse:
        for rect_id in range(num_rects):
            chunk = durations[rect_id]
            x_start = chunk[0]
            x_end = chunk[0]+chunk[1]
            y_pos = axes[2].get_ylim()[1]
            duration = chunk[1]
            axes[4].set_yticklabels(times_by_hour)
            axes[4].set_xlim([0,0.1])
            rect = patches.Rectangle((y_pos, x_start), 0.1, duration, linewidth=10, color = 'black')
            axes[4].add_patch(rect)
            axes[4].axvline(x=x_start-0.5, color = 'gray', linestyle = '--', linewidth = 1)
            axes[4].axvline(x=x_end+0.5, color = 'gray', linestyle = '--', linewidth = 1)
            axes[4].get_xaxis().set_visible(False)
            axes[4].get_yaxis().set_visible(False)
            #axes[2].axis('off')

def make_summary_plots_vocs(df, save_dir, fontsize = 12):
    """
    take a vocalization/rfid dataframe for a given deployment and audiomoth and save summary heatmaps and plot
    of power spectral density a nd occupancy 

    """

    #for each time, check whether it was during daylight
    df['daylight'] = [check_sunup(i) for i in df['minute']]

    # subset by hour for plotting
    times = list(df['minute'])
    interval = 1 #interval in hours (amount of time you want in each bin)
    times_by_hour = [str(i).split(' ')[-1] for i in get_times_with_interval(times,interval)]
    times_with_date = get_times_with_interval(times,interval)
    df_by_hour = df.loc[df['minute'].isin(times_with_date)].reset_index()
    #print(df_by_hour)
    
    #get info about which times are during daylight and which not
    night_starts = df_by_hour.index[df_by_hour['daylight'].diff() == -1]
    day_starts = df_by_hour.index[df_by_hour['daylight'].diff() == 1]
    durations = []
    for night_start, day_start in zip(night_starts, day_starts):
        duration =day_start - night_start
        durations.append((night_start, duration))

    #get a list of plots by date
    cry_plots = [df['cry_count']]
    USV_plots = [df['USV_count']]
    mouseplots = [df['occupants_count']]
    tempplots = [df['temp']]

    #reshape by adding nan
    cry_plots = [add_nan(i, int(interval*60)) for i in cry_plots]
    USV_plots = [add_nan(i, int(interval*60)) for i in USV_plots]
    mouseplots = [add_nan(i, int(interval*60)) for i in mouseplots]
    tempplots = [add_nan(i, int(interval*60)) for i in tempplots]

    from scipy.ndimage import gaussian_filter
    from matplotlib.collections import LineCollection
    import matplotlib.patches as patches

    #number of periods of night to draw
    num_rects = len(durations)

    #get the data
    marginal_energy_cry = np.nanmean(cry_plots[0], axis=1)
    marginal_energy_USV = np.nanmean(USV_plots[0], axis=1)
    marginal_mice = np.nanmean(mouseplots[0], axis=1)
    marginal_temp = np.nanmean(tempplots[0], axis=1)

    #minmax scale it
    marginal_energy_minmaxscale_cry = [(i - np.nanmin(marginal_energy_cry))/(np.nanmax(marginal_energy_cry) - np.nanmin(marginal_energy_cry)) for i in marginal_energy_cry]
    marginal_energy_minmaxscale_USV = [(i - np.nanmin(marginal_energy_USV))/(np.nanmax(marginal_energy_USV) - np.nanmin(marginal_energy_USV)) for i in marginal_energy_USV]
    marginal_mice_minmaxscale = [(i - np.nanmin(marginal_mice))/(np.nanmax(marginal_mice) - np.nanmin(marginal_mice)) for i in marginal_mice]

    #gaussian on non-scaled data
    cry_sound_filtered = gaussian_filter(marginal_energy_cry, sigma=1.75)
    USV_sound_filtered = gaussian_filter(marginal_energy_USV, sigma=1.75)
    mice_filtered = gaussian_filter(marginal_mice, sigma=1.75)

    #gaussion of scaled data
    sound_filtered_minmaxscale_cry = gaussian_filter(marginal_energy_minmaxscale_cry, sigma=1.75)
    sound_filtered_minmaxscale_USV = gaussian_filter(marginal_energy_minmaxscale_USV, sigma=1.75)
    mice_filtered_minmaxscale = gaussian_filter(marginal_mice_minmaxscale, sigma=1.75)

    #-------
    #make heatmaps and save to save_dir
    save_name = ('_').join(['heat_maps',df['deployment'][0], df['moth'][0], str(df['box'][0])])+'.jpeg'

    # plot https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
    plt.style.use('default')
    fig, axes =plt.subplots(nrows =1, 
                            ncols= 6, 
                            figsize=[16,4], 
                            sharey=True, 
                            gridspec_kw={'width_ratios': [0.025, 1,0.025, 1, 0.025, 1],'wspace':0.01, 'hspace':0.01})

    axes[1].set_title("number cries", fontsize = fontsize)
    
    
    sns.heatmap(cry_plots[0], 
                ax=axes[1], 
                mask = np.isnan(cry_plots[0]), 
                cmap='viridis', 
                square = False, 
                cbar=True, 
                yticklabels=times_by_hour, 
                xticklabels=True)

    axes[3].set_title("number USVs", fontsize = fontsize)
    sns.heatmap(USV_plots[0], 
                ax=axes[3], 
                mask = np.isnan(USV_plots[0]), 
                cmap='viridis', 
                square = False, 
                cbar=True, 
                yticklabels=times_by_hour, 
                xticklabels=True)


    axes[5].set_title("number of mice in the box", fontsize = fontsize)
    sns.heatmap(mouseplots[0], 
                ax=axes[5], 
                mask = np.isnan(mouseplots[0]), 
                cmap='plasma', 
                square = False, 
                cbar=True, 
                yticklabels=times_by_hour, 
                xticklabels=True)
    
#    axes[5].set_title("temperature (˚C)", fontsize = fontsize)
#    sns.heatmap(tempplots[0], 
#                ax=axes[5], 
#                mask = np.isnan(tempplots[0]), 
#                cmap='plasma', 
#                square = False, 
#                cbar=True, 
#                yticklabels=times_by_hour, 
#                xticklabels=True)
    
    #draw day and night
    axes[0].set_ylabel("hour of day", fontsize = fontsize)
    for rect_id in range(num_rects):
        chunk = durations[rect_id]
        x_start = chunk[0]
        x_end = chunk[0]+chunk[1]
        y_pos = axes[2].get_ylim()[1]
        duration = chunk[1]
        axes[0].set_yticklabels(times_by_hour, fontsize=4)
        axes[0].set_xlim([0,0.1])
        rect = patches.Rectangle((y_pos, x_start), 0.1, duration, linewidth=10, color = 'black')
        axes[0].add_patch(rect)
        axes[0].axvline(x=x_start-0.5, color = 'gray', linestyle = '--', linewidth = 1)
        axes[0].axvline(x=x_end+0.5, color = 'gray', linestyle = '--', linewidth = 1)
        axes[0].get_xaxis().set_visible(False)
        #axes[0].get_yaxis().set_visible(False)
        #axes[0].axis('off')

    for rect_id in range(num_rects):
        chunk = durations[rect_id]
        x_start = chunk[0]
        x_end = chunk[0]+chunk[1]
        y_pos = axes[2].get_ylim()[1]
        duration = chunk[1]
        axes[2].set_yticklabels(times_by_hour, fontsize=4)
        axes[2].set_xlim([0,0.1])
        rect = patches.Rectangle((y_pos, x_start), 0.1, duration, linewidth=10, color = 'black')
        axes[2].add_patch(rect)
        axes[2].axvline(x=x_start-0.5, color = 'gray', linestyle = '--', linewidth = 1)
        axes[2].axvline(x=x_end+0.5, color = 'gray', linestyle = '--', linewidth = 1)
        axes[2].get_xaxis().set_visible(False)
        axes[2].get_yaxis().set_visible(False)
        #axes[2].axis('off')

    for rect_id in range(num_rects):
        chunk = durations[rect_id]
        x_start = chunk[0]
        x_end = chunk[0]+chunk[1]
        y_pos = axes[2].get_ylim()[1]
        duration = chunk[1]
        axes[4].set_yticklabels(times_by_hour, fontsize=4)
        axes[4].set_xlim([0,0.1])
        rect = patches.Rectangle((y_pos, x_start), 0.1, duration, linewidth=10, color = 'black')
        axes[4].add_patch(rect)
        axes[4].axvline(x=x_start-0.5, color = 'gray', linestyle = '--', linewidth = 1)
        axes[4].axvline(x=x_end+0.5, color = 'gray', linestyle = '--', linewidth = 1)
        axes[4].get_xaxis().set_visible(False)
        axes[4].get_yaxis().set_visible(False)
        #axes[2].axis('off')
        

    axes[1].set_xlabel("minute of hour", fontsize = fontsize)
    axes[0].tick_params(left = False)
    axes[1].tick_params(left = False)
    axes[2].tick_params(left = False)
    axes[3].tick_params(left = False)
    axes[4].tick_params(left = False)
    axes[5].tick_params(left = False)
    
    axes[1].set_xticklabels(np.arange(1,61,1), fontsize=4)
    axes[3].set_xticklabels(np.arange(1,61,1), fontsize=4)
    axes[5].set_xticklabels(np.arange(1,61,1), fontsize=4)

    plt.savefig(os.path.join(save_dir, save_name), dpi=600)
    print("saved heatmaps to", save_dir)


#    #make summary plots and save to save_dir
#
#    save_name = ('_').join(['bars_dots_lines',df['deployment'][0], df['moth'][0], str(df['box'][0])])+'.jpeg'
#
#    #Row 1----------
#
#    fig, axes = plt.subplots(nrows=3,
#                             ncols=3, 
#                             constrained_layout = True, 
#                             figsize = [18,15])
#
#    axes[0,0].set_xlabel("number of cries")
#    if len(df['cry_count']) > 0:
#        sns.histplot(df['cry_count'], 
#                     color='deeppink', 
#                     linewidth=0.001, 
#                     ax=axes[0,0])
#
#    axes[0,1].set_xlabel("number of USVs")
#    if len(df['USV_count']) > 0:
#        sns.histplot(df['USV_count'], 
#                     color='thistle', 
#                     linewidth=0.001, binwidth=(0.1)*(axes[0,1].get_xlim()[1] - axes[0,1].get_xlim()[0]),
#                     ax=axes[0,1])
#
#    axes[0,2].set_xlabel("number of mice in the box")
#    if len(df['occupants_count']) > 0:
#        sns.histplot(df['occupants_count'], 
#                     color='black', 
#                     linewidth=0.001, 
#                     ax=axes[0,2])
#
#    #Row 2--------------
#
#    axes[1,0].set_ylabel("number of USVs")
#    axes[1,0].set_xlabel("lnumber of cries")
#    sns.scatterplot(df['cry_count'], 
#                    df['USV_count'], 
#                    s=10, 
#                    alpha=.25, 
#                    color = 'k',                
#                    ax = axes[1,0])
#
#    axes[1,1].set_ylabel("number of cries")
#    axes[1,1].set_xlabel("number of mice in the box")
#    sns.stripplot(df['occupants_count'],
#                    df['cry_count'], 
#                    jitter = True,
#                    s=2.5, 
#                    alpha=.25, 
#                    color = 'k',                
#                    ax = axes[1,1])
#
#    axes[1,2].set_ylabel("number of USVs")
#    axes[1,2].set_xlabel("number of mice in the box")
#    sns.stripplot(x=df['occupants_count'],
#                  y=df['USV_count'], 
#                  s=2.5, 
#                  jitter=True,
#                  alpha=.25, 
#                  color = 'k',                
#                  ax = axes[1,2])
#    sns.despine()
#
#    #Row 3--------------
#
#    axes[2,0].plot(marginal_energy_cry, drawstyle='steps-pre', linewidth=1, color='grey')
#    axes[2,0].plot(marginal_energy_USV, drawstyle='steps-pre', linewidth=1, color='grey')
#
#    axes[2,0].plot(USV_sound_filtered, linewidth=5, color='thistle')
#    axes[2,0].plot(cry_sound_filtered, linewidth=5, color='deeppink')
#    axes[2,0].set_xticks(range(len(times_by_hour)))
#    axes[2,0].set_xticklabels(times_by_hour, rotation=90)
#    axes[2,0].set_title("avg cries and avg USVs per hour vs time")
#
#    #draw day.night
#    for rect_id in range(num_rects):
#        chunk = durations[rect_id]
#        x_start = chunk[0]
#        x_end = chunk[0]+chunk[1]
#        y_pos = axes[2,0].get_ylim()[1]
#        duration = chunk[1]
#        rect = patches.Rectangle((x_start, y_pos), duration, (0.1)*(axes[2,0].get_ylim()[1] - axes[2,0].get_ylim()[0]), linewidth=10, color = 'black')
#        axes[2,0].add_patch(rect)
#        axes[2,0].axvline(x=x_start-0.5, color = 'gray', linestyle = '--', linewidth = 1)
#        axes[2,0].axvline(x=x_end+0.5, color = 'gray', linestyle = '--', linewidth = 1)
#
#    axes[2,1].plot(marginal_mice, drawstyle='steps-pre', linewidth=0.5, color='grey')
#    axes[2,1].plot(mice_filtered, linewidth=5, color='black')
#    axes[2,1].set_xticks(range(len(times_by_hour)))
#    axes[2,1].set_xticklabels(times_by_hour, rotation=90)
#    axes[2,1].set_title("avg number of mice in the box per hour vs time")
#
#    #draw day.night
#    for rect_id in range(num_rects):
#        chunk = durations[rect_id]
#        x_start = chunk[0]
#        x_end = chunk[0]+chunk[1]
#        y_pos = axes[2,1].get_ylim()[1]
#        duration = chunk[1]
#        rect = patches.Rectangle((x_start, y_pos), duration, (0.1)*(axes[2,1].get_ylim()[1] - axes[2,1].get_ylim()[0]), linewidth=10, color = 'black')
#        axes[2,1].add_patch(rect)
#        axes[2,1].axvline(x=x_start-0.5, color = 'gray', linestyle = '--', linewidth = 1)
#        axes[2,1].axvline(x=x_end+0.5, color = 'gray', linestyle = '--', linewidth = 1)
#
#    axes[2,2].plot(sound_filtered_minmaxscale_cry, linewidth=5, color='deeppink', alpha = 0.3)
#    axes[2,2].plot(sound_filtered_minmaxscale_USV, linewidth=5, color='thistle', alpha = 0.3)
#    axes[2,2].plot(mice_filtered_minmaxscale, linewidth=5, color='black')
#    axes[2,2].set_xticks(range(len(times_by_hour)))
#    axes[2,2].set_xticklabels(times_by_hour, rotation=90)
#    axes[2,2].set_title("combined and scaled to same y-axis")
#
#    #draw day.night
#    for rect_id in range(num_rects):
#        chunk = durations[rect_id]
#        x_start = chunk[0]
#        x_end = chunk[0]+chunk[1]
#        y_pos = axes[2,2].get_ylim()[1]
#        duration = chunk[1]
#        rect = patches.Rectangle((x_start, y_pos), duration, (0.1)*(axes[2,2].get_ylim()[1] - axes[2,2].get_ylim()[0]), linewidth=10, color = 'black')
#        axes[2,2].add_patch(rect)
#        axes[2,2].axvline(x=x_start-0.5, color = 'gray', linestyle = '--', linewidth = 1)
#        axes[2,2].axvline(x=x_end+0.5, color = 'gray', linestyle = '--', linewidth = 1)
#
#    sns.despine()
#
#
#    plt.savefig(os.path.join(save_dir, save_name), dpi=600)
#    print("saved summary plots to", save_dir)
#    print("done.")
#    
#    #plot voc counts
#    
#    save_name = ('_').join(['voc_counts',df['deployment'][0], df['moth'][0], str(df['box'][0])])+'.jpeg'
#    
#    #make a dateframe of vocalization counts
#    df = df.sort_values(by='minute')
#
#    #plot
#    from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
#    fig, ax = plt.subplots(nrows=1, 
#                           ncols=1, 
#                           figsize=[15,10])
#
#    #plot cry count
#    sns.barplot(data = df, 
#                x = 'minute', 
#                y='cry_count', 
#                ax=ax, 
#                color='deeppink', 
#                alpha = .8, 
#                label='cry')
#
#    #plot USV count
#    sns.barplot(data = df, 
#                x = 'minute', 
#                y='USV_count', 
#                ax=ax, 
#                color='grey', 
#                alpha = .8, 
#                label='USV')
#
#    #prettify
#    sns.despine()
#    every_nth = 60
#    labels=[]
#    for n, label in enumerate(ax.xaxis.get_ticklabels()):
#        labels.append(label)
#        if n % every_nth != 0:
#            label.set_visible(False)
#
#    ax.set_xticklabels(labels, rotation = 90, fontsize=9)
#    ax.set_ylabel(['vocalization count'])
#    ax.legend(frameon = False)
#    
#    plt.savefig(os.path.join(save_dir, save_name), dpi=600)
#    print("saved summary plots to", save_dir)
#    print("done.")
      
def make_summary_plots_psd(df, save_dir):
    """
    take a df of power spectral density and occupancy for a given deployment and audiomoth and save summary heatmaps and plot
    of power spectral density and occupancy 
    
    """

    #for each time, check whether it was during daylight
    df['daylight'] = [check_sunup(i) for i in df['minute']]
    
    # subset by hour for plotting
    times = df['minute']
    interval = 1 #interval in hours (amount of time you want in each bin)
    times_by_hour = get_times_with_interval(times,interval)
    df_by_hour = df.loc[df['minute'].isin(times_by_hour)].reset_index()

    #get info about which times are during daylight and which not
    night_starts = df_by_hour.index[df_by_hour['daylight'].diff() == -1]
    day_starts = df_by_hour.index[df_by_hour['daylight'].diff() == 1]
    durations = []
    for night_start, day_start in zip(night_starts, day_starts):
        duration =day_start - night_start
        durations.append((night_start, duration))

    #get a list of plots by date
    high_soundplots = [np.log10(df['[55000, 75000]'])]
    low_soundplots = [np.log10(df['[15000, 35000]'])]
    mouseplots = [df['occupants_count']]

    #reshape by adding nan
    high_soundplots = [add_nan(i, int(interval*60)) for i in high_soundplots]
    low_soundplots = [add_nan(i, int(interval*60)) for i in low_soundplots]
    mouseplots = [add_nan(i, int(interval*60)) for i in mouseplots]
    
    from scipy.ndimage import gaussian_filter
    from matplotlib.collections import LineCollection
    import matplotlib.patches as patches

    #number of periods of night to draw
    num_rects = len(durations)

    #get the data
    marginal_energy_low = np.nanmean(low_soundplots[0], axis=1)
    marginal_energy_high = np.nanmean(high_soundplots[0], axis=1)
    marginal_mice = np.nanmean(mouseplots[0], axis=1)

    #minmax scale it
    marginal_energy_minmaxscale_low = [(i - np.nanmin(marginal_energy_low))/(np.nanmax(marginal_energy_low) - np.nanmin(marginal_energy_low)) for i in marginal_energy_low]
    marginal_energy_minmaxscale_high = [(i - np.nanmin(marginal_energy_high))/(np.nanmax(marginal_energy_high) - np.nanmin(marginal_energy_high)) for i in marginal_energy_high]
    marginal_mice_minmaxscale = [(i - np.nanmin(marginal_mice))/(np.nanmax(marginal_mice) - np.nanmin(marginal_mice)) for i in marginal_mice]

    #gaussian on non-scaled data
    low_sound_filtered = gaussian_filter(marginal_energy_low, sigma=1.75)
    high_sound_filtered = gaussian_filter(marginal_energy_high, sigma=1.75)
    mice_filtered = gaussian_filter(marginal_mice, sigma=1.75)

    #gaussion of scaled data
    sound_filtered_minmaxscale_low = gaussian_filter(marginal_energy_minmaxscale_low, sigma=1.75)
    sound_filtered_minmaxscale_high = gaussian_filter(marginal_energy_minmaxscale_high, sigma=1.75)
    mice_filtered_minmaxscale = gaussian_filter(marginal_mice_minmaxscale, sigma=1.75)

    #-------
    #make heatmaps and save to save_dir
    save_name = ('_').join(['heat_maps',os.path.split(df['source_file'][0])[0].split('/')[-1]])+'.jpeg'

    # plot https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
    plt.style.use('default')
    fig, axes =plt.subplots(nrows =1, 
                            ncols= 6, 
                            figsize=[40,10], 
                            sharey=True, 
                            gridspec_kw={'width_ratios': [0.025, 1,0.025, 1, 0.025, 1],'wspace':0.01, 'hspace':0.01})

    axes[1].set_title("15-35 kHz bandpower (log)")
    sns.heatmap(low_soundplots[0], 
                ax=axes[1], 
                mask = np.isnan(low_soundplots[0]), 
                cmap='viridis', 
                square = False, 
                cbar=True, 
                yticklabels=times_by_hour, 
                xticklabels=True)

    axes[3].set_title("55-75 kHz bandpower (log)")
    sns.heatmap(high_soundplots[0], 
                ax=axes[3], 
                mask = np.isnan(high_soundplots[0]), 
                cmap='viridis', 
                square = False, 
                cbar=True, 
                yticklabels=times_by_hour, 
                xticklabels=True)


    axes[5].set_title("number of mice in the box")
    sns.heatmap(mouseplots[0], 
                ax=axes[5], 
                mask = np.isnan(mouseplots[0]), 
                cmap='plasma', 
                square = False, 
                cbar=True, 
                yticklabels=times_by_hour, 
                xticklabels=True)

    for rect_id in range(num_rects):
        chunk = durations[rect_id]
        x_start = chunk[0]
        x_end = chunk[0]+chunk[1]
        y_pos = axes[2].get_ylim()[1]
        duration = chunk[1]
        axes[0].set_yticklabels(times_by_hour)
        axes[0].set_xlim([0,0.1])
        rect = patches.Rectangle((y_pos, x_start), 0.1, duration, linewidth=10, color = 'black')
        axes[0].add_patch(rect)
        axes[0].axvline(x=x_start-0.5, color = 'gray', linestyle = '--', linewidth = 1)
        axes[0].axvline(x=x_end+0.5, color = 'gray', linestyle = '--', linewidth = 1)
        axes[0].get_xaxis().set_visible(False)
        #axes[0].get_yaxis().set_visible(False)
        #axes[0].axis('off')

    for rect_id in range(num_rects):
        chunk = durations[rect_id]
        x_start = chunk[0]
        x_end = chunk[0]+chunk[1]
        y_pos = axes[2].get_ylim()[1]
        duration = chunk[1]
        axes[2].set_yticklabels(times_by_hour)
        axes[2].set_xlim([0,0.1])
        rect = patches.Rectangle((y_pos, x_start), 0.1, duration, linewidth=10, color = 'black')
        axes[2].add_patch(rect)
        axes[2].axvline(x=x_start-0.5, color = 'gray', linestyle = '--', linewidth = 1)
        axes[2].axvline(x=x_end+0.5, color = 'gray', linestyle = '--', linewidth = 1)
        axes[2].get_xaxis().set_visible(False)
        axes[2].get_yaxis().set_visible(False)
        #axes[2].axis('off')

    for rect_id in range(num_rects):
        chunk = durations[rect_id]
        x_start = chunk[0]
        x_end = chunk[0]+chunk[1]
        y_pos = axes[2].get_ylim()[1]
        duration = chunk[1]
        axes[4].set_yticklabels(times_by_hour)
        axes[4].set_xlim([0,0.1])
        rect = patches.Rectangle((y_pos, x_start), 0.1, duration, linewidth=10, color = 'black')
        axes[4].add_patch(rect)
        axes[4].axvline(x=x_start-0.5, color = 'gray', linestyle = '--', linewidth = 1)
        axes[4].axvline(x=x_end+0.5, color = 'gray', linestyle = '--', linewidth = 1)
        axes[4].get_xaxis().set_visible(False)
        axes[4].get_yaxis().set_visible(False)
        #axes[2].axis('off')

    plt.savefig(os.path.join(save_dir, save_name), dpi=600)
    print("saved heatmaps to", save_dir)
    
    
    #make summary plots and save to save_dir
    
    save_name = ('_').join(['bars_dots_lines',os.path.split(df['source_file'][0])[0].split('/')[-1]])+'.jpeg'

    #Row 1----------

    fig, axes = plt.subplots(nrows=3,
                             ncols=3, 
                             constrained_layout = True, 
                             figsize = [18,15])

    axes[0,0].set_xlabel("log(15000 kHz - 35000 kHz) bandpower")
    sns.histplot(np.log10(df['[15000, 35000]']), 
                 color='deeppink', 
                 linewidth=0.001, binwidth=(0.1)*(axes[0,0].get_xlim()[1] - axes[0,0].get_xlim()[0]), 
                 ax=axes[0,0])

    axes[0,1].set_xlabel("log(55000 kHz - 75000 kHz) bandpower")
    sns.histplot(np.log10(df['[55000, 75000]']), 
                 color='thistle', 
                 linewidth=0.001, binwidth=(0.1)*(axes[0,1].get_xlim()[1] - axes[0,1].get_xlim()[0]),
                 ax=axes[0,1])

    axes[0,2].set_xlabel("number of mice in the box")
    sns.histplot(df['occupants_count'], 
                 color='black', 
                 linewidth=0.001, 
                 ax=axes[0,2])

    #Row 2--------------

    axes[1,0].set_ylabel("log(55000 kHz - 75000 kHz) bandpower")
    axes[1,0].set_xlabel("log(15000 kHz - 35000 kHz) bandpower")
    sns.scatterplot(np.log(df['[15000, 35000]']), 
                    np.log(df['[55000, 75000]']), 
                    s=10, 
                    alpha=.25, 
                    color = 'k',                
                    ax = axes[1,0])

    axes[1,1].set_ylabel("log(15000 kHz - 35000 kHz) bandpower")
    axes[1,1].set_xlabel("number of mice in the box")
    sns.stripplot(df['occupants_count'],
                    np.log(df['[15000, 35000]']), 
                    jitter = True,
                    s=2.5, 
                    alpha=.25, 
                    color = 'k',                
                    ax = axes[1,1])

    axes[1,2].set_ylabel("log(55000 kHz - 75000 kHz) bandpower")
    axes[1,2].set_xlabel("number of mice in the box")
    sns.stripplot(x=df['occupants_count'],
                  y=np.log(df['[55000, 75000]']), 
                  s=2.5, 
                  jitter=True,
                  alpha=.25, 
                  color = 'k',                
                  ax = axes[1,2])
    sns.despine()

    #Row 3--------------

    axes[2,0].plot(marginal_energy_low, drawstyle='steps-pre', linewidth=1, color='grey')
    axes[2,0].plot(marginal_energy_high, drawstyle='steps-pre', linewidth=1, color='grey')

    axes[2,0].plot(high_sound_filtered, linewidth=5, color='thisle')
    axes[2,0].plot(low_sound_filtered, linewidth=5, color='deeppink')
    axes[2,0].set_xticks(range(len(times_by_hour)))
    axes[2,0].set_xticklabels(times_by_hour, rotation=90)
    axes[2,0].set_title("avg log(15-35kHz) orange - avg log(55-75kHz) red per hour vs time")

    #draw day.night
    for rect_id in range(num_rects):
        chunk = durations[rect_id]
        x_start = chunk[0]
        x_end = chunk[0]+chunk[1]
        y_pos = axes[2,0].get_ylim()[1]
        duration = chunk[1]
        rect = patches.Rectangle((x_start, y_pos), duration, (0.1)*(axes[2,0].get_ylim()[1] - axes[2,0].get_ylim()[0]), linewidth=10, color = 'black')
        axes[2,0].add_patch(rect)
        axes[2,0].axvline(x=x_start-0.5, color = 'gray', linestyle = '--', linewidth = 1)
        axes[2,0].axvline(x=x_end+0.5, color = 'gray', linestyle = '--', linewidth = 1)

    axes[2,1].plot(marginal_mice, drawstyle='steps-pre', linewidth=0.5, color='grey')
    axes[2,1].plot(mice_filtered, linewidth=5, color='black')
    axes[2,1].set_xticks(range(len(times_by_hour)))
    axes[2,1].set_xticklabels(times_by_hour, rotation=90)
    axes[2,1].set_title("avg number of mice in the box per hour vs time")

    #draw day.night
    for rect_id in range(num_rects):
        chunk = durations[rect_id]
        x_start = chunk[0]
        x_end = chunk[0]+chunk[1]
        y_pos = axes[2,1].get_ylim()[1]
        duration = chunk[1]
        rect = patches.Rectangle((x_start, y_pos), duration, (0.1)*(axes[2,1].get_ylim()[1] - axes[2,1].get_ylim()[0]), linewidth=10, color = 'black')
        axes[2,1].add_patch(rect)
        axes[2,1].axvline(x=x_start-0.5, color = 'gray', linestyle = '--', linewidth = 1)
        axes[2,1].axvline(x=x_end+0.5, color = 'gray', linestyle = '--', linewidth = 1)

    axes[2,2].plot(sound_filtered_minmaxscale_low, linewidth=5, color='orange', alpha = 0.3)
    axes[2,2].plot(sound_filtered_minmaxscale_high, linewidth=5, color='red', alpha = 0.3)
    axes[2,2].plot(mice_filtered_minmaxscale, linewidth=5, color='black')
    axes[2,2].set_xticks(range(len(times_by_hour)))
    axes[2,2].set_xticklabels(times_by_hour, rotation=90)
    axes[2,2].set_title("combined and scaled to same y-axis")

    #draw day.night
    for rect_id in range(num_rects):
        chunk = durations[rect_id]
        x_start = chunk[0]
        x_end = chunk[0]+chunk[1]
        y_pos = axes[2,2].get_ylim()[1]
        duration = chunk[1]
        rect = patches.Rectangle((x_start, y_pos), duration, (0.1)*(axes[2,2].get_ylim()[1] - axes[2,2].get_ylim()[0]), linewidth=10, color = 'black')
        axes[2,2].add_patch(rect)
        axes[2,2].axvline(x=x_start-0.5, color = 'gray', linestyle = '--', linewidth = 1)
        axes[2,2].axvline(x=x_end+0.5, color = 'gray', linestyle = '--', linewidth = 1)

    sns.despine()


    plt.savefig(os.path.join(save_dir, save_name), dpi=600)
    print("saved summary plots to", save_dir)
    print("done.")
def spectrograms_from_umap(df, wavs_dir, from_umap, cry_spec_params, USV_spec_params, cry_seed, USV_seed, save):
    """
    Show a umap embedding with spectrogram image examples taken directly from the umap or regenerated with different parameters
    
    """
    
    voc_colors = load_json('/Volumes/LaCie_barn/mouse_barn_audiomoth/parameters/vocalization_colors.json')
    
    #set up the axes
    fig, axes = plt.subplot_mosaic(mosaic="AAAABCDEFG;AAAAHIJKLM;AAAANOPQRS", 
                                   figsize=[8,3], 
                                   tight_layout=True,
                                   dpi=600)

    # #parameters
    num_freq_bins=128
    num_time_bins=128
    dot_size = 3
    dot_alpha = .5
    text_color = 'black'
    text_size = 7
    
    xlim = [-5,15]
    ylim = [-5,15]


    #plot the umap
    axes["A"].scatter(
        df['umap1'],
        df['umap2'],
        c = df['label'].map(voc_colors),
        s = .5,
        alpha = .5, 
        linewidth=0)

    USV_spec_axes = ["B", "C", "D", "H", "I", "J", "N", "O", "P"]
    cry_spec_axes = ["E", "F", "G", "K", "L", "M", "Q", "R", "S"]
    all_spec_axes  = USV_spec_axes + cry_spec_axes

    #remove ticks
    axes["A"].axis('off')
    sns.despine()
    for ax in all_spec_axes:
        axes[ax].set_axis_off()
        axes[ax].set_axis_off()
    fig.subplots_adjust(wspace=0, hspace = 0)

    #sample example vocalizations
    USV_examples = df.loc[df['label'] == 'USV'].sample(n=9, random_state=USV_seed)
    cry_examples = df.loc[df['label'] == 'cry'].sample(n=9, random_state=cry_seed)

    #overlay them on the umap as black dots
    axes["A"].scatter(
        USV_examples['umap1'],
        USV_examples['umap2'],
        c = 'black',
        s = dot_size, 
        linewidth=0,
        alpha = dot_alpha)

    axes["A"].scatter(
        cry_examples['umap1'],
        cry_examples['umap2'],
        c = 'black',
        s = dot_size, 
        linewidth=0,
        alpha = dot_alpha)

    axes["A"].set_xlim(xlim)
    axes["A"].set_ylim(ylim)

    #label them so you know which spec goes with which axis
    USV_examples = USV_examples.sort_values(by='umap2', ascending=False)
    cry_examples = cry_examples.sort_values(by='umap2', ascending=False)
    USV_examples['plot_axis'] = USV_spec_axes
    cry_examples['plot_axis'] = cry_spec_axes

    #annotate the dots with numbers and move some of them so they aren't on top of other numbers
    txt = [1,2,3,4,5,6,7,8,9]
    for txt, umap1, umap2 in zip(txt,USV_examples['umap1'], USV_examples['umap2']):

        if txt == 9:
            axes["A"].annotate(text=txt, xy=(umap1,  umap2), xytext=(-4,0), textcoords = 'offset points', color=text_color, fontsize=text_size)
        elif txt == 7:
            axes["A"].annotate(text=txt, xy=(umap1,  umap2), xytext=(-4,0), textcoords = 'offset points', color=text_color, fontsize=text_size)
        elif txt == 5:
            axes["A"].annotate(text=txt, xy=(umap1,  umap2), xytext=(-4,-4), textcoords = 'offset points', color=text_color, fontsize=text_size)
        elif txt == 3:
            axes["A"].annotate(text=txt, xy=(umap1,  umap2), xytext=(-4,0), textcoords = 'offset points', color=text_color, fontsize=text_size)
        else:
            axes["A"].annotate(text=txt, xy=(umap1,  umap2), xytext=(0.5,0.5), textcoords = 'offset points', color=text_color, fontsize=text_size)

    txt = [10,11,12,13,14,15,16,17,18]
    for txt, umap1, umap2 in zip(txt,cry_examples['umap1'], cry_examples['umap2']):
        if txt == 12:
            axes["A"].annotate(text=txt, xy=(umap1,  umap2), xytext=(-9,0), textcoords = 'offset points', color=text_color, fontsize=text_size)
        elif txt == 18:
            axes["A"].annotate(text=txt, xy=(umap1,  umap2), xytext=(-9,0), textcoords = 'offset points', color=text_color, fontsize=text_size)
        else:
            axes["A"].annotate(text=txt, xy=(umap1,  umap2), xytext=(0.5,0.5), textcoords = 'offset points', color=text_color, fontsize=text_size)

    if from_umap:
        #put the USV spectrograms on their axes
        txt = [1,2,3,4,5,6,7,8,9]
        for ax,label in zip(USV_spec_axes, txt):

            #get the spectrogram
            pixels = [str(i) for i in range(num_freq_bins*num_time_bins)]
            to_plot = USV_examples.loc[USV_examples['plot_axis'] == ax][pixels]
            img = np.array(to_plot).reshape((num_freq_bins, num_time_bins))      

            #plot
            axes[ax].axis('off')
            axes[ax].imshow(img, origin = 'lower', cmap='viridis', extent = (num_freq_bins, 0, num_time_bins, 0 ))
            axes[ax].text(118, 15, label, ha="center", va="center", color="w", fontsize=9, fontname='Arial')

        #put the cry spectrograms on their axes
        txt = [10,11,12,13,14,15,16,17,18]
        for ax, label in zip(cry_spec_axes, txt):

            #get the spectrogram
            to_plot = cry_examples.loc[cry_examples['plot_axis'] == ax][pixels]
            img = np.array(to_plot).reshape((num_freq_bins, num_time_bins))

            #plot
            axes[ax].axis('off')
            axes[ax].imshow(img, origin = 'lower', cmap='viridis', extent = (num_freq_bins, 0, num_time_bins, 0 ))
            axes[ax].text(115, 15, label, ha="center", va="center", color="w", fontsize=9, fontname='Arial')
            
    else:
        #make the spectrograms
        
        #get the paths to the voc clips
        cry_clips = [os.path.join(wavs_dir,moth,deployment,wav) for moth, deployment, wav in zip(cry_examples['moth'], cry_examples['deployment'], cry_examples['clip_names'])]
        USV_clips = [os.path.join(wavs_dir,moth,deployment,wav) for moth, deployment, wav in zip(USV_examples['moth'], USV_examples['deployment'], USV_examples['clip_names'])]

        cry_specs = specs_from_wavs(clips_to_process = cry_clips, spec_params = cry_spec_params, num_to_process='all')
        USV_specs = specs_from_wavs(clips_to_process = USV_clips, spec_params = USV_spec_params, num_to_process='all')
        
        #put the USV spectrograms on their axes
        txt = [1,2,3,4,5,6,7,8,9]
        for ax,label in zip(USV_spec_axes, txt):

            #get the spectrogram
            spec_name = USV_examples['clip_names'].loc[USV_examples['plot_axis'] == ax].iloc[0]
            img = USV_specs[0][USV_specs[1].index(spec_name)]      

            #plot
            axes[ax].axis('off')
            axes[ax].imshow(img, origin = 'lower', cmap='viridis', extent = (num_freq_bins, 0, num_time_bins, 0 ))
            axes[ax].text(118, 15, label, ha="center", va="center", color="w", fontsize=9, fontname='Arial')

        #put the cry spectrograms on their axes
        txt = [10,11,12,13,14,15,16,17,18]
        for ax, label in zip(cry_spec_axes, txt):

            #get the spectrogram
            spec_name = cry_examples['clip_names'].loc[cry_examples['plot_axis'] == ax].iloc[0]
            img = cry_specs[0][cry_specs[1].index(spec_name)]

            #plot
            axes[ax].axis('off')
            axes[ax].imshow(img, origin = 'lower', cmap='viridis', extent = (num_freq_bins, 0, num_time_bins, 0 ))
            axes[ax].text(115, 15, label, ha="center", va="center", color="w", fontsize=9, fontname='Arial')
def customize_axis(ax, ylim=None, xlim=None, xlab=None, ylab=None, rot_x_lab=False, fontsize = 9):
    for label in ax.get_yticklabels(): 
        label.set_size(fontsize)
    for label in ax.get_xticklabels(): 
        label.set_size(fontsize)
        if rot_x_lab:
            label.set_rotation(90)

    #ax.margins(x=0)
    ax.tick_params(width=0.5)
    ax.tick_params(axis='x', length=2)
    
    if xlab is not None:
        ax.set_xlabel(xlab, fontsize = fontsize)
        
    if ylab is not None:
        ax.set_ylabel(ylab, fontsize = fontsize)
    
    if ylim is not None:
        ax.set_ylim(ylim)  # If you want to set a specific y-axis limit
        
    if xlim is not None:
        ax.set_ylim(xlim)  # If you want to set a specific x-axis limit

    sns.despine()
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    
    return ax
def get_colors_from_numbers(cmap, numbers):
	"""
	Give a list of numbers and a colormap name, get a list of colors generated by mapping those numbers onto that colormap
	"""
        
    # Define a colormap
	colormap = plt.get_cmap(cmap)  
	
	# Normalize the integers
	norm = Normalize(vmin=min(list(numbers)), vmax=max(list(numbers)))

	# Get colors for each integer in the list
	colors = [colormap(norm(value)) for value in list(numbers)]
	
	return colors
def stay_ethograms(dataframes):
	"""
	give a list of stay dataframes (one per mouse)
	get an ethogram-style plot of stay times with one row per mouse
	"""

	height = 0.1

	fig, ax = plt.subplots(figsize=(16, 4))


	min_timestamp = mdates.date2num(min(df['entry_time'].min()  if 'entry_time' in df.columns else df['overlap_start_time'].min() for df in dataframes))
	max_timestamp = mdates.date2num(max(df['exit_time'].max() if 'exit_time' in df.columns else df['overlap_start_time'].min() for df in dataframes ))
	

	for i, df in enumerate(dataframes):

		for _, row in df.iterrows():

			if 'entry_time' in df.columns:
				entry_time = mdates.date2num(pd.to_datetime(row['entry_time'])) 
				exit_time = mdates.date2num(pd.to_datetime(row['exit_time'])) 
			elif 'overlap_start_time' in df.columns:
				entry_time = mdates.date2num(pd.to_datetime(row['overlap_start_time'])) 
				exit_time = mdates.date2num(pd.to_datetime(row['overlap_end_time'])) 
				
			# Add rectangle to the plot
			if 'entry_time' in df.columns:
				rect = Rectangle((entry_time, (i/4)+height+0.1), exit_time - entry_time, 0.19, edgecolor='black', facecolor='salmon')
				ax.add_patch(rect)
			elif 'overlap_start_time' in df.columns:
				rect = Rectangle((entry_time, (i/4)+height+0.1), exit_time - entry_time, 0.19, edgecolor='black', facecolor='grey')
				ax.add_patch(rect)

	ax.xaxis_date()
	ax.xaxis.set_major_locator(mdates.HourLocator())
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))

	ax.set_xlabel('Time')
	ax.set_yticklabels([])
	ax.set_yticks([])
	ax.set_xlim([min_timestamp, max_timestamp])
	ax.set_ylim([0.19, height*len(dataframes)+(0.1)*4+0.2])
	sns.despine(left=True)
	
	plt.xticks(rotation=90)
	plt.show()
    

def convert_interbox_values_to_longform(distance_df, value_name='distance'):
    """
    Convert a square matrix DataFrame into a long-form DataFrame with unique pairwise combinations.
    
    Parameters:
    - distance_df: pd.DataFrame, square matrix DataFrame where columns and index are box numbers.
    - distance_col_name: str, name for the 'distance' column in the result.
    
    Returns:
    - pd.DataFrame: long-form DataFrame with columns for box_1, box_2, and distance.
    """
    # Melt the DataFrame to long format
    melted_df = distance_df.reset_index().melt(id_vars='index', var_name='box_2', value_name=value_name)
    melted_df.columns = ['box_1', 'box_2', value_name]

    # Ensure unique pairwise combinations ignoring order
    melted_df['min_box'] = np.minimum(melted_df['box_1'], melted_df['box_2'])
    melted_df['max_box'] = np.maximum(melted_df['box_1'], melted_df['box_2'])

    # Drop duplicates to keep only unique pairs
    unique_pairs_df = melted_df.drop(columns=['box_1', 'box_2']).drop_duplicates().rename(
        columns={'min_box': 'box_1', 'max_box': 'box_2'})

    # Reset the index for clarity
    unique_pairs_df.reset_index(drop=True, inplace=True)

    return unique_pairs_df
    

