"""
Make a pie chart - see
http://matplotlib.sf.net/matplotlib.pylab.html#-pie for the docstring.

This example shows a basic pie chart with labels optional features,
like autolabeling the percentage, offsetting a slice with "explode",
adding a shadow, and changing the starting angle.

"""
from pylab import *

# make a square figure and axes
figure(1, figsize=(6,6))
ax = axes([0.1, 0.1, 0.8, 0.8])

# The slices will be ordered and plotted counter-clockwise.
labels = 'Domestic', 'UK', 'USA', 'Germany', 'France', 'Italy', 'Spain', 'Other Europe', 'Asia', 'America', 'Africa/Australia'
fracs = [19, 15, 9, 8, 6, 5, 4, 19, 7, 5, 3]
#explode=(0, 0.05, 0, 0)
# explode=explode, 
pie(fracs, labels=labels,
                autopct='%1.1f%%', shadow=True, startangle=90)
                # The default startangle is 0, which would start
                # the Frogs slice on the x-axis.  With startangle=90,
                # everything is rotated counter-clockwise by 90 degrees,
                # so the plotting starts on the positive y-axis.

title('Arrivals in hotels')

show()

figure(1, figsize=(6,6))
ax = axes([0.1, 0.1, 0.8, 0.8])

# The slices will be ordered and plotted counter-clockwise.
labels = 'Domestic', 'UK', 'USA', 'Germany', 'France', 'Italy', 'Spain', 'Other Europe',                   'Asia',                 'America',          'Africa/Australia'
fracs = [1315,       213,   79,    26,       71,        16,     11,       27+21+13+13+10+6+4+4+3+3+2+2+9,   9+9+7+6+6+4+4+4+3+3+3+2+4, 23+9+6+6+6+2+2+3 , 2+2+2+1]
print fracs
fracs = [100*float(x)/sum(fracs) for x in fracs]
print fracs
#explode=(0, 0.05, 0, 0)
# explode=explode, 
print len(labels)
print len(fracs)
pie(fracs,
                autopct='%1.1f%%', shadow=True, startangle=90)
                # The default startangle is 0, which would start
                # the Frogs slice on the x-axis.  With startangle=90,
                # everything is rotated counter-clockwise by 90 degrees,
                # so the plotting starts on the positive y-axis.

#plt.axis('equal')
plt.legend(labels, loc=(-0.05, 0.05), shadow=True)

title('Twitter user: country of r')

show()