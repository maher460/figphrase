
from PIL import ImageDraw
from PIL import Image

import csv

with open("explore2b3_method_matrix.csv", "r") as f:
    lines = csv.reader(f, delimiter=",")

    for l in lines:
    	img = Image.open('/afs/cs/projects/kovashka/maher/vol3/matrix_results_dist_custoff/' + l[0] + ".png")

		ImageDraw.Draw(img).text(
		    (5, 5),  # Coordinates
		    l[1] + ", " + str(l[3]),  # Text
		    (0, 0, 0)  # Color
		)


		img.save('/afs/cs/projects/kovashka/maher/vol3/matrix_results_dist_custoff_labels/' + l[0] + ".png")