import cadquery as cq
w0=cq.Workplane('ZX',origin=(0,100,0))
w1=cq.Workplane('XY',origin=(0,0,-33))
r=w0.sketch().segment((-59,-3),(-59,3)).segment((-