import cadquery as cq
w0=cq.Workplane('ZX',origin=(0,100,0))
w1=cq.Workplane('XY',origin=(0,0,-33))
r=w0.sketch().segment((-59,-3),(-59,3)).segment((-58,3)).arc((-56,13),(-52,23)).arc((-56,13),(-59,3)).segment((-59,5)).arc((-60,0),(-59,-3)).assemble().reset().face(w0.sketch().arc((-58,-5),(0,-58),(58,-5)).segment((52,-5)).arc((50,-13),(46,-21)).segment((46,-23)).segment((45,-23)).arc((0,-50),(-45,-23)).segment((-46,-23)).segment((-46,-21)).arc((-50,-13),(-52,-5)).close().assemble()).finalize().extrude(-199).union(w1.sketch().circle(100).circle(99,mode='s').finalize().extrude(10))