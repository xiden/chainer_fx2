$config = "clas6.ini" 
$z = "-z0.005"
$dataset = "-d0_2400000_2400000" 
#$dataset = "-d0_1200000_1200000" 
#$dataset = "-d0_20000_40000" 

for ($i=0; $i -lt 1000; $i++) {
	python .\fx.py $config $dataset -g0 -mtrainhr -t10 -b0 $z
	python .\fx.py $config $dataset -g0 -mplotw
}
