$config = "clas23.ini" 
$z = "-z0.005"
$dataset1 = "-d0_300000_300000" 
$dataset2 = "-d0_600000_600000" 
$dataset3 = "-d0_1200000_1200000" 
$dataset4 = "-d0_2400000_2400000" 
$datasetTest = "-d0_1440_1440" 

for ($i=0; $i -lt 1000; $i++) {
	python .\fx.py $config $dataset4 -g0 -mtrainhr -t10 -b0 $z
	python .\fx.py $config $dataset4 -g0 -mplotw
	python .\fx.py $config $datasetTest -g0 -mtesthr -z0
}
