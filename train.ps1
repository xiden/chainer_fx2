$config = "clas4.ini" 
$z = "-z0.005"
$dataset = "-d0_2400000_2400000" 

for ($i=0; $i -lt 47; $i++) {
	python .\chainer_fx.py $config $dataset -g0 -mtrainhr -t10 -b0 $z
	python .\chainer_fx.py $config $dataset -g0 -mplotw
}
