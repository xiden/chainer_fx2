$config = "clas2.ini" 
$netType = "-nOpenHighLowDiv2N10N1"  
$z = "-z0.005"
$datasets = "-d-1_40000_50000", "-d-2_40000_50000", "-d0_40000_50000" 
	
for ($j=0; $j -lt 3; $j++) {
	$datasets | % {
		python .\chainer_fx.py $config $netType $_ -g0 -mtrainhr -t10 -b0 $z
		python .\chainer_fx.py $config $netType $dataset -mplotw
		python .\chainer_fx.py $config "-d0_1440_1440" -mtesthr "-z0"
	}
}
