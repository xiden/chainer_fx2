$config = "squeezeAve.ini" 
$netType = "-nN15ReluSqueeze"  
$z = "-z0.005"
$b = "-b1"
python .\chainer_fx.py $config $netType -d0_10000_50000 -g1 -mtrainhr -t10 $b $z
