$config = "squeezeAve2.ini" 
$netType = "-nN15ReluSqueeze"  
$z = "-z0.0"

python .\chainer_fx.py $config $netType -d0_0_1440 -g1 -mtesthr $z
