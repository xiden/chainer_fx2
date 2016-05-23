$config = "squeezeAve2.ini" 
$netType = "-nN15ReluSqueeze"  
$count = 10
$z = "-z0.005"
    
for ($j=0; $j -lt 10; $j++) {
    for ($i=0; $i -lt $count; $i++) {
        if($i -eq ($count - 1)) {
            $b = "-b1"
        } else {
            $b = "-b0"
        }
        python .\chainer_fx.py $config $netType -d0_10000_50000 -g0 -mtrainhr -t10 $b $z
    }
    for ($i=0; $i -lt $count; $i++) {
        if($i -eq ($count - 1)) {
            $b = "-b1"
        } else {
            $b = "-b0"
        }
        python .\chainer_fx.py $config $netType -d-1_10000_50000 -g0 -mtrainhr -t10 $b $z
    }
}
