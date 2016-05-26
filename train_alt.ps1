$config = "rnn1.ini" 
$netType = "-nN6ReluSqueezeL2Squeeze"  
$count = 1
$z = "-z0.005"
    
for ($j=0; $j -lt 30; $j++) {
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
