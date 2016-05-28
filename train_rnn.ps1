$config = "rnn1.ini" 
$netType = "-nOpenDiv4N5L1N1B2L1N1"  
$count = 100
$z = "-z0.005"
    
for ($i=0; $i -lt $count; $i++) {
    if($i -eq ($count - 1)) {
        $b = "-b1"
    } else {
        $b = "-b0"
    }
    python .\chainer_fx.py $config $netType -d0_10000_10000 -g0 -mtrainhr -t50 $b $z
}
