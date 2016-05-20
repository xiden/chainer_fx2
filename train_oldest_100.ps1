$netTypes = "-nN10Relu", "-nN15Relu"  
$count = 10

for ($j=0; $j -lt 2; $j++) {
    $netType = $netTypes[$j] 
    for ($i=0; $i -lt $count; $i++) {
        if($i -eq ($count - 1)) {
            $b = "-b1"
        } else {
            $b = "-b0"
        }
        python .\chainer_fx.py test9.ini $netType -d-1_100000_500000 -g0 -mtrainhr -t10 $b
    }
}
