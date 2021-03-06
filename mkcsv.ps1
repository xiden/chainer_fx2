param([System.Int32]$lines)

$src = get-content -path ".\USDJPY-cd1.csv" -encoding "Ascii" -tail $lines
$lastLine = $src[$src.Length - 1]
$fields = $lastLine.Split(',')
$date = $fields[0].Replace(".", "")
$time = $fields[1].Replace(":", "")
$file = join-path -path "Datasets" -childpath ($date + "_" + $time + "_" + $lines + ".csv")
set-content -path $file -encoding "Ascii" $src
python graph.py $file
