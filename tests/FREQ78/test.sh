function valid () {
    if [ $? -eq 0 ]; then
        echo OK
    else
        echo ERROR
        exit 1
    fi
}

test=$($1 -i $2/FREQ78.ms -o $2/residuals.ms -O $2/mod_out.fits -m $2/mod_in_0.fits -p $2/mem/ -X 16 -Y 16 -V 256 -z 0.001 -Z 0.005,0.0 -t 500000000 -g 2 --verbose --print-images)
valid $test

#Comment the following lines to see the results of the test
rm -rf $2/residuals.ms
rm -rf $2/mem/
rm $2/alpha.fits
rm $2/mod_out.fits
