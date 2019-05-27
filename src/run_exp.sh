numExp=50

## complete information
complete_information=1
for Lambda in 0.1 0.3 0.5 0.7 0.9 1.1 1.3 1.5 2 4 6 8 10
do
	python main.py $numExp $Lambda 5 '../data/redwine.dat' 0.8 0.5 $complete_information 0
done



## incomplete information
complete_information=0
overEstimate=0
for Lambda in 0.1 0.3 0.5 0.7 0.9 1.1 1.3 1.5 2 4 6 8 10
do
	python main.py $numExp $Lambda 5 '../data/redwine.dat' 0.8 0.5 $complete_information $overEstimate
done

overEstimate=1
for Lambda in 0.1 0.3 0.5 0.7 0.9 1.1 1.3 1.5 2 4 6 8 10
do
	python main.py $numExp $Lambda 5 '../data/redwine.dat' 0.8 0.5 $complete_information $overEstimate
done
