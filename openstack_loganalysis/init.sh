file="./output/"
if [ -e $file ]
then
  echo "$file exists"
else
  mkdir -p $file
fi

file="./output/bert"
if [ -e $file ]
then
  echo "$file exists"
else
  mkdir -p $file
fi


