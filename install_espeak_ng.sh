rm -rf espeak
mkdir -p espeak
cd espeak
wget https://github.com/espeak-ng/espeak-ng/archive/refs/tags/1.51.zip
unzip -qq 1.51.zip
cd espeak-ng-1.51
./autogen.sh
./configure --prefix=`pwd`/../usr
make
make install
