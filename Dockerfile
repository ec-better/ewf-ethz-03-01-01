FROM docker-co.terradue.com/ec-better/ewf-ethz-03-01-01:1.8


ENV PATH=/opt/anaconda/envs/p36-ethz-03-01-01/bin:/opt/anaconda/bin:/opt/anaconda/condabin:/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:$PATH

RUN conda install -y gdal=2.2.2 "poppler<0.62"

RUN rpm --rebuilddb
RUN touch /var/lib/rpm/* && yum install -y wxGTK3-devel proj-devel gdal-devel jasper-devel libtiff-devel unixODBC-devel
RUN touch /var/lib/rpm/* && yum install -y gcc gcc-c++ make automake libtool git
RUN touch /var/lib/rpm/* && yum install -y centos-release-scl
RUN touch /var/lib/rpm/* && yum install -y devtoolset-7
RUN touch /var/lib/rpm/* && yum install -y /usr/include/libpq-fe.h
ENV PATH=/usr/libexec/wxGTK3:/opt/rh/devtoolset-7/root/usr/bin:$PATH
RUN mkdir -p /application/saga

RUN git clone git://git.code.sf.net/p/saga-gis/code /application/saga/saga-gis-code
WORKDIR /application/saga/saga-gis-code/saga-gis
RUN autoreconf -fi
RUN ./configure --with-proj-libraries=/opt/anaconda/pkgs/proj4-5.0.1-h14c3975_0/lib --with-proj-includes=/opt/anaconda/pkgs/proj4-5.0.1-h14c3975_0/include
RUN make
RUN make install
RUN echo SAGA GIS installation successfully concluded
WORKDIR /
ENV PATH=/opt/anaconda/bin:$PATH