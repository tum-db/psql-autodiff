FROM ubuntu:20.04
LABEL Clemens Ruck - Maintainer
#ENV DEBIAN_FRONTEND noninteractive

EXPOSE 5432/tcp
EXPOSE 5432/udp

# Install some tools
RUN apt-get update && apt-get install -y g++ clang-7 llvm-7 python make python2.7-dev bison flex libreadline-dev zlib1g-dev perl
COPY . /psql
RUN rm -rf /psql/install
RUN cd /psql && mkdir install && ./configure --with-llvm --prefix=/psql/install/ && make -j && make install && cd src/ext/ && make install-postgres-bitcode
RUN cd /psql/src/ext/ && cp *.so /psql/install/share/ && cp *.so /psql/install/lib/
RUN useradd -ms /bin/bash pguser && mkdir /pgdata && chown pguser /pgdata
#RUN mkdir /results

# postgres as admin user
USER pguser
RUN /psql/install/bin/initdb -D /pgdata

# allow remote access to internal files
#VOLUME  ["/results"]

# Set the default command to run when starting the container
CMD /psql/install/bin/postmaster -D /pgdata