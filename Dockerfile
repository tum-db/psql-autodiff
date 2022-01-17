# FROM ubuntu:20.04
# LABEL Clemens Ruck - Maintainer
# #ENV DEBIAN_FRONTEND noninteractive

# EXPOSE 5432/tcp
# EXPOSE 5432/udp

# # Install some tools
# RUN apt-get update && apt-get install -y g++ clang-7 llvm-7 python make python2.7-dev bison flex libreadline-dev zlib1g-dev perl
# COPY . /psql
# RUN rm -rf /psql/install
# RUN cd /psql && mkdir install && ./configure --with-llvm --prefix=/psql/install/ && make -j && make install && cd src/ext/ && make install-postgres-bitcode
# RUN cd /psql/src/ext/ && cp *.so /psql/install/share/ && cp *.so /psql/install/lib/
# RUN useradd -ms /bin/bash pguser && mkdir /pgdata && chown pguser /pgdata
# #RUN mkdir /results

# # postgres as admin user
# USER pguser
# RUN /psql/install/bin/initdb -D /pgdata

# # allow remote access to internal files
# #VOLUME  ["/results"]

# # Set the default command to run when starting the container
# CMD /psql/install/bin/postmaster -D /pgdata



FROM ubuntu:20.04
LABEL Clemens Ruck - Maintainer
#ENV DEBIAN_FRONTEND noninteractive

# Install some tools
RUN apt-get update && apt-get install -y g++ clang-7 llvm-7 python make python2.7-dev bison flex libreadline-dev zlib1g-dev perl
COPY . /psql
RUN rm -rf /psql/install
RUN cd /psql && mkdir install && ./configure --with-llvm --prefix=/psql/install/ CFLAGS='-fopenmp' CXXFLAGS='-lpthread -lpq -lm -fopenmp' && make -j && make install && cd src/ext/ && make install-postgres-bitcode
RUN cd /psql/src/ext/ && cp *.so /psql/install/share/ && cp *.so /psql/install/lib/
RUN useradd -ms /bin/bash postgres && mkdir /pgdata && chown postgres /pgdata && chmod 750 /pgdata
#RUN mkdir /results

USER postgres
RUN /psql/install/bin/pg_ctl initdb -D /pgdata
RUN /psql/install/bin/pg_ctl start -D /pgdata && /psql/install/bin/psql --command "CREATE USER docker WITH SUPERUSER PASSWORD 'docker';" && /psql/install/bin/createdb -O docker docker
# allow remote connections
RUN echo "host all  all    0.0.0.0/0  md5" >> /pgdata/pg_hba.conf && echo "listen_addresses='*'" >> /pgdata/postgresql.conf
EXPOSE 5432
VOLUME  ["/pgdata", "/var/log/postgresql", "/psql/install/lib/postgresql"]
# Set the default command to run when starting the container
CMD ["/psql/install/bin/postgres", "-D", "/pgdata", "-c", "config_file=/pgdata/postgresql.conf"]