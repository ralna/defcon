# Creates a Firedrake defcon container

FROM firedrakeproject/firedrake

# This DockerFile is looked after by
MAINTAINER Jack Betteridge <j.betteridge@imperial.ac.uk>

USER firedrake

WORKDIR /home/firedrake

# Clone defcon
RUN bash -c "cd /home/firedrake/firedrake/src && git clone https://bitbucket.org/pefarrell/defcon.git"

# Install defcon
RUN bash -c "source /home/firedrake/firedrake/bin/activate && pip install -e /home/firedrake/firedrake/src/defcon"

ENV OMP_NUM_THREADS=1
