# Franka Panda (system 4.2.x) control tools built against libfranka 0.9.2.
#
# Build (from the repo root):
#   docker build -t franka-real -f examples/franka_real/Dockerfile examples/franka_real
# Run:
#   docker run --rm -it --network host --cap-add SYS_NICE --ulimit rtprio=99 --ulimit memlock=-1 \
#     franka-real go_home 172.16.0.2

FROM ubuntu:22.04

ARG LIBFRANKA_VERSION=0.9.2

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates libpoco-dev libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --recursive --branch ${LIBFRANKA_VERSION} --depth 1 https://github.com/frankaemika/libfranka.git /opt/libfranka \
    && cmake -S /opt/libfranka -B /opt/libfranka/build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=ON \
    && cmake --build /opt/libfranka/build -j"$(nproc)" \
    && cmake --install /opt/libfranka/build \
    && ldconfig

# libfranka examples (communication_test, echo_robot_state, ...) are handy for diagnostics.
ENV PATH="/opt/libfranka/build/examples:${PATH}"

COPY CMakeLists.txt go_home.cpp /opt/franka_real/
RUN cmake -S /opt/franka_real -B /opt/franka_real/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/franka_real/build \
    && cmake --install /opt/franka_real/build

CMD ["bash"]
