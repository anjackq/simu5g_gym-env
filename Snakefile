import glob

rule all:
    input: ["src/simu5g_gym_exp_dbg", "src/simu5g_gym_exp"]

rule protobuf:
    input: "src/protobuf/{file}.proto"
    output:
        cpp=multiext("src/protobuf/{file}", ".pb.cc", ".pb.h"),
    shell: "env protoc --proto_path src/protobuf --cpp_out src/protobuf {input}"

rule configure:
    input:
        code_files=[glob.glob(f"src/**/*.{ext}", recursive=True) for ext in ["msg", "cc", "h"]],
        cpp=multiext("src/protobuf/simu5g_gym.pb.", "cc", "h"),
    output: "src/Makefile"
    params:
        include_flags = ' '.join(['-I.', '-I../lib/simu5g/src' ,'-I../lib/veins/src', '-I../lib/veins-vlc/src', '-I../lib/zmq/src', '-I/usr/local/include']),
        link_flags = ' '.join(['-L../lib/simu5g/src/', '-lsimu5g\\$\(D\)', '-L../lib/veins/src/', '-lveins\\$\(D\)', '-L../lib/veins-vlc/src/', '-lveins-vlc\\$\(D\)', '-L/usr/local/lib', '-lzmq', '-lprotobuf', '-lpthread']),
        flags = ' '.join(['-f', '--deep', '-o', 'simu5g_gym_exp', '-O', 'out']),
    shell: "env -C src opp_makemake {params.flags} {params.include_flags} {params.link_flags}"

rule configure_veins:
    input: [glob.glob(f"lib/veins/src/**/*.{ext}", recursive=True) for ext in ["msg", "cc", "h"]]
    output: "lib/veins/src/Makefile"
    shell: "env -C lib/veins ./configure"

rule configure_veins_vlc:
    input: [glob.glob(f"lib/veins-vlc/src/**/*.{ext}", recursive=True) for ext in ["msg", "cc", "h"]]
    output: "lib/veins-vlc/src/Makefile"
    shell: "env -C lib/veins-vlc ./configure --with-veins=../veins"
    
rule configure_simu5g:
    input: [glob.glob(f"lib/simu5g/src/**/*.{ext}", recursive=True) for ext in ["msg", "cc", "h"]]
    output: "lib/simu5g/src/Makefile"
    shell: "env -C lib/simu5g make makefiles"

rule build_veins:
    input: "lib/veins/src/Makefile",
    output: "lib/veins/src/libveins{dbg,(_dbg)?}.so"
    params: mode=lambda wildcards, output: "debug" if "_dbg" == wildcards.dbg else "release"
    threads: workflow.cores
    shell: "make -j{threads} -C lib/veins/src MODE={params.mode}"

rule build_veins_vlc:
    input: rules.build_veins.output, "lib/veins-vlc/src/Makefile"
    output: "lib/veins-vlc/src/libveins-vlc{dbg,(_dbg)?}.so"
    params: mode=lambda wildcards, output: "debug" if "_dbg" == wildcards.dbg else "release"
    threads: workflow.cores
    shell: "make -j{threads} -C lib/veins-vlc/src MODE={params.mode}"
    
rule build_simu5g:
    input: "lib/simu5g/src/Makefile",
    output: "lib/simu5g/src/libsimu5g{dbg,(_dbg)?}.so"
    params: mode=lambda wildcards, output: "debug" if "_dbg" == wildcards.dbg else "release"
    threads: workflow.cores
    shell: "make -j{threads} -C lib/simu5g/src MODE={params.mode}"

rule build:
    input: rules.build_veins.output, rules.build_veins_vlc.output, rules.build_simu5g.output, "src/Makefile"
    output: "src/simu5g_gym_exp{dbg,(_dbg)?}"
    params: mode=lambda wildcards, output: "debug" if "_dbg" == wildcards.dbg else "release"
    threads: workflow.cores
    shell: "make -j{threads} -C src MODE={params.mode}"
