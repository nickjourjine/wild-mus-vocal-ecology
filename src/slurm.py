### This code adapted from Brock Wooldrige and Andi Kautt

import os
import glob
import subprocess

### create dictionary to catch slurm job IDs for all jobs submitted by function below
slurm_ids = {}

### function to create slurm files and run them
def make_job(cmd_string,job_name,mem, time, err_dir, out_dir, scripts_dir, gpu = False, echo=False,run=True,write=True,         
             N='1',n='1',c='1', array='1-1%1',p='hoekstra,shared'):
    
    if not gpu:
        SLURM = ('''#!/bin/bash\n'''
               '''#SBATCH -N {nodes}\n'''
            '''#SBATCH -n {tasks}\n'''
            '''#SBATCH -c {cpus}\n'''
            '''#SBATCH -t {time}\n'''
            '''#SBATCH --mem={mem}\n'''
            '''#SBATCH -p {partition}\n'''
            '''#SBATCH --array {array}\n'''
            '''#SBATCH --job-name {job_name}\n'''
            '''#SBATCH -e {err_dir}/%x_%A_%a.err\n'''
            '''#SBATCH -o {out_dir}/%x_%A_%a.out\n'''
            '''{cmd_string}\n''').format(       
             job_name=job_name,
                cmd_string=cmd_string,
                partition=p,
                time=time,
                mem=mem,
                tasks=n,
                cpus=c,
                nodes=N,
                array=array,
                err_dir=err_dir, 
                out_dir=out_dir
            )

    elif gpu:
        SLURM = ('''#!/bin/bash\n'''
            
            '''#SBATCH -n {tasks}\n'''
            '''#SBATCH -t {time}\n'''
            '''#SBATCH --mem={mem}\n'''
            '''#SBATCH -p {partition}\n'''
            '''#SBATCH --array {array}\n'''
            '''#SBATCH --job-name {job_name}\n'''
            '''#SBATCH -e {err_dir}/%x_%A_%a.err\n'''
            '''#SBATCH -o {out_dir}/%x_%A_%a.out\n'''
            '''#SBATCH --gres=gpu:1\n'''
            '''{cmd_string}\n''').format(
             job_name=job_name,
                cmd_string=cmd_string,
                partition=p,
                time=time,
                mem=mem,
                tasks=n,
                cpus=c,
                nodes=N,
                array=array,
                err_dir=err_dir,
                out_dir=out_dir
            )

    # Show SLURM command?
    if echo == True:
        print(SLURM)

    # Write to file and/or submit to SLURM?
    if write == True:
        filename = os.path.join(scripts_dir,job_name+'.slurm')
        with open(filename, 'w') as outfile:
            outfile.write(SLURM)
            print('"%s" slurm script written to %s\n' %(job_name,scripts_dir))
        # Run
        if run == True:
            sbatch_response = subprocess.getoutput('sbatch {}'.format(filename))
            print(sbatch_response)
            job_id = sbatch_response.split(' ')[-1].strip()
            slurm_ids[job_name] = job_id
            print('"%s" job submitted ' %(job_name))
            
    return

### function to cancel a job
def cancel_job(job_ID):
    response = subprocess.getoutput('scancel {}'.format(job_ID))
    print(response)
### function to retrieve running job info (sacct)
def get_job_status(job_ID):
    
    """
    Give: a job ID number
    Get: see the output of sacct (ie, status check on your job)
    """

    sacct_out = [ subprocess.getoutput('sacct -j {}'.format(job_ID)) ]

    output = []
    for line in sacct_out:
        jobs = line.split("\n")
        for entry in jobs:
            fields = entry.split()
            output.append(fields[0:9])
    print(*output,sep="\n")

def show_log(job_ID, err_dir, out_dir):

    """
    Give a job ID and a path to the logs directory
    Get the contents of the log for that job
    """

    err_files = glob.glob(os.path.join(err_dir, '*.err'))
    out_files = glob.glob(os.path.join(out_dir, '*.out'))

    print('err file:')
    err_file = [i for i in err_files if str(job_ID) in i]
    assert len(err_file) == 1
    with open(err_file[0], 'r') as file:
        for line in file:
            print(line, end='')
    print('\n')

    print('out file:')
    out_file = [i for i in out_files if str(job_ID) in i]
    assert len(out_file) == 1
    with open(out_file[0], 'r') as file:
        for line in file:
            print(line, end='')
