import os, random, shutil, sys

Nruns = int(sys.argv[1])

time_delay_ca3 = 95 # ms
time_delay = 400 # ms, for NEURON to reach a steady state

n_place_fields = 41
for nrun in range(1,Nruns+1):
    print 'RUN '+str(nrun)+' ...\n'

    random.seed(nrun)
    p='run_' + str(nrun) + '/'
    source='runs_produced_by_python_prelearning/' + p
    dest='Inputs_linear_prelearning/' + p

    if not os.path.exists(dest):
        os.makedirs(dest)
    
    
    listdirs = []
    for pl in xrange(1,n_place_fields+1):    
        listdirs.append('place_field_'+str(pl))

    counter=0
    for mydir in listdirs:
        L=len(os.listdir(source+mydir))
        for i in range(L):
            shutil.copy2(source+mydir+'/s'+str(i)+'.txt', dest+'g'+str(counter)+'_EC.txt') # complete target filename given
            shutil.copy2(source+mydir+'/s'+str(i)+'.txt', dest+'g'+str(counter)+'_CA3.txt') # complete target filename given

            # Input form EC
            with open(dest+'g'+str(counter)+'_EC.txt', 'r') as f:
                lines = f.readlines()
                lines = [str(int(x.strip())+time_delay) for x in lines]
            with open(dest+'g'+str(counter)+'_EC.txt', 'w') as f:
                for line in lines:
                    nline=str(int(line))
                    f.write(nline + '\n')
                    
            # Input form CA3 + delay 95ms
            rn=time_delay_ca3
            with open(dest+'g'+str(counter)+'_CA3.txt', 'r') as f:
                lines = f.readlines()
                lines = [str(int(x.strip())+time_delay+rn) for x in lines]
            with open(dest+'g'+str(counter)+'_CA3.txt', 'w') as f:
                for line in lines:                 
                    nline=str(int(line))
                    f.write(nline + '\n')

            counter+=1

    Ncells = counter
    # Check all files!
    for cnt in xrange(Ncells):
        with open(dest+'g'+str(cnt)+'_CA3.txt', 'r') as f:
            lines = f.readlines()
            lines = [int(x.strip()) for x in lines]
            for j in xrange(len(lines)-1):
                if lines[j] == lines[j+1]:
                    print cnt,lines[j],lines[j+1]
