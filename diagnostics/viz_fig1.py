import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from scipy import interpolate as interp
import lib.utils as utils

def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def save_fig1(model, data_samples, savedir, ntimes=101, memory=0.01, device='cpu'):
    model.eval()

    #  Sample from prior
    # z_samples = torch.randn(20, 200).to(device)
    z_samples = torch.randn(20, 50).to(device)

    with torch.no_grad():
    #     # We expect the model is a chain of CNF layers wrapped in a SequentialFlow container.
        logp_samples = torch.sum(standard_normal_logprob(z_samples), 1, keepdim=True)
        t = 0
        for cnf in model.chain:
            end_time = (cnf.sqrt_end_time * cnf.sqrt_end_time)
            integration_times = torch.linspace(0, end_time, ntimes)

            z_traj, _ = cnf(z_samples, logp_samples, integration_times=integration_times, reverse=True)
            z_traj = z_traj.cpu().numpy()


        makedirs(savedir)
        for sample in range(z_traj.shape[1]):
            plt.clf()
            plt.imshow(z_traj[:,sample,:],cmap='plasma')
            plt.xaxis
            plt.savefig(os.path.join(savedir, "fig1_"+str(sample)+".jpg"))

def save_fig1_rev(model, data_samples, savedir, ntimes=101, memory=0.01, device='cpu'):
    model.eval()
    data_samples=torch.tensor(data_samples).float().cuda()

    #  Sample from prior
    # z_samples = torch.randn(20, 200).to(device)
    z_samples = torch.randn(20, 50).to(device)

    with torch.no_grad():
    #     # We expect the model is a chain of CNF layers wrapped in a SequentialFlow container.
        logp_samples = torch.sum(standard_normal_logprob(z_samples), 1, keepdim=True)
        t = 0
        for cnf in model.chain:
            end_time = (cnf.sqrt_end_time * cnf.sqrt_end_time)
            integration_times = torch.linspace(0, end_time, ntimes)

            z_traj, _ = cnf(data_samples[0:1], logp_samples[0:1], integration_times=integration_times, reverse=False)
            z_traj = z_traj.cpu().numpy()


        print('zt',z_traj.shape)
        makedirs(savedir)
        plt.clf()
        plt.imshow(data_samples[0:1],cmap='plasma')
        plt.savefig(os.path.join(savedir, "fig1_data.jpg"))       
        for sample in range(z_traj.shape[1]):
            plt.clf()
            plt.imshow(z_traj[:,sample,:],cmap='plasma')
            plt.savefig(os.path.join(savedir, "fig1_forward"+str(sample)+".jpg"))       

def save_fig1_1d_ptd(model, data_samples, savedir, ntimes=101, memory=0.01, device='cpu',itr=''):
    model.eval()
    

    # data_samples=torch.tensor(data_samples).float().cuda()

    #  Sample from prior
    z_samples = torch.randn(30, 1).to(device)

    # linspace for plotting
    npts=500

    z_samples = np.linspace(-4,4,100)
    z_samples = torch.from_numpy(z_samples[:,np.newaxis]).type(torch.float32).to(device)
    znp = np.linspace(-4,4,npts)
    z = torch.from_numpy(znp[:,np.newaxis]).type(torch.float32).to(device)
                    
    with torch.no_grad():
    #     # We expect the model is a chain of CNF layers wrapped in a SequentialFlow container.
        logp_samples = torch.sum(standard_normal_logprob(z_samples), -1, keepdim=True)
        logp_z = torch.sum(standard_normal_logprob(z), -1, keepdim=True)
        t = 0
        for cnf in model.chain:
            end_time = (cnf.sqrt_end_time * cnf.sqrt_end_time)
            integration_times = torch.linspace(0, end_time, ntimes)

            def log_prob(t):
                z_traj,dlogp_traj = cnf(z,torch.zeros_like(logp_z),integration_times = torch.tensor([t,end_time]),reverse = False)
                z_traj = z_traj
                logp_z_traj = standard_normal_logprob(z_traj)
                dlogp_traj = dlogp_traj.cpu().numpy()
                return logp_z_traj.cpu().numpy() - dlogp_traj

            
            logp = []
            for t in integration_times:
                logp.append(log_prob(t))

            # The differential equation evaluated at some t and x.
            def _differential(t, x):
                t = torch.tensor(t).to(device)
                x = torch.tensor(x).to(device)
                return cnf.odefunc.odefunc.diffeq(t, x)

            ts = np.linspace(0,end_time,100)
            xs = np.linspace(-4,4,100)
            dxs = torch.zeros(ts.shape[0],xs.shape[0])
            
            for ti , t in enumerate(ts):
                for xi,x in enumerate(xs):
                    dxs[ti,xi]= -_differential(t,[x])

            dxs = torch.tensor(dxs)
            dts = torch.ones_like(dxs)


            # z_traj, logp_traj= cnf(z, logp_z, integration_times=integration_times, reverse=False)
            z_traj, logp_traj= cnf(z_samples, logp_samples, integration_times=integration_times, reverse=True)
            # z_traj, logp_traj= cnf(z, logp_z, integration_times=integration_times, reverse=True)
            # z_traj = z_traj.cpu().numpy()
            # logp_traj= logp_traj.cpu().numpy()

    makedirs(savedir)
    plt.clf()

    probs = np.exp(np.array(logp)[:,:,0])
    maxs = np.amax(probs,axis=1,keepdims=True)
    probs = probs / maxs


    plt.rcParams.update({'font.size': 13})
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True,
                                                            gridspec_kw={'height_ratios': [1,5, 1]},
                                                            figsize=(4, 7))
    fig.set_tight_layout({'pad': 0.1, 'h_pad': -1.0})

    axes[0].scatter(znp,np.exp(np.array(logp)[0,:,0]),s=0.5,marker=None,linestyle='-',c=np.exp(np.array(logp)[0,:,0]),cmap='viridis')
    axes[0].set_xlim(-4,4)
    axes[0].set_ylabel(r"$p(z(t_1))$",labelpad=20)
    axes[0].set_yticks([])
    axes[0].get_xaxis().set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].set_ylim(bottom=0.0)

    axes[1].imshow(probs,cmap='viridis',extent=[-4,4,0,0.5],aspect=10)
    axes[1].streamplot(xs,ts,dxs,dts,color='white',linewidth=0.9,density=(0.7,0.5),arrowsize=0.8)
    axes[1].set_xlim(-4,4)
    axes[1].set_yticks([0,0.5])
    axes[1].set_yticklabels([r"$0$",r"$1$"])
    axes[1].set_ylabel(r"$t$")
    axes[1].get_xaxis().set_visible(False)


    # axes[2].plot(znp,-np.exp(np.array(logp)[-1,:,0]))
    axes[2].scatter(znp,np.exp(np.array(logp)[-1,:,0]),s=0.5,marker=None,linestyle='-',c=np.exp(np.array(logp)[-1,:,0]),cmap='viridis')
    axes[2].set_xlim(-4,4)
    axes[2].set_ylabel(r"$p(z(t_0))$",labelpad=20)
    axes[2].set_xlabel(r"$z$")
    axes[2].set_yticks([])
    axes[2].set_xticks([])
    # axes[2].get_xaxis().set_visible(False)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[2].set_ylim(bottom=0.)

    # fig.subplots_adjust(hspace=0.)
    pos0 = axes[0].get_position(original=False)
    pos1 = axes[1].get_position(original=False)
    pos2 = axes[2].get_position(original=False)

    print(pos0.y0)
    print(pos1.y0+pos1.height)
    axes[0].set_position([pos1.x0,pos0.y0+0.4,pos1.x1,pos0.height])
    axes[2].set_position([pos1.x0,pos2.y0,pos1.x1,pos2.height])

    plt.savefig(os.path.join(savedir, "fig1_1d_together"+str(itr)+".png"),pad_inches=0,bbox_inches='tight',dpi=350) 

def save_fig1_1d_ptd_timescrub(model, data_samples, savedir, ntimes=101, memory=0.01, device='cpu',itr=''):
    model.eval()
    

    # data_samples=torch.tensor(data_samples).float().cuda()

    #  Sample from prior
    z_samples = torch.randn(30, 1).to(device)

    # linspace for plotting
    npts=500

    z_samples = np.linspace(-4,4,100)
    z_samples = torch.from_numpy(z_samples[:,np.newaxis]).type(torch.float32).to(device)
    znp = np.linspace(-4,4,npts)
    z = torch.from_numpy(znp[:,np.newaxis]).type(torch.float32).to(device)
                    
    with torch.no_grad():
    #     # We expect the model is a chain of CNF layers wrapped in a SequentialFlow container.
        logp_samples = torch.sum(standard_normal_logprob(z_samples), -1, keepdim=True)
        logp_z = torch.sum(standard_normal_logprob(z), -1, keepdim=True)
        t = 0
        for cnf in model.chain:
            end_time = (cnf.sqrt_end_time * cnf.sqrt_end_time)
            integration_times = torch.linspace(0, end_time, ntimes)

            def log_prob(t):
                z_traj,dlogp_traj = cnf(z,torch.zeros_like(logp_z),integration_times = torch.tensor([t,end_time]),reverse = False)
                z_traj = z_traj
                logp_z_traj = standard_normal_logprob(z_traj)
                dlogp_traj = dlogp_traj.cpu().numpy()
                return logp_z_traj.cpu().numpy() - dlogp_traj

            
            logp = []
            for t in integration_times:
                logp.append(log_prob(t))

            # The differential equation evaluated at some t and x.
            def _differential(t, x):
                t = torch.tensor(t).to(device)
                x = torch.tensor(x).to(device)
                return cnf.odefunc.odefunc.diffeq(t, x)

            ts = np.linspace(0,end_time,100)
            xs = np.linspace(-4,4,100)
            dxs = torch.zeros(ts.shape[0],xs.shape[0])
            
            for ti , t in enumerate(ts):
                for xi,x in enumerate(xs):
                    dxs[ti,xi]= -_differential(t,[x])

            dxs = torch.tensor(dxs)
            dts = torch.ones_like(dxs)


            # z_traj, logp_traj= cnf(z, logp_z, integration_times=integration_times, reverse=False)
            z_traj, logp_traj= cnf(z_samples, logp_samples, integration_times=integration_times, reverse=True)
            # z_traj, logp_traj= cnf(z, logp_z, integration_times=integration_times, reverse=True)
            # z_traj = z_traj.cpu().numpy()
            # logp_traj= logp_traj.cpu().numpy()

    makedirs(savedir)
    for timerow in range(integration_times.shape[0]):
        plt.clf()

        probs = np.exp(np.array(logp)[:,:,0])
        maxs = np.amax(probs,axis=1,keepdims=True)
        probs = probs / maxs


        plt.rcParams.update({'font.size': 13})
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True,
                                                                gridspec_kw={'height_ratios': [1,5]},
                                                                figsize=(8,13))
        fig.set_tight_layout({'pad': 0.1, 'h_pad': -1.0})

        axes[0].scatter(znp,np.exp(np.array(logp)[::-1][timerow,:,0]),s=0.5,marker=None,linestyle='-',c=np.exp(np.array(logp)[::-1][timerow,:,0]),cmap='viridis')
        axes[0].set_xlim(-4,4)
        axes[0].set_ylim(0,.42)
        axes[0].set_ylabel(r"$p(z(t))$",labelpad=20)

        axes[0].set_yticks([])
        axes[0].get_xaxis().set_visible(False)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[0].set_ylim(bottom=0.0)

        axes[1].imshow(probs,cmap='viridis',extent=[-4,4,0,0.5],aspect=10)
        axes[1].streamplot(xs,ts,dxs,dts,color='white',linewidth=0.9,density=(0.7,0.5),arrowsize=0.8)
        axes[1].plot([-4,4],[integration_times[timerow],integration_times[timerow]],c='red',zorder=100)
        axes[1].set_xlim(-4,4)
        axes[1].set_yticks([0,0.5])
        axes[1].set_xticks([])
        axes[1].set_yticklabels([r"$0$",r"$1$"])
        axes[1].set_ylabel(r"$t$")
        axes[1].set_xlabel(r"$z$")
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['bottom'].set_visible(False)
        axes[1].get_xaxis().set_visible(True)


        pos0 = axes[0].get_position(original=False)
        pos1 = axes[1].get_position(original=False)

        print(pos0.y0)
        print(pos1.y0+pos1.height)
        axes[0].set_position([pos1.x0,pos0.y0+0.15,pos1.x1,pos0.height])

        plt.savefig(os.path.join(savedir, "fig1_1d_scrub"+str(timerow)+".png"),pad_inches=0,bbox_inches='tight') 
        plt.close()

def save_fig1_1d_NF(model, data_samples, savedir, ntimes=101, memory=0.01, device='cpu',dpi=350):
    model.eval()
    

    z_samples = torch.randn(30, 1).to(device)

    # linspace for plotting
    npts=500

    z_samples = np.linspace(-4,4,100)
    z_samples = torch.from_numpy(z_samples[:,np.newaxis]).type(torch.float32).to(device)
    znp = np.linspace(-4,4,npts)
    z = torch.from_numpy(znp[:,np.newaxis]).type(torch.float32).to(device)
                    
    with torch.no_grad():
    #     # We expect the model is a chain of CNF layers wrapped in a SequentialFlow container.
        logp_samples = torch.sum(standard_normal_logprob(z_samples), -1, keepdim=True)
        logp_z = torch.sum(standard_normal_logprob(z), -1, keepdim=True)
        t = 0
        # for planar in model.chain:
            # end_time = (cnf.sqrt_end_time * cnf.sqrt_end_time)
            # integration_times = torch.linspace(0, end_time, ntimes)

        def log_prob(t):
            for planar in model.chain[t:]:
                print(t)
            z_traj,dlogp_traj = cnf(z,torch.zeros_like(logp_z),integration_times = torch.tensor([t,end_time]),reverse = False)
            z_traj = z_traj
            logp_z_traj = standard_normal_logprob(z_traj)
            dlogp_traj = dlogp_traj.cpu().numpy()
            return logp_z_traj.cpu().numpy() - dlogp_traj

            
            logp = []
            for t in integration_times:
                logp.append(log_prob(t))

            # The differential equation evaluated at some t and x.
            def _differential(t, x):
                t = torch.tensor(t).to(device)
                x = torch.tensor(x).to(device)
                return cnf.odefunc.odefunc.diffeq(t, x)

            ts = np.linspace(0,end_time,100)
            xs = np.linspace(-4,4,100)
            dxs = torch.zeros(ts.shape[0],xs.shape[0])
            
            for ti , t in enumerate(ts):
                for xi,x in enumerate(xs):
                    dxs[ti,xi]= -_differential(t,[x])

            dxs = torch.tensor(dxs)
            dts = torch.ones_like(dxs)


            # z_traj, logp_traj= cnf(z, logp_z, integration_times=integration_times, reverse=False)
            z_traj, logp_traj= cnf(z_samples, logp_samples, integration_times=integration_times, reverse=True)
            # z_traj, logp_traj= cnf(z, logp_z, integration_times=integration_times, reverse=True)
            # z_traj = z_traj.cpu().numpy()
            # logp_traj= logp_traj.cpu().numpy()

    makedirs(savedir)
    plt.clf()
    # plt.imshow(logp_traj[:,:,0],cmap='plasma')
    # plt.imshow(np.exp(np.array(logp)[:,:,0]),cmap='plasma',extent=[-4,4,0,1])
    # plt.tight_layout()
    # plt.savefig(os.path.join(savedir, "fig1_1d.jpg"))       

    # plt.clf()
    # plt.plot(z_traj[:,:,0].cpu().numpy())
    # plt.savefig(os.path.join(savedir, "fig1_1d_traj.jpg"))       

    # nm = matplotlib.colors.Normalize(0.05,0.45,True)

    probs = np.exp(np.array(logp)[:,:,0])
    maxs = np.amax(probs,axis=1,keepdims=True)
    probs = probs / maxs
    

    # plt.clf()
    # plt.axis('off')
    # plt.tight_layout()
    # plt.imshow(probs,cmap='viridis',extent=[-4,4,0,0.5],aspect=40.)
    # plt.streamplot(xs,ts,dxs,dts,color='white',linewidth=0.7,density=(0.5,2.))
    # plt.savefig(os.path.join(savedir, "fig1_1d_stream.pdf"),pad_inches=0,bbox_inches='tight')       

    # plt.clf()
    # plt.axis('off')
    # plt.tight_layout()
    # plt.plot(np.exp(np.array(logp)[0,:,0]))
    # plt.savefig(os.path.join(savedir, "fig1_1d_t1.pdf"),pad_inches=0,bbox_inches='tight')       

    # plt.clf()
    # plt.axis('off')
    # plt.tight_layout()
    # plt.plot(np.exp(np.array(logp)[-1,:,0]))
    # plt.savefig(os.path.join(savedir, "fig1_1d_t0.pdf"),pad_inches=0,bbox_inches='tight')       


    # plt.clf()
    # plt.axis('off')
    # plt.tight_layout()
    # plt.subplot2grid((8,1),(0,0))
    # plt.axis('off')
    # plt.tight_layout()
    # plt.plot(znp,np.exp(np.array(logp)[-1,:,0]))

    # plt.subplot2grid((8,1),(1,0),rowspan=6)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.imshow(probs,cmap='viridis',extent=[-4,4,0,0.5],aspect=30.)
    # plt.streamplot(xs,ts,dxs,dts,color='white',linewidth=0.7,density=(0.5,2.))

    # plt.subplot2grid((8,1),(7,0))
    # plt.axis('off')
    # plt.tight_layout()
    # plt.plot(znp,np.exp(np.array(logp)[0,:,0]))

 
    plt.rcParams.update({'font.size': 13})
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True,
                                                            gridspec_kw={'height_ratios': [1,5, 1]},
                                                            figsize=(4, 7))
    fig.set_tight_layout({'pad': 0.1, 'h_pad': -1.0})
    # axes[1].set_aspect(30, share=True)
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    # axes[0].plot(znp,np.exp(np.array(logp)[0,:,0]))
    axes[0].scatter(znp,np.exp(np.array(logp)[0,:,0]),s=0.5,marker=None,linestyle='-',c=np.exp(np.array(logp)[0,:,0]),cmap='viridis')
    axes[0].set_xlim(-4,4)
    axes[0].set_ylabel(r"$p(z(t_1))$",labelpad=20)
    axes[0].set_yticks([])
    axes[0].get_xaxis().set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    # axes[0].spines['bottom'].set_visible(False)
    # axes[0].axis('off')

    axes[1].imshow(probs,cmap='viridis',extent=[-4,4,0,0.5],aspect=10)
    axes[1].streamplot(xs,ts,dxs,dts,color='white',linewidth=0.9,density=(0.4,2.5))
    # axes[1].set_axis_off()
    axes[1].set_xlim(-4,4)
    # axes[1].axis('off')
    axes[1].set_yticks([0,0.5])
    axes[1].set_yticklabels([r"$0$",r"$1$"])
    axes[1].set_ylabel(r"$t$")
    axes[1].get_xaxis().set_visible(False)

    # axes[2].plot(znp,-np.exp(np.array(logp)[-1,:,0]))
    axes[2].scatter(znp,np.exp(np.array(logp)[-1,:,0]),s=0.5,marker=None,linestyle='-',c=np.exp(np.array(logp)[-1,:,0]),cmap='viridis')
    axes[2].set_xlim(-4,4)
    axes[2].set_ylabel(r"$p(z(t_0))$",labelpad=20)
    axes[2].set_xlabel(r"$z$")
    axes[2].set_yticks([])
    axes[2].set_xticks([])
    # axes[2].get_xaxis().set_visible(False)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)

    # fig.subplots_adjust(hspace=0.)
    pos0 = axes[0].get_position(original=False)
    pos1 = axes[1].get_position(original=False)
    pos2 = axes[2].get_position(original=False)

    print(pos0.y0)
    print(pos1.y0+pos1.height)
    axes[0].set_position([pos1.x0,pos0.y0+0.4,pos1.x1,pos0.height])
    axes[2].set_position([pos1.x0,pos2.y0,pos1.x1,pos2.height])

    plt.savefig(os.path.join(savedir, "fig1_1d_together.pdf"),pad_inches=0,bbox_inches='tight') 
# if __name__ == '__main__':

    

            # plt.figure(figsize=(8, 8))
            # for _ in range(z_traj.shape[0]):

    #             plt.clf()

    #             # plot target potential function
    #             ax = plt.subplot(2, 2, 1, aspect="equal")

    #             ax.hist2d(data_samples[:, 0], data_samples[:, 1], range=[[-4, 4], [-4, 4]], bins=200)
    #             ax.invert_yaxis()
    #             ax.get_xaxis().set_ticks([])
    #             ax.get_yaxis().set_ticks([])
    #             ax.set_title("Target", fontsize=32)

    #             # plot the density
    #             ax = plt.subplot(2, 2, 2, aspect="equal")

    #             z, logqz = grid_z_traj[t], grid_logpz_traj[t]

    #             xx = z[:, 0].reshape(npts, npts)
    #             yy = z[:, 1].reshape(npts, npts)
    #             qz = np.exp(logqz).reshape(npts, npts)

    #             plt.pcolormesh(xx, yy, qz)
    #             ax.set_xlim(-4, 4)
    #             ax.set_ylim(-4, 4)
    #             cmap = matplotlib.cm.get_cmap(None)
    #             ax.set_axis_bgcolor(cmap(0.))
    #             ax.invert_yaxis()
    #             ax.get_xaxis().set_ticks([])
    #             ax.get_yaxis().set_ticks([])
    #             ax.set_title("Density", fontsize=32)

    #             # plot the samples
    #             ax = plt.subplot(2, 2, 3, aspect="equal")

    #             zk = z_traj[t]
    #             ax.hist2d(zk[:, 0], zk[:, 1], range=[[-4, 4], [-4, 4]], bins=200)
    #             ax.invert_yaxis()
    #             ax.get_xaxis().set_ticks([])
    #             ax.get_yaxis().set_ticks([])
    #             ax.set_title("Samples", fontsize=32)

    #             # plot vector field
    #             ax = plt.subplot(2, 2, 4, aspect="equal")

    #             K = 13j
    #             y, x = np.mgrid[-4:4:K, -4:4:K]
    #             K = int(K.imag)
    #             zs = torch.from_numpy(np.stack([x, y], -1).reshape(K * K, 2)).to(device, torch.float32)
    #             logps = torch.zeros(zs.shape[0], 1).to(device, torch.float32)
    #             dydt = cnf.odefunc(integration_times[t], (zs, logps))[0]
    #             dydt = -dydt.cpu().numpy()
    #             dydt = dydt.reshape(K, K, 2)

    #             logmag = 2 * np.log(np.hypot(dydt[:, :, 0], dydt[:, :, 1]))
    #             ax.quiver(
    #                 x, y, dydt[:, :, 0], dydt[:, :, 1],
    #                 np.exp(logmag), cmap="coolwarm", scale=20., width=0.015, pivot="mid"
    #             )
    #             ax.set_xlim(-4, 4)
    #             ax.set_ylim(-4, 4)
    #             ax.axis("off")
    #             ax.set_title("Vector Field", fontsize=32)

    #             makedirs(savedir)
    #             plt.savefig(os.path.join(savedir, f"viz-{t:05d}.jpg"))
    #             t += 1


# def trajectory_to_video(savedir):
    # import subprocess
    # bashCommand = 'ffmpeg -y -i {} {}'.format(os.path.join(savedir, 'viz-%05d.jpg'), os.path.join(savedir, 'traj.mp4'))
    # process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    # output, error = process.communicate()


# if __name__ == '__main__':
    # import argparse
    # import sys

    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

    # import lib.toy_data as toy_data
    # from train_misc import count_parameters
    # from train_misc import set_cnf_options, add_spectral_norm, create_regularization_fns
    # from train_misc import build_model_toy2d

    # def get_ckpt_model_and_data(args):
    #     # Load checkpoint.
    #     checkpt = torch.load(args.checkpt, map_location=lambda storage, loc: storage)
    #     ckpt_args = checkpt['args']
    #     state_dict = checkpt['state_dict']

    #     # Construct model and restore checkpoint.
    #     regularization_fns, regularization_coeffs = create_regularization_fns(ckpt_args)
    #     model = build_model_toy2d(ckpt_args, regularization_fns).to(device)
    #     if ckpt_args.spectral_norm: add_spectral_norm(model)
    #     set_cnf_options(ckpt_args, model)

    #     model.load_state_dict(state_dict)
    #     model.to(device)

    #     print(model)
    #     print("Number of trainable parameters: {}".format(count_parameters(model)))

    #     # Load samples from dataset
    #     data_samples = toy_data.inf_train_gen(ckpt_args.data, batch_size=2000)

    #     return model, data_samples

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--checkpt', type=str, required=True)
    # parser.add_argument('--ntimes', type=int, default=101)
    # parser.add_argument('--memory', type=float, default=0.01, help='Higher this number, the more memory is consumed.')
    # parser.add_argument('--save', type=str, default='trajectory')
    # args = parser.parse_args()

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model, data_samples = get_ckpt_model_and_data(args)
    # save_trajectory(model, data_samples, args.save, ntimes=args.ntimes, memory=args.memory, device=device)
    # trajectory_to_video(args.save)

def save_fig1_1d_icml(model, data_samples, savedir, ntimes=101, memory=0.01, device='cpu',itr=''):
    model.eval()
    

    # data_samples=torch.tensor(data_samples).float().cuda()

    #  Sample from prior
    z_samples = torch.randn(30, 1).to(device)

    # linspace for plotting
    npts=500

    z_samples = np.linspace(-4,4,100)
    z_samples = torch.from_numpy(z_samples[:,np.newaxis]).type(torch.float32).to(device)
    znp = np.linspace(-4,4,npts)
    z = torch.from_numpy(znp[:,np.newaxis]).type(torch.float32).to(device)
                    
    with torch.no_grad():
    #     # We expect the model is a chain of CNF layers wrapped in a SequentialFlow container.
        logp_samples = torch.sum(standard_normal_logprob(z_samples), -1, keepdim=True)
        logp_z = torch.sum(standard_normal_logprob(z), -1, keepdim=True)
        t = 0
        for cnf in model.chain:
            end_time = (cnf.sqrt_end_time * cnf.sqrt_end_time)
            integration_times = torch.linspace(0, end_time, ntimes)

            def log_prob(t):
                z_traj,dlogp_traj = cnf(z,torch.zeros_like(logp_z),integration_times = torch.tensor([t,end_time]),reverse = False)
                z_traj = z_traj
                logp_z_traj = standard_normal_logprob(z_traj)
                dlogp_traj = dlogp_traj.cpu().numpy()
                return logp_z_traj.cpu().numpy() - dlogp_traj

            
            logp = []
            for t in integration_times:
                logp.append(log_prob(t))

            # The differential equation evaluated at some t and x.
            def _differential(t, x):
                t = torch.tensor(t).to(device)
                x = torch.tensor(x).to(device)
                return cnf.odefunc.odefunc.diffeq(t, x)

            ts = np.linspace(0,end_time,101)
            xs = np.linspace(-4,4,100)
            dxs = torch.zeros(ts.shape[0],xs.shape[0])
            
            for ti , t in enumerate(ts):
                for xi,x in enumerate(xs):
                    dxs[ti,xi]= -_differential(t,[x])

            dxs = torch.tensor(dxs)
            dts = torch.ones_like(dxs)


            # z_traj, logp_traj= cnf(z, logp_z, integration_times=integration_times, reverse=False)
            z_traj, logp_traj= cnf(z_samples, logp_samples, integration_times=integration_times, reverse=True)
            # z_traj, logp_traj= cnf(z, logp_z, integration_times=integration_times, reverse=True)
            # z_traj = z_traj.cpu().numpy()
            # logp_traj= logp_traj.cpu().numpy()

    makedirs(savedir)
    plt.clf()
    
    plt.rcParams.update({'font.size': 13})
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True,
                                                            gridspec_kw={'height_ratios': [1,5, 1]},
                                                            figsize=(4, 8))
    fig.set_tight_layout({'pad': -1.0, 'h_pad': 0.0})


    probs = np.exp(np.array(logp)[:,:,0])
    maxs = np.amax(probs,axis=1,keepdims=True)
    probs = probs
    # probs = probs / maxs

    # for smple in range(len(z_samples)):
    for smple in [38,50,55, 59]:
        animate = True
        if animate:
            Trange = range(len(integration_times))
        else:
            Trange = [len(integration_times)-1]
        for T in Trange: 
            plt.clf()
            plt.close()
        
            plt.rcParams.update({'font.size': 13})
            fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True,
                                                                    gridspec_kw={'height_ratios': [1,5, 1]},
                                                                    figsize=(4, 7))
            fig.set_tight_layout({'pad': -1.0, 'h_pad': 0.0})

            print("plotting ",T)
            ztr0 = z_traj.numpy()[0,smple,0]
            pztr0 = np.exp(np.array(logp_traj)[0,smple,0])
            ztrT = z_traj.numpy()[T,smple,0]
            pztrT = np.exp(np.array(logp_traj)[T,smple,0])


            probts = np.exp(np.array(logp))[::-1]

            sc0 = axes[0].scatter(znp,probts[T,:,0],s=0.5,marker=None,linestyle='-',c=probts[T,:,0],cmap='viridis',zorder=3)
            sc0.set_clim(0.,maxs[-1][0])
            axes[0].scatter(ztrT,[0.], color="#F012BE",s=10.,zorder=5,clip_on=False)
            axes[0].plot([ztr0, ztr0],[0.,pztr0], color="#F012BE",linestyle='--',alpha=0.4,linewidth=0.5,zorder=2)
            axes[0].plot([ztr0, znp[0]],[pztr0,pztr0], color="#39CCCC",linestyle='--',alpha=0.4,linewidth=0.5,clip_on=False,zorder=2)
            axes[0].plot([ztrT, ztrT],[0.,pztrT], color="#F012BE",linestyle='--',alpha=0.8,linewidth=0.5,zorder=4)
            axes[0].plot([ztrT, znp[0]],[pztrT,pztrT], color="#2ECC40",linestyle='--',alpha=0.8,linewidth=0.5,zorder=4)
            axes[0].plot([znp[0], znp[0]],[pztr0,pztrT], color="#2ECC40",alpha=1.0,linewidth=1,zorder=10,clip_on=False)
            axes[0].set_xlim(-4,4)
            axes[0].set_ylim(0.,maxs[-1][0])
            axes[0].set_ylabel(r"$p(z_{t})$",labelpad=20)
            axes[0].set_yticks([min(0.9*pztr0,(pztrT+pztr0)/2)])
            axes[0].set_yticklabels([r"$\Delta$"], color="#2ECC40")
            axes[0].tick_params(width=0,labelsize=10)
            axes[0].get_xaxis().set_visible(False)
            axes[0].spines['top'].set_visible(False)
            axes[0].spines['right'].set_visible(False)
            # axes[0].set_clip_on(False)
            axes[0].set_zorder(2)

            axes[1].imshow(probs,cmap='viridis',extent=[-4,4,0,0.5],aspect=10)
            axes[1].plot(z_traj[0:T,smple,0].numpy(),ts[0:T], color="#F012BE")
            axes[1].scatter([ztr0],[0.], color="#F012BE", s=10.,zorder=5)
            axes[1].scatter([ztrT],[ts[T]], color="#F012BE",s=10.,zorder=5,clip_on=True)
            axes[1].scatter([ztrT],[ts[-1]], color="#F012BE",s=10.,zorder=5,clip_on=True,visible=False)#hack
            axes[1].streamplot(xs,ts,dxs,dts,color='white',linewidth=0.3,density=(0.7,0.5),arrowsize=0.5)
            axes[1].set_xlim(-4,4)
            axes[1].set_yticks([0,0.5])
            axes[1].set_yticklabels([r"$0$",r"$1$"])
            axes[1].set_ylabel(r"$t$")
            # axes[1].get_xaxis().set_visible(False)
            axes[1].spines['bottom'].set_position('zero')
            # axes[1].spines['bottom'].set_zorder(0.)
            axes[1].spines['top'].set_position(('data',0.5))
            # axes[1].spines['top'].set_zorder(0.)
            axes[1].spines['left'].set_bounds(0.,0.5)
            axes[1].spines['right'].set_bounds(0.,0.5)
            axes[1].set_clip_on(True)



            # axes[2].plot(znp,-np.exp(np.array(logp)[-1,:,0]))
            sc2 = axes[2].scatter(znp,np.exp(np.array(logp)[-1,:,0]),s=0.5,marker=None,linestyle='-',c=probs[-1,:],cmap='viridis')
            axes[2].scatter([ztr0],[0.],color="#F012BE",s=10.,zorder=5,clip_on=False)
            axes[2].plot([ztr0, ztr0],[0.,pztr0], color="#F012BE",linestyle='--',alpha=0.8,linewidth=0.5)
            axes[2].plot([ztr0, znp[0]],[pztr0,pztr0], color="#39CCCC",linestyle='--',alpha=0.8,linewidth=0.5,clip_on=False)
            axes[2].set_xlim(-4,4)
            axes[2].set_ylabel(r"$p(z_{t_0})$",labelpad=20)
            axes[2].set_xlabel(r"$z$")
            axes[2].set_yticks([])
            axes[2].set_xticks([])
            # axes[2].get_xaxis().set_visible(False)
            axes[2].spines['top'].set_visible(False)
            axes[2].spines['right'].set_visible(False)
            axes[2].set_ylim(bottom=0.)

            # fig.subplots_adjust(hspace=0.)
            pos0 = axes[0].get_position(original=False)
            pos1 = axes[1].get_position(original=False)
            pos2 = axes[2].get_position(original=False)

            # print(pos0.y0)
            # print(pos1.y0+pos1.height)
            axes[0].set_position([pos1.x0,pos0.y0+0.4,pos1.x1,pos0.height])
            axes[2].set_position([pos1.x0,pos2.y0,pos1.x1,pos2.height])

            if animate:
                # utils.mkdirs(os.path.join(savedir, "anim",'{:0>4}'.format(str(smple))))
                plt.savefig(os.path.join(savedir, "anim",'{:0>4}'.format(str(smple)),"img-"+'{:0>4}'.format(str(T))+".png"),pad_inches=0.05,bbox_inches='tight',dpi=300) 

        plt.savefig(os.path.join(savedir, "fig1_1d_together"+'{:0>4}'.format(str(smple))+".png"),pad_inches=0.05,bbox_inches='tight',dpi=350) 

#;
# ffmpeg -r 24  -i experiments/fig1_1d_toy/fig1_ani/anim/0038/%04.png -c:v libx264 -crf 20   -pix_fmt yuv420p  experiments/fig1_1d_toy/fig1_ani/animate.mp4 
# ffmpeg -r 24 -i %03.png -c:v libx264 -crf 20   -pix_fmt yuv420p animate.mp4 

# ffmpeg -f concat -safe 0 -i anim-list -r 24 -c:v libx264 -crf 20   -pix_fmt yuv420p ffjord-sample-rev.mp4 

def save_fig1_1d_icml_rev(model, data_samples, savedir, ntimes=101, memory=0.01, device='cpu',itr=''):
    model.eval()
    

    # data_samples=torch.tensor(data_samples).float().cuda()

    #  Sample from prior
    z_samples = torch.randn(30, 1).to(device)

    # linspace for plotting
    npts=500

    z_samples = np.linspace(-4,4,100)
    z_samples = torch.from_numpy(z_samples[:,np.newaxis]).type(torch.float32).to(device)
    znp = np.linspace(-4,4,npts)
    z = torch.from_numpy(znp[:,np.newaxis]).type(torch.float32).to(device)
                    
    with torch.no_grad():
    #     # We expect the model is a chain of CNF layers wrapped in a SequentialFlow container.
        logp_samples = torch.sum(standard_normal_logprob(z_samples), -1, keepdim=True)
        logp_z = torch.sum(standard_normal_logprob(z), -1, keepdim=True)
        t = 0
        for cnf in model.chain:
            end_time = (cnf.sqrt_end_time * cnf.sqrt_end_time)
            integration_times = torch.linspace(0, end_time, ntimes)

            def log_prob(t):
                z_traj,dlogp_traj = cnf(z,torch.zeros_like(logp_z),integration_times = torch.tensor([t,end_time]),reverse = False)
                z_traj = z_traj
                logp_z_traj = standard_normal_logprob(z_traj)
                dlogp_traj = dlogp_traj.cpu().numpy()
                return logp_z_traj.cpu().numpy() - dlogp_traj

            
            logp = []
            for t in integration_times:
                logp.append(log_prob(t))

            # The differential equation evaluated at some t and x.
            def _differential(t, x):
                t = torch.tensor(t).to(device)
                x = torch.tensor(x).to(device)
                return cnf.odefunc.odefunc.diffeq(t, x)

            ts = np.linspace(0,end_time,101)
            xs = np.linspace(-4,4,100)
            dxs = torch.zeros(ts.shape[0],xs.shape[0])
            
            for ti , t in enumerate(ts):
                for xi,x in enumerate(xs):
                    dxs[ti,xi]= -_differential(t,[x])

            dxs = torch.tensor(dxs)
            dts = torch.ones_like(dxs)


            # z_traj, logp_traj= cnf(z, logp_z, integration_times=integration_times, reverse=False)
            z_traj, logp_traj= cnf(z_samples, logp_samples, integration_times=integration_times, reverse=True)
            z_traj = z_traj.numpy()[::-1,:,:]
            logp_traj = logp_traj.numpy()[::-1,:,:]
            # z_traj, logp_traj= cnf(z, logp_z, integration_times=integration_times, reverse=True)
            # z_traj = z_traj.cpu().numpy()
            # logp_traj= logp_traj.cpu().numpy()

    makedirs(savedir)
    plt.clf()
    
    plt.rcParams.update({'font.size': 13})
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True,
                                                            gridspec_kw={'height_ratios': [1,5, 1]},
                                                            figsize=(4, 8))
    fig.set_tight_layout({'pad': -1.0, 'h_pad': 0.0})


    probs = np.exp(np.array(logp)[:,:,0])
    maxs = np.amax(probs,axis=1,keepdims=True)
    probs = probs[::-1]
    # probs = probs / maxs

    # for smple in range(len(z_samples)):
    for smple in [38,50,55, 59]:
        animate = True
        if animate:
            Trange = range(len(integration_times))
        else:
            Trange = [len(integration_times)-1]
        for T in Trange: 
            plt.clf()
            plt.close()
        
            plt.rcParams.update({'font.size': 13})
            fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True,
                                                                    gridspec_kw={'height_ratios': [1,5, 1]},
                                                                    figsize=(4, 7))
            fig.set_tight_layout({'pad': -1.0, 'h_pad': 0.0})

            print("plotting ",T)
            ztr0 = z_traj[0,smple,0]
            pztr0 = np.exp(np.array(logp_traj)[0,smple,0])
            ztrT = z_traj[T,smple,0]
            pztrT = np.exp(np.array(logp_traj)[T,smple,0])


            probts = np.exp(np.array(logp))

            sc0 = axes[0].scatter(znp,probts[T,:,0],s=0.5,marker=None,linestyle='-',c=probts[T,:,0],cmap='viridis',zorder=3)
            sc0.set_clim(0.,maxs[-1][0])
            axes[0].scatter(ztrT,[0.], color="#F012BE",s=10.,zorder=5,clip_on=False)
            axes[0].plot([ztr0, ztr0],[0.,pztr0], color="#F012BE",linestyle='--',alpha=0.4,linewidth=0.5,zorder=2)
            axes[0].plot([ztr0, znp[0]],[pztr0,pztr0], color="#39CCCC",linestyle='--',alpha=0.4,linewidth=0.5,clip_on=False,zorder=2)
            axes[0].plot([ztrT, ztrT],[0.,pztrT], color="#F012BE",linestyle='--',alpha=0.8,linewidth=0.5,zorder=4)
            axes[0].plot([ztrT, znp[0]],[pztrT,pztrT], color="#2ECC40",linestyle='--',alpha=0.8,linewidth=0.5,zorder=4)
            axes[0].plot([znp[0], znp[0]],[pztr0,pztrT], color="#2ECC40",alpha=1.0,linewidth=1,zorder=10,clip_on=False)
            axes[0].set_xlim(-4,4)
            axes[0].set_ylim(0.,maxs[-1][0])
            axes[0].set_ylabel(r"$p(z_{t})$",labelpad=20)
            axes[0].set_yticks([min(0.9*pztr0,(pztrT+pztr0)/2)])
            axes[0].set_yticklabels([r"$\Delta$"], color="#2ECC40")
            axes[0].tick_params(width=0,labelsize=10)
            axes[0].get_xaxis().set_visible(False)
            axes[0].spines['top'].set_visible(False)
            axes[0].spines['right'].set_visible(False)
            # axes[0].set_clip_on(False)
            axes[0].set_zorder(2)

            axes[1].imshow(probs,cmap='viridis',extent=[-4,4,0,0.5],aspect=10)
            axes[1].plot(z_traj[0:T,smple,0],ts[0:T], color="#F012BE")
            axes[1].scatter([ztr0],[0.], color="#F012BE", s=10.,zorder=5)
            axes[1].scatter([ztrT],[ts[T]], color="#F012BE",s=10.,zorder=5,clip_on=True)
            axes[1].scatter([ztrT],[ts[-1]], color="#F012BE",s=10.,zorder=5,clip_on=True,visible=False)#hack
            axes[1].streamplot(xs,ts,dxs.numpy()[:,::-1],dts,color='white',linewidth=0.3,density=(0.7,0.5),arrowsize=0.5)
            axes[1].set_xlim(-4,4)
            axes[1].set_yticks([0,0.5])
            axes[1].set_yticklabels([r"$0$",r"$1$"])
            axes[1].set_ylabel(r"$t$")
            # axes[1].get_xaxis().set_visible(False)
            axes[1].spines['bottom'].set_position('zero')
            # axes[1].spines['bottom'].set_zorder(0.)
            axes[1].spines['top'].set_position(('data',0.5))
            # axes[1].spines['top'].set_zorder(0.)
            axes[1].spines['left'].set_bounds(0.,0.5)
            axes[1].spines['right'].set_bounds(0.,0.5)
            axes[1].set_clip_on(True)



            # axes[2].plot(znp,-np.exp(np.array(logp)[-1,:,0]))
            sc2 =axes[2].scatter(znp,probs[-1,:],s=0.5,marker=None,linestyle='-',c=probs[-1,:],cmap='viridis')
            sc2.set_clim(0.,maxs[-1][0])
            axes[2].scatter([ztr0],[0.],color="#F012BE",s=10.,zorder=5,clip_on=False)
            axes[2].plot([ztr0, ztr0],[0.,pztr0], color="#F012BE",linestyle='--',alpha=0.8,linewidth=0.5)
            axes[2].plot([ztr0, znp[0]],[pztr0,pztr0], color="#39CCCC",linestyle='--',alpha=0.8,linewidth=0.5,clip_on=False)
            axes[2].set_xlim(-4,4)
            axes[2].set_ylabel(r"$p(z_{t_T})$",labelpad=20)
            axes[2].set_xlabel(r"$z$")
            axes[2].set_yticks([])
            axes[2].set_xticks([])
            # axes[2].get_xaxis().set_visible(False)
            axes[2].spines['top'].set_visible(False)
            axes[2].spines['right'].set_visible(False)
            axes[2].set_ylim(0.,maxs[-1][0])

            # fig.subplots_adjust(hspace=0.)
            pos0 = axes[0].get_position(original=False)
            pos1 = axes[1].get_position(original=False)
            pos2 = axes[2].get_position(original=False)

            # print(pos0.y0)
            # print(pos1.y0+pos1.height)
            axes[0].set_position([pos1.x0,pos0.y0+0.4,pos1.x1,pos0.height])
            axes[2].set_position([pos1.x0,pos2.y0,pos1.x1,pos2.height])

            if animate:
                # utils.mkdirs(os.path.join(savedir, "anim",'{:0>4}'.format(str(smple))))
                plt.savefig(os.path.join(savedir, "anim_rev",'{:0>4}'.format(str(smple)),"img-"+'{:0>4}'.format(str(T))+".png"),pad_inches=0.05,bbox_inches='tight',dpi=300) 

        plt.savefig(os.path.join(savedir, "fig1_1d_together_rev"+'{:0>4}'.format(str(smple))+".png"),pad_inches=0.05,bbox_inches='tight',dpi=350) 
def save_fig1_1d_ptd_timescrub(model, data_samples, savedir, ntimes=101, memory=0.01, device='cpu',itr=''):
    model.eval()
    

    # data_samples=torch.tensor(data_samples).float().cuda()

    #  Sample from prior
    z_samples = torch.randn(30, 1).to(device)

    # linspace for plotting
    npts=500

    z_samples = np.linspace(-4,4,100)
    z_samples = torch.from_numpy(z_samples[:,np.newaxis]).type(torch.float32).to(device)
    znp = np.linspace(-4,4,npts)
    z = torch.from_numpy(znp[:,np.newaxis]).type(torch.float32).to(device)
                    
    with torch.no_grad():
    #     # We expect the model is a chain of CNF layers wrapped in a SequentialFlow container.
        logp_samples = torch.sum(standard_normal_logprob(z_samples), -1, keepdim=True)
        logp_z = torch.sum(standard_normal_logprob(z), -1, keepdim=True)
        t = 0
        for cnf in model.chain:
            end_time = (cnf.sqrt_end_time * cnf.sqrt_end_time)
            integration_times = torch.linspace(0, end_time, ntimes)

            def log_prob(t):
                z_traj,dlogp_traj = cnf(z,torch.zeros_like(logp_z),integration_times = torch.tensor([t,end_time]),reverse = False)
                z_traj = z_traj
                logp_z_traj = standard_normal_logprob(z_traj)
                dlogp_traj = dlogp_traj.cpu().numpy()
                return logp_z_traj.cpu().numpy() - dlogp_traj

            
            logp = []
            for t in integration_times:
                logp.append(log_prob(t))

            # The differential equation evaluated at some t and x.
            def _differential(t, x):
                t = torch.tensor(t).to(device)
                x = torch.tensor(x).to(device)
                return cnf.odefunc.odefunc.diffeq(t, x)

            ts = np.linspace(0,end_time,100)
            xs = np.linspace(-4,4,100)
            dxs = torch.zeros(ts.shape[0],xs.shape[0])
            
            for ti , t in enumerate(ts):
                for xi,x in enumerate(xs):
                    dxs[ti,xi]= -_differential(t,[x])

            dxs = torch.tensor(dxs)
            dts = torch.ones_like(dxs)


            # z_traj, logp_traj= cnf(z, logp_z, integration_times=integration_times, reverse=False)
            z_traj, logp_traj= cnf(z_samples, logp_samples, integration_times=integration_times, reverse=True)
            # z_traj, logp_traj= cnf(z, logp_z, integration_times=integration_times, reverse=True)
            # z_traj = z_traj.cpu().numpy()
            # logp_traj= logp_traj.cpu().numpy()

    makedirs(savedir)
    for timerow in range(integration_times.shape[0]):
        plt.clf()

        probs = np.exp(np.array(logp)[:,:,0])
        maxs = np.amax(probs,axis=1,keepdims=True)
        probs = probs / maxs


        plt.rcParams.update({'font.size': 13})
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True,
                                                                gridspec_kw={'height_ratios': [1,5]},
                                                                figsize=(8,13))
        fig.set_tight_layout({'pad': 0.1, 'h_pad': -1.0})

        axes[0].scatter(znp,np.exp(np.array(logp)[::-1][timerow,:,0]),s=0.5,marker=None,linestyle='-',c=np.exp(np.array(logp)[::-1][timerow,:,0]),cmap='viridis')
        axes[0].set_xlim(-4,4)
        axes[0].set_ylim(0,.42)
        axes[0].set_ylabel(r"$p(z(t))$",labelpad=20)

        axes[0].set_yticks([])
        axes[0].get_xaxis().set_visible(False)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[0].set_ylim(bottom=0.0)

        axes[1].imshow(probs,cmap='viridis',extent=[-4,4,0,0.5],aspect=10)
        axes[1].streamplot(xs,ts,dxs,dts,color='white',linewidth=0.9,density=(0.7,0.5),arrowsize=0.8)
        axes[1].plot([-4,4],[integration_times[timerow],integration_times[timerow]],c='red',zorder=100)
        axes[1].set_xlim(-4,4)
        axes[1].set_yticks([0,0.5])
        axes[1].set_xticks([])
        axes[1].set_yticklabels([r"$0$",r"$1$"])
        axes[1].set_ylabel(r"$t$")
        axes[1].set_xlabel(r"$z$")
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['bottom'].set_visible(False)
        axes[1].get_xaxis().set_visible(True)


        pos0 = axes[0].get_position(original=False)
        pos1 = axes[1].get_position(original=False)

        print(pos0.y0)
        print(pos1.y0+pos1.height)
        axes[0].set_position([pos1.x0,pos0.y0+0.15,pos1.x1,pos0.height])

        plt.savefig(os.path.join(savedir, "fig1_1d_scrub"+str(timerow)+".png"),pad_inches=0,bbox_inches='tight') 
        plt.close()
