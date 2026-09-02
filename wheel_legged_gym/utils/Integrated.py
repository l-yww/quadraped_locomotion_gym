import torch.nn.functional as F
import torch
import copy
import os

# --------------- vallina ------------------
class Integrated_vallina_policy(torch.nn.Module):
    def __init__(self, actor, num_single_obs, frame_stack, actor_input_stack):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu()
        self.num_single_obs = num_single_obs
        self.actor_input_stack = actor_input_stack
        self.frame_stack = frame_stack

    def forward(self, obs):
        action_mean = self.actor.forward(torch.cat([obs[:, (self.frame_stack - self.actor_input_stack) * self.num_single_obs:]], dim=-1))
        #return torch.tanh(action_mean) * 4.5  #防爆炸，又有足够运动空间
        return action_mean  

    def export(self, path, name):
        path = os.path.join(path, f"{name}.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

    def export_onnx(self, path, name):
        """Export policy to ONNX format."""
        onnx_path = os.path.join(path, f"{name}.onnx")
        self.to("cpu")
        self.eval()
        input_dim = self.frame_stack * self.num_single_obs
        dummy_input = torch.zeros(1, input_dim, dtype=torch.float32)
        torch.onnx.export(
            self,
            dummy_input,
            onnx_path,
            input_names=["obs"],
            output_names=["actions"],
            opset_version=14,
            do_constant_folding=True,
        )
        print(f"  ONNX input dim: {input_dim}, saved to: {onnx_path}")

# --------------- RMA ------------------
class Integrated_RMA_policy(torch.nn.Module):
    def __init__(self, actor, adaptation_encoder, num_single_obs, frame_stack, actor_input_stack):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu()
        self.adaptation_encoder = copy.deepcopy(adaptation_encoder).cpu()
        self.num_single_obs = num_single_obs
        self.actor_input_stack = actor_input_stack
        self.frame_stack = frame_stack
    
    def forward(self, obs):
        latent = self.adaptation_encoder.forward(obs)
        action_mean = self.actor.forward(torch.cat([obs[:, (self.frame_stack - self.actor_input_stack) * self.num_single_obs:], latent], dim=-1))
        return action_mean

    def export(self, path, name):
        path = os.path.join(path, f"{name}.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)




# zsy added
# --------------- PPO-height-scans ------------------
class Integrated_PPO_policy(torch.nn.Module):
    def __init__(self, actor, height_scan_encoder, num_height_scan_input, 
                num_single_obs, frame_stack):
        super().__init__() 
        self.actor = copy.deepcopy(actor).cpu()
        self.height_scan_encoder = copy.deepcopy(height_scan_encoder).cpu()
        self.num_height_scan_input = num_height_scan_input
        self.num_single_obs = num_single_obs
        self.frame_stack = frame_stack
        

    # obs: (one-height)+(Five-history)
    ## NOTE: [121,25*5]
    def forward(self, obs):
        obs_height = obs[:, :self.num_height_scan_input] # 121
        obs_input = obs[:, self.num_height_scan_input: ] # 125
        height_latent = self.height_scan_encoder.forward(obs_height) #hist obs    
        action_mean = self.actor.forward(torch.cat([height_latent, obs_input], dim=-1))
        return action_mean


    def export(self, path, name):
        path = os.path.join(path, f"{name}.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

# zsy added
# --------------- ROA ------------------
class Integrated_ROA_policy(torch.nn.Module):
    def __init__(self, actor, history_encoder, height_scan_encoder, 
                num_height_scan_input, 
                num_single_obs, frame_stack, actor_input_stack):
        super().__init__() 
        self.actor = copy.deepcopy(actor).cpu()
        self.history_encoder = copy.deepcopy(history_encoder).cpu()
        self.height_scan_encoder = copy.deepcopy(height_scan_encoder).cpu()
        self.num_single_obs = num_single_obs
        self.actor_input_stack = actor_input_stack
        self.frame_stack = frame_stack
        self.num_height_scan_input = num_height_scan_input
        self.obs_now_start_point = (self.frame_stack - self.actor_input_stack) * self.num_single_obs

    # obs: (one-height)+(Five-history)
    ## NOTE: [121,25*5]
    def forward(self, obs): 
        obs_height = obs[:, :self.num_height_scan_input] #121
        obs_hist = obs[:, self.num_height_scan_input:] #25x5
        obs_input = obs[:, self.num_height_scan_input + self.obs_now_start_point:] #25
        height_latent = self.height_scan_encoder.forward(obs_height) #hist obs
        latent = self.history_encoder.forward(obs_hist) #hist obs     
        action_mean = self.actor.forward(torch.cat([obs_input, latent, height_latent], dim=-1))
        return action_mean


    def export(self, path, name):
        path = os.path.join(path, f"{name}.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

# --------------- EST ------------------
class Integrated_EST_policy(torch.nn.Module):
    def __init__(self, actor, estimator, height_scan_encoder,
                 num_height_scan_input,
                num_single_obs, frame_stack, actor_input_stack):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu()
        self.estimator = copy.deepcopy(estimator).cpu()
        self.height_scan_encoder = copy.deepcopy(height_scan_encoder).cpu()
        self.num_single_obs = num_single_obs
        self.actor_input_stack = actor_input_stack
        self.frame_stack = frame_stack
        self.num_height_scan_input = num_height_scan_input
        self.obs_now_start_point = (self.frame_stack - self.actor_input_stack) * self.num_single_obs


    def forward(self, obs): 
        obs_height = obs[:, :self.num_height_scan_input] #121
        obs_hist = obs[:, self.num_height_scan_input:] #25x5
        obs_input = obs[:, self.num_height_scan_input + self.obs_now_start_point:] #25
        height_latent = self.height_scan_encoder.forward(obs_height) #hist obs
        height_latent_multi = torch.cat([height_latent, height_latent, height_latent, height_latent, height_latent], dim=-1)
        latent = self.estimator.forward(obs_hist) #hist obs     
        action_mean = self.actor.forward(torch.cat([obs_input, latent, height_latent_multi], dim=-1))
        return action_mean


    def export(self, path, name):
        path = os.path.join(path, f"{name}.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

# --------------- HIM ------------------
class Integrated_HIM_policy(torch.nn.Module):
    def __init__(self, actor, encoder, num_single_obs, num_est_prob):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu()
        self.encoder = copy.deepcopy(encoder).cpu()
        self.num_single_obs = num_single_obs
        self.num_est_prob = num_est_prob

    def forward(self, obs):
        parts = self.encoder.forward(obs)
        vel, height, arm_2_root_pos, z = parts[..., :3], parts[..., 3:4], parts[..., 4:self.num_est_prob], parts[..., self.num_est_prob:]
        z = F.normalize(z, dim=-1, p=2.0)
        action_mean = self.actor.forward(torch.cat([obs[:,-self.num_single_obs:], vel, height, arm_2_root_pos, z], dim=-1))
        output = torch.cat((action_mean, vel, height, arm_2_root_pos), dim=-1) #校验
        return output


    def export(self, path, name):
        path = os.path.join(path, f"{name}.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


# teacher student
# --------------- TS ------------------
class Integrated_TS_policy(torch.nn.Module):
    def __init__(self, actor, estimator, height_scan_encoder,
                num_height_scan_input,
                num_single_obs, frame_stack):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu()
        self.estimator = copy.deepcopy(estimator).cpu()
        self.height_scan_encoder = copy.deepcopy(height_scan_encoder).cpu()
        self.num_single_obs = num_single_obs
        self.frame_stack = frame_stack
        self.num_height_scan_input = num_height_scan_input


    def forward(self, obs): 
        obs_height = obs[:, :self.num_height_scan_input] #121
        obs_hist = obs[:, self.num_height_scan_input:] #25*25
        obs_input = obs_hist
        height_latent = self.height_scan_encoder.forward(obs_height) #hist obs
        latent = self.estimator.forward(obs_hist) #hist obs     
        action_mean = self.actor.forward(torch.cat([obs_input, latent, height_latent], dim=-1))
        return action_mean

    def export(self, path, name):
        path = os.path.join(path, f"{name}.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

class Integrated_EST_Plane_policy(torch.nn.Module):
    def __init__(self, actor, estimator, num_single_obs, frame_stack, actor_input_stack):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu()
        self.estimator = copy.deepcopy(estimator).cpu()
        self.num_single_obs = num_single_obs
        self.actor_input_stack = actor_input_stack
        self.frame_stack = frame_stack

    def forward(self, obs):
        latent = self.estimator.forward(obs)
        action_mean = self.actor.forward(torch.cat([obs[:, (self.frame_stack - self.actor_input_stack) * self.num_single_obs:], latent], dim=-1))
        output = torch.cat((action_mean, latent), dim=-1)
        return output

    def export(self, path,name):
        path = os.path.join(path, f"{name}.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path) 

class Integrated_DH_policy(torch.nn.Module):
    def __init__(self, actor, estimator, long_history_encoder, num_single_obs, frame_stack, short_frame_stack, in_channels, num_proprio_obs, long_history_type):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu()
        self.estimator = copy.deepcopy(estimator).cpu()
        self.long_history_encoder = copy.deepcopy(long_history_encoder).cpu()
        self.num_single_obs = num_single_obs
        self.num_short_obs = short_frame_stack * num_single_obs
        self.frame_stack = frame_stack
        self.in_channels = in_channels
        self.num_proprio_obs = num_proprio_obs
        self.long_history_type = long_history_type

    def forward(self, obs):
        with torch.no_grad():
            short_history = obs[..., -self.num_short_obs:]
            estimated_prob = self.estimator(short_history)
            if self.long_history_type == 'cnn':
                # reshaped_long_history = obs.view(-1, self.in_channels, self.num_proprio_obs)
                reshaped_long_history = obs.view(-1, self.num_proprio_obs, self.in_channels)
            else:
                reshaped_long_history = obs
            compressed_long_history = self.long_history_encoder(reshaped_long_history)
            actor_obs = torch.cat((short_history, estimated_prob, compressed_long_history),dim=-1)
            action_mean = self.actor(actor_obs)
            output = action_mean.clone().detach()
            # 删除中间变量
            # del reshaped_long_history, actor_obs, short_history, estimated_prob, compressed_long_history, action_mean
            return output
        
    def export(self, path,name):
        path = os.path.join(path, f"{name}.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path) 


class Integrated_Prop_policy(torch.nn.Module):
    def __init__(self, actor, estimator, long_history_encoder, num_single_obs, frame_stack, short_frame_stack, in_channels, num_proprio_obs):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu()
        self.estimator = copy.deepcopy(estimator).cpu()
        self.long_history_encoder = copy.deepcopy(long_history_encoder).cpu()
        self.num_single_obs = num_single_obs
        self.num_short_obs = short_frame_stack * num_single_obs
        self.frame_stack = frame_stack
        self.in_channels = in_channels
        self.num_proprio_obs = num_proprio_obs

    def forward(self, obs, commands):
        with torch.no_grad():
            short_history = obs[..., -self.num_short_obs:]
            estimated_prob = self.estimator(short_history)
            reshaped_long_history = obs.view(-1, self.in_channels, self.num_proprio_obs)
            compressed_long_history = self.long_history_encoder(reshaped_long_history)
            actor_obs = torch.cat((commands, short_history, estimated_prob, compressed_long_history),dim=-1)
            action_mean = self.actor(actor_obs)
            output = action_mean.clone().detach()
            # 删除中间变量
            # del reshaped_long_history, actor_obs, short_history, estimated_prob, compressed_long_history, action_mean
            return output
        
    def export(self, path,name):
        path = os.path.join(path, f"{name}.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path) 


class Integrated_VAE_policy(torch.nn.Module):
    def __init__(self, actor_critic, num_single_obs, frame_stack, actor_input_stack):
        super().__init__()

        self.actor_critic = copy.deepcopy(actor_critic).cpu()
        self.num_single_obs = num_single_obs
        self.actor_input_stack = actor_input_stack
        self.frame_stack = frame_stack

    def forward(self, obs_history):
        _, _, _, z = self.actor_critic.vae_forward(obs_history)

        current_obs = obs_history[:, (self.frame_stack - self.actor_input_stack) * self.num_single_obs:]
        combined_obs = torch.cat([current_obs, z], dim=-1)
        action_mean = self.actor_critic.actor(combined_obs)
        output = torch.cat((action_mean, z), dim=-1)
        return output

    def export(self, path, name):
        path = os.path.join(path, f"{name}.pt")
        self.to("cpu")
        example_input = torch.randn(1, self.frame_stack * self.num_single_obs)
        traced_script_module = torch.jit.trace(self, example_input)
        traced_script_module.save(path)


class Integrated_DH_Map_policy(torch.nn.Module):
    def __init__(self, actor, estimator, long_history_encoder, height_history_encoder, 
                 num_single_obs, frame_stack, short_frame_stack, num_height_maps, 
                 height_history_len, in_channels, num_proprio_obs, long_history_type):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu()
        self.estimator = copy.deepcopy(estimator).cpu()
        self.long_history_encoder = copy.deepcopy(long_history_encoder).cpu()
        self.height_history_encoder = copy.deepcopy(height_history_encoder).cpu()
        self.num_single_obs = num_single_obs
        self.num_short_obs = short_frame_stack * num_single_obs
        self.frame_stack = frame_stack
        self.in_channels = in_channels
        self.num_proprio_obs = num_proprio_obs
        self.long_history_type = long_history_type
        self.num_height_maps = num_height_maps
        self.height_history_len = height_history_len

    def forward(self, obs, height_maps):
        with torch.no_grad():
            short_history = obs[..., -self.num_short_obs:]
            estimated_prob = self.estimator(short_history)
            reshaped_long_history = obs.view(-1, self.num_proprio_obs, self.in_channels)
            compressed_long_history = self.long_history_encoder(reshaped_long_history)
            reshaped_height_history = height_maps.view(-1, self.num_height_maps, self.height_history_len)
            compressed_height_history = self.height_history_encoder(reshaped_height_history)
            actor_obs = torch.cat((short_history, estimated_prob, compressed_long_history, compressed_height_history),dim=-1)
            action_mean = self.actor(actor_obs)
            output = action_mean.clone().detach()
            return output

    def export(self, path,name):
        path = os.path.join(path, f"{name}.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

# --------------- MoE CTS ------------------
class Integrated_MoE_CTS_policy(torch.nn.Module):
    def __init__(self, actor, student_moe_encoder, num_obs, history_length):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu()
        self.student_moe_encoder = copy.deepcopy(student_moe_encoder).cpu()
        self.num_obs = num_obs
        self.history_length = history_length

    def forward(self, history_flat):
        # history_flat: [batch, 1475] = [batch, history_length * num_obs]
        # sim2real: 外部维护 history buffer，每次直接传入 1475 维
        latent, _ = self.student_moe_encoder(history_flat)
        current_obs = history_flat[:, -self.num_obs:]  # 取最新帧 [295]
        x = torch.cat([latent, current_obs], dim=-1)
        action_mean = self.actor(x)
        return action_mean

    def export(self, path, name):
        path = os.path.join(path, f"{name}.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


# --------------- HIM plane (flat ground) ------------------
class Integrated_HIM_Plane_policy(torch.nn.Module):
    """Integrated policy for HIM on flat ground.
    Mirrors ActorCritic_HIM.act_inference:
      short_history = obs_history[:, -num_short_obs:]
      estimated_prob, latent = estimator(obs_history)
      actor_obs = cat(short_history, estimated_prob, latent)
      actions_mean = actor(actor_obs)
    """
    def __init__(self, actor, estimator, num_single_obs, frame_stack,
                 short_frame_stack, num_est_prob, lh_output_dim):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu()
        self.estimator = copy.deepcopy(estimator).cpu()
        self.num_single_obs = num_single_obs
        self.num_short_obs = short_frame_stack * num_single_obs
        self.frame_stack = frame_stack

    def forward(self, obs_history):
        with torch.no_grad():
            short_history = obs_history[:, -self.num_short_obs:]
            estimated_prob, latent = self.estimator(obs_history)
            actor_obs = torch.cat((short_history, estimated_prob, latent), dim=-1)
            action_mean = self.actor(actor_obs)
        return action_mean

    def export(self, path, name):
        path = os.path.join(path, f"{name}.pt")
        self.to("cpu")
        dummy = torch.randn(1, self.frame_stack * self.num_single_obs)
        traced_script_module = torch.jit.trace(self, dummy)
        traced_script_module.save(path)

    def export_onnx(self, path, name):
        onnx_path = os.path.join(path, f"{name}.onnx")
        self.to("cpu")
        self.eval()
        dummy = torch.randn(1, self.frame_stack * self.num_single_obs)
        torch.onnx.export(self, dummy, onnx_path, opset_version=11,
                          input_names=["obs_history"],
                          output_names=["actions"])


class Integrated_HIM_HeightScan_policy(torch.nn.Module):
    """Export wrapper for the height-scan HIM history-latent policy."""

    def __init__(self, actor, estimator, num_single_obs, frame_stack):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu()
        self.estimator = copy.deepcopy(estimator).cpu()
        self.num_single_obs = num_single_obs
        self.num_history_obs = num_single_obs * frame_stack

    def forward(self, obs):
        history = obs[:, -self.num_history_obs:]
        latest_frame = history[:, -self.num_single_obs:]
        estimated_state, dynamic_latent = self.estimator(history)
        actor_input = torch.cat(
            (latest_frame, estimated_state, dynamic_latent), dim=-1
        )
        return self.actor(actor_input)

    def export(self, path, name):
        output_path = os.path.join(path, f"{name}.pt")
        self.to("cpu")
        self.eval()
        dummy = torch.zeros(1, self.num_history_obs, dtype=torch.float32)
        torch.jit.trace(self, dummy).save(output_path)

    def export_onnx(self, path, name):
        output_path = os.path.join(path, f"{name}.onnx")
        self.to("cpu")
        self.eval()
        dummy = torch.zeros(1, self.num_history_obs, dtype=torch.float32)
        torch.onnx.export(
            self,
            dummy,
            output_path,
            input_names=["obs"],
            output_names=["actions"],
            opset_version=14,
            do_constant_folding=True,
        )
