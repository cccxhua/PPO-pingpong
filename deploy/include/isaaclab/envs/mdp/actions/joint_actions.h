// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/manager/action_manager.h"

namespace isaaclab
{

class JointAction : public ActionTerm
{
public:
    JointAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :ActionTerm(cfg, env)
    {
        if(cfg["joint_ids"].IsNull()) {
            _action_dim = env->robot->data.joint_ids_map.size();
        } else {
            _joint_ids = cfg["joint_ids"].as<std::vector<int>>();
            _action_dim = _joint_ids.size();
        }
        _raw_actions.resize(_action_dim, 0.0f);
        _processed_actions.resize(_action_dim, 0.0f);
        if(!cfg["scale"].IsNull()) {
            _scale = cfg["scale"].as<std::vector<float>>();
        }
        if(!cfg["offset"].IsNull()) {
            _offset = cfg["offset"].as<std::vector<float>>();
        }
        if(!cfg["clip"].IsNull()) {
            _clip = cfg["clip"].as<std::vector<std::vector<float> >>();
        }
    }

    virtual void process_actions(std::vector<float> actions)
    {
        // TODO: modify action by joint_ids
        _raw_actions = actions;
        for(int i(0); i<_action_dim; ++i)
        {
            if(!_scale.empty()) {
                _processed_actions[i] = _raw_actions[i] * _scale[i];
            } else {
                _processed_actions[i] = _raw_actions[i];
            }
            if(!_offset.empty()) {
                _processed_actions[i] += _offset[i];
            }
        }
        if(!_clip.empty())
        {
            for(int i(0); i<_action_dim; ++i) {
                _processed_actions[i] = std::clamp(_processed_actions[i], _clip[i][0], _clip[i][1]);
            }
        }
    }


    int action_dim() 
    {
        return _action_dim;
    }

    std::vector<float> raw_actions() 
    {
        return _raw_actions;
    }
    
    std::vector<float> processed_actions() 
    {
        return _processed_actions;
    }

    void reset()
    {
        _raw_actions.assign(_action_dim, 0.0f);
    }

protected:
    int _action_dim;
    std::vector<int> _joint_ids;

    std::vector<float> _raw_actions;
    std::vector<float> _processed_actions;

    std::vector<float> _scale;
    std::vector<float> _offset;
    std::vector<std::vector<float> > _clip;
};


class JointPositionAction : public JointAction
{
public:
    JointPositionAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :JointAction(cfg, env)
    {
    }
};

class JointVelocityAction : public JointAction
{
public:
    JointVelocityAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :JointAction(cfg, env)
    {
    }
};


// ReferenceResidualJointAction
//
// 与 sim 端 source/.../tasks/table_tennis/mdp/actions.py 中
// ReferenceResidualJointAction.process_actions 一一对应:
//
//     effective  = action_delay_buffer[delay_step]            (delay_max>0 时, 否则 = actions)
//     residual   = alpha * prev_residual + (1-alpha) * effective    (EMA, sim2real 关键)
//     target     = ref_dof(phase) + residual * residual_scale
//
// 部署端必须自己提供 ref_dof 来源(motion command 或参考 npz 当前帧 lookup).
// 默认 _lookup_ref_dof() 返回零向量, 子类或本类内部应改成项目实际接口.
class ReferenceResidualJointAction : public ActionTerm
{
public:
    ReferenceResidualJointAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    : ActionTerm(cfg, env)
    {
        // joint_ids
        if(cfg["joint_ids"].IsNull()) {
            _action_dim = env->robot->data.joint_ids_map.size();
        } else {
            _joint_ids = cfg["joint_ids"].as<std::vector<int>>();
            _action_dim = _joint_ids.size();
        }

        // residual_scale (scalar 或 per-joint vector)
        if(!cfg["residual_scale"].IsNull()) {
            if(cfg["residual_scale"].IsScalar()) {
                float s = cfg["residual_scale"].as<float>();
                _scale.assign(_action_dim, s);
            } else {
                _scale = cfg["residual_scale"].as<std::vector<float>>();
            }
        } else {
            _scale.assign(_action_dim, 1.0f);
        }

        // command_name: 用于在 _lookup_ref_dof 里向 motion command 索要 ref_dof
        if(cfg["command_name"]) {
            _command_name = cfg["command_name"].as<std::string>();
        }

        // EMA smoothing on residual (alpha=0 表示无滤波)
        if(cfg["action_smoothing_alpha"]) {
            _smoothing_alpha = cfg["action_smoothing_alpha"].as<float>();
        }

        // action delay buffer (训练时 DR_STAGE>=2 才开; 部署用确定性 _min_delay)
        if(cfg["action_delay_steps_min"]) {
            _min_delay = cfg["action_delay_steps_min"].as<int>();
        }
        if(cfg["action_delay_steps_max"]) {
            _max_delay = cfg["action_delay_steps_max"].as<int>();
        }

        // 状态缓冲
        _raw_actions.assign(_action_dim, 0.0f);
        _processed_actions.assign(_action_dim, 0.0f);
        _prev_residual.assign(_action_dim, 0.0f);
        if(_max_delay > 0) {
            _action_buffer.assign(_max_delay + 1, std::vector<float>(_action_dim, 0.0f));
        }
        _delay_step = _min_delay;
    }

    int action_dim() override { return _action_dim; }
    std::vector<float> raw_actions() override { return _raw_actions; }
    std::vector<float> processed_actions() override { return _processed_actions; }

    void process_actions(std::vector<float> actions) override
    {
        _raw_actions = actions;

        // (1) action delay: 滚动 buffer, 取出 _delay_step 处的旧 action
        std::vector<float> effective;
        if(_max_delay > 0) {
            for(int i = static_cast<int>(_action_buffer.size()) - 1; i > 0; --i) {
                _action_buffer[i] = _action_buffer[i - 1];
            }
            _action_buffer[0] = actions;
            effective = _action_buffer[_delay_step];
        } else {
            effective = actions;
        }

        // (2) EMA 一阶低通: smoothed = alpha * prev + (1 - alpha) * effective
        // alpha=0 时直接 pass-through; 训练默认 0.7, 必须从 deploy.yaml 读, 不写死.
        std::vector<float> residual(_action_dim);
        if(_smoothing_alpha > 0.0f) {
            for(int i = 0; i < _action_dim; ++i) {
                residual[i] = _smoothing_alpha * _prev_residual[i]
                            + (1.0f - _smoothing_alpha) * effective[i];
            }
            _prev_residual = residual;
        } else {
            residual = effective;
        }

        // (3) target = ref_dof(phase) + residual * scale
        std::vector<float> ref = _lookup_ref_dof();
        for(int i = 0; i < _action_dim; ++i) {
            _processed_actions[i] = ref[i] + residual[i] * _scale[i];
        }
    }

    void reset() override
    {
        _raw_actions.assign(_action_dim, 0.0f);
        std::fill(_prev_residual.begin(), _prev_residual.end(), 0.0f);
        if(_max_delay > 0) {
            for(auto & buf : _action_buffer) {
                std::fill(buf.begin(), buf.end(), 0.0f);
            }
        }
    }

protected:
    // !! 部署侧必须提供当前 phase 对应的 ref_dof, 长度 == _action_dim, 顺序与 _scale 一致.
    // 在你 deploy 分支已有的 motion command / 参考 npz lookup 接口上接一行返回即可.
    // 例: return env->command_manager->get_term(_command_name)->ref_dof();
    virtual std::vector<float> _lookup_ref_dof()
    {
        return std::vector<float>(_action_dim, 0.0f);
    }

    int _action_dim;
    std::vector<int> _joint_ids;
    std::string _command_name;

    std::vector<float> _scale;
    std::vector<float> _raw_actions;
    std::vector<float> _processed_actions;

    // EMA 状态
    float _smoothing_alpha = 0.0f;
    std::vector<float> _prev_residual;

    // 动作延迟缓冲
    int _min_delay = 0;
    int _max_delay = 0;
    int _delay_step = 0;
    std::vector<std::vector<float>> _action_buffer;
};


REGISTER_ACTION(JointPositionAction);
REGISTER_ACTION(JointVelocityAction);
REGISTER_ACTION(ReferenceResidualJointAction);

};