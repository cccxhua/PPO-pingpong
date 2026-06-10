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


// forehand_middle_a1_whip.npz: 105 frames x 7 joints, fps=30, duration=3.5s
#include "isaaclab/envs/mdp/actions/a1_forehand_middle_ref.h"

// ReferenceResidualJointAction
//
// 与 sim 端 ReferenceResidualJointAction.process_actions 一一对应:
//   policy output = [residual(7D), phase_speed(1D)]
//   effective  = action_delay_buffer[delay_step]
//   residual   = alpha * prev_residual + (1-alpha) * effective[0:7]
//   phase_speed = speed_min + (clamp(effective[7],-1,1)+1)*0.5*(speed_max-speed_min)
//   phase     += step_dt / duration * phase_speed
//   target     = ref_dof(phase) + residual * residual_scale
class ReferenceResidualJointAction : public ActionTerm
{
public:
    ReferenceResidualJointAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    : ActionTerm(cfg, env)
    {
        // joint_ids: 残差作用的关节数 (7)
        if(cfg["joint_ids"].IsNull()) {
            _joint_dim = env->robot->data.joint_ids_map.size();
        } else {
            _joint_ids = cfg["joint_ids"].as<std::vector<int>>();
            _joint_dim = _joint_ids.size();
        }

        // residual_scale
        if(!cfg["residual_scale"].IsNull()) {
            if(cfg["residual_scale"].IsScalar()) {
                float s = cfg["residual_scale"].as<float>();
                _scale.assign(_joint_dim, s);
            } else {
                _scale = cfg["residual_scale"].as<std::vector<float>>();
            }
        } else {
            _scale.assign(_joint_dim, 1.0f);
        }

        // phase speed range
        if(cfg["speed_min"]) _speed_min = cfg["speed_min"].as<float>();
        if(cfg["speed_max"]) _speed_max = cfg["speed_max"].as<float>();

        // EMA smoothing
        if(cfg["action_smoothing_alpha"]) {
            _smoothing_alpha = cfg["action_smoothing_alpha"].as<float>();
        }

        // action delay buffer
        if(cfg["action_delay_steps_min"]) {
            _min_delay = cfg["action_delay_steps_min"].as<int>();
        }
        if(cfg["action_delay_steps_max"]) {
            _max_delay = cfg["action_delay_steps_max"].as<int>();
        }

        // 总输入维度 = 7(residual) + 1(phase_speed)
        _input_dim = _joint_dim + 1;

        // 状态缓冲
        _raw_actions.assign(_input_dim, 0.0f);
        _processed_actions.assign(_joint_dim, 0.0f);
        _prev_residual.assign(_joint_dim, 0.0f);
        if(_max_delay > 0) {
            _action_buffer.assign(_max_delay + 1, std::vector<float>(_input_dim, 0.0f));
        }
        _delay_step = _min_delay;
        _phase = 0.0f;
    }

    // 策略输出 8D: 7(residual) + 1(phase_speed)
    int action_dim() override { return _input_dim; }
    std::vector<float> raw_actions() override { return _raw_actions; }
    // 返回 7D 关节位置目标
    std::vector<float> processed_actions() override { return _processed_actions; }

    void process_actions(std::vector<float> actions) override
    {
        _raw_actions = actions;

        // (1) action delay
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

        // (2) 提取 phase_speed (第 8 维), 推进 phase
        float raw_speed = std::clamp(effective[_joint_dim], -1.0f, 1.0f);
        float phase_speed;
        if(raw_speed >= 0.0f) {
            phase_speed = 1.0f + raw_speed * (_speed_max - 1.0f);
        } else {
            phase_speed = 1.0f + raw_speed * (1.0f - _speed_min);
        }
        _phase += env->step_dt / REF_DURATION * phase_speed;
        if(_phase >= 1.0f) _phase = std::fmod(_phase, 1.0f);
        if(_phase < 0.0f) _phase = 0.0f;

        // 同步到 env->global_phase 供 observation 使用
        env->global_phase = _phase;

        // (3) EMA on residual (前 7 维)
        std::vector<float> residual(_joint_dim);
        if(_smoothing_alpha > 0.0f) {
            for(int i = 0; i < _joint_dim; ++i) {
                residual[i] = _smoothing_alpha * _prev_residual[i]
                            + (1.0f - _smoothing_alpha) * effective[i];
            }
            _prev_residual = residual;
        } else {
            for(int i = 0; i < _joint_dim; ++i) {
                residual[i] = effective[i];
            }
        }

        // (4) target = ref_dof(phase) + residual * scale
        std::vector<float> ref = _lookup_ref_dof();
        for(int i = 0; i < _joint_dim; ++i) {
            _processed_actions[i] = ref[i] + residual[i] * _scale[i];
        }
    }

    void reset() override
    {
        _raw_actions.assign(_input_dim, 0.0f);
        _processed_actions.assign(_joint_dim, 0.0f);
        std::fill(_prev_residual.begin(), _prev_residual.end(), 0.0f);
        if(_max_delay > 0) {
            for(auto & buf : _action_buffer) {
                std::fill(buf.begin(), buf.end(), 0.0f);
            }
        }
        _phase = 0.0f;
    }

    float phase() const { return _phase; }
    void set_phase(float p) { _phase = p; }

protected:
    std::vector<float> _lookup_ref_dof()
    {
        std::vector<float> ref(_joint_dim);
        float frame_f = _phase * (REF_NUM_FRAMES - 1);
        int lo = std::clamp(static_cast<int>(frame_f), 0, REF_NUM_FRAMES - 2);
        int hi = lo + 1;
        float alpha = frame_f - lo;
        for(int j = 0; j < _joint_dim; ++j) {
            ref[j] = (1.0f - alpha) * A1_FOREHAND_MIDDLE_REF[lo][j]
                   + alpha          * A1_FOREHAND_MIDDLE_REF[hi][j];
        }
        return ref;
    }

    int _joint_dim;
    int _input_dim;
    std::vector<int> _joint_ids;

    std::vector<float> _scale;
    std::vector<float> _raw_actions;
    std::vector<float> _processed_actions;

    // phase 状态
    float _phase = 0.0f;
    float _speed_min = 0.85f;
    float _speed_max = 1.15f;

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