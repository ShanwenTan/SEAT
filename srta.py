from inc.test import *
from peft import PeftModel
import os.path as osp
import math
import torch
from typing import Optional

from safety_risk_utils import (
    compute_step_risk,
    update_sequence_risk,
    compute_safe_scores,
)


class SearchNode:
    def __init__(self, sess: Sess, reward: float = -np.inf, risk_R: float = 0.0):
        self.sess = sess
        self.reward = reward
        self.risk_R = risk_R

    def add(self, *tokens: int, reward=-np.inf, risk_R: Optional[float] = None) -> Self:
        if risk_R is None:
            risk_R = self.risk_R
        return type(self)(sess=self.sess.add(*tokens), reward=reward, risk_R=risk_R)


class Gen(AidedGen):  # Safe Multifurcation (q is the MRM)
    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument('--budget', type=int, help='number of searched tokens = budget * max_new_tokens',
                            required=True)
        parser.add_argument('--top_p', type=float, default=0.8, help='top_p for children pruning; cannot be disabled')

        parser.add_argument('--q_unseen_mask', type=str,
                            default='/data/saffron/models/Saffron-1-1B/unseen_mask.pt',
                            help='path to the mask of unseen tokens during Q model training')
        parser.add_argument('--q_base', type=str,
                            default='/data/saffron/local_model_cache/LLM-Research/Llama-Guard-3-1B',
                            help='local path to the base of the Q model')
        parser.add_argument('--q_peft', type=str,
                            default='/data/saffron/models/Saffron-1-1B',
                            help='local path to the PEFT folder of the Q model')
        parser.add_argument('--q_batch_size', type=int, default=4, help='batch size of the Q model')
        parser.add_argument('--min_new_tokens', type=int, default=16,
                            help='minimum number of searched tokens before EOS')

        parser.add_argument('--max_retries', type=int, default=2, help='max number of safety retries')
        parser.add_argument('--inf', type=float, default=1024., help='substitute for inf in the rewards')

        parser.add_argument(
            '--safety_threshold',
            type=float,
            default=0.75,
            help='sequence-level safety threshold for pruning (on safety score S_t^{(j)})',
        )

    def __init__(self, args):
        copy_attrs(self, args, 'budget', 'top_p', 'min_new_tokens', 'inf', 'q_batch_size', 'count_flops')
        self.max_retries = args.max_retries
        super().__init__(
            name=f'{osp.splitext(osp.split(__file__)[1])[0]}{self.budget}tp{self.top_p}{args.q_peft.split("~")[-1]}',
            args=args,
        )


        self.alpha_fit = 38
        self.beta_fit = 5
        self.gamma_fit = 1
        self.delta_fit = 4


        self.safety_threshold = args.safety_threshold 

        self.risk_kappa = 0.8     
        self.risk_lambda_c = 0   
        self.risk_C0 = 0.90        

        self.risk_rho = 0.8        
        self.risk_gamma_abs = 0.7  
        self.risk_gamma_rel = 0.3  
        self.risk_w_abs = 0.8
        self.risk_w_rel = 1 - self.risk_w_abs


        q_tokenizer = AutoTokenizer.from_pretrained(
            args.q_base,
            padding_side='left',
            token=None  #
        )

        q_model = AutoModelForCausalLM.from_pretrained(
            args.q_base,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        )
        q_model.eval()

        q_model = PeftModel.from_pretrained(
            model=q_model,
            model_id=args.q_peft
        )

        with torch.inference_mode():
            weight = q_model.lm_head.weight
            device = weight.device

            q_model.lm_head = torch.nn.Linear(
                in_features=weight.shape[1],
                out_features=weight.shape[0],
                bias=True,
                dtype=weight.dtype
            ).to(device)
            q_model.lm_head.weight.data = weight

            q_model.lm_head.bias.data = torch.load(
                osp.join(args.q_peft, 'lm_head_bias.pt'),
                map_location=device,
                weights_only=True
            ).to(q_model.lm_head.bias.dtype)

            self.unseen_mask = torch.load(
                args.q_unseen_mask,
                map_location=device,
                weights_only=True
            )

        q_model.eval()

        self.q_id = self.add_aid(
            batch_size=self.q_batch_size,
            cache=True,
            profile=False,
            hf_model=q_model,
            hf_tokenizer=q_tokenizer,
            prefix=[128000, 271, 1502, 25],
            infix=[271, 17230, 25],
        )

    def get_safety_threshold(self, N: int) -> float:
        return self.safety_threshold

    def _run_search_once(
        self,
        policy_model: Model,
        start_sess: Sess,
        max_new_tokens: int,
        base_sess_for_safe: Sess,
        safe_response_tokens: List[int],
    ) -> Tuple[Sess, List[List[SearchNode]], bool]:

        nodes: List[SearchNode] = [SearchNode(sess=start_sess)]
        verbose: List[List[SearchNode]] = [nodes]

        for cur_new_tokens in range(max_new_tokens):
            next_nodes: List[SearchNode] = []
            cand_indices: List[int] = []
            sess_list: List[Sess] = []


            for idx, node in enumerate(nodes):
                if policy_model.is_eos(node.sess.last_output_token):
                    next_nodes.append(node)
                else:
                    cand_indices.append(idx)
                    sess_list.append(node.sess)

            if sess_list:
                # 2. policy logits -> probs
                cand_policies = policy_model(sess_list)
                if cur_new_tokens < self.min_new_tokens:
                    cand_policies[:, policy_model.tokenizer.eos_token_id] = -np.inf

                cand_policies, cand_tokens = cand_policies.sort(dim=-1, descending=True)
                cand_policies = cand_policies.softmax(dim=-1)
                cand_mask = (cand_policies.cumsum(dim=-1) - cand_policies < self.top_p)


                cand_indices_tensor = torch.as_tensor(
                    cand_indices,
                    dtype=torch.long,
                    device=cand_tokens.device
                )[:, None]
                cand_indices_expanded = cand_indices_tensor.expand(-1, cand_tokens.shape[-1])


                cand_rewards = self.call_aid(
                    aid_id=self.q_id,
                    sess_list=sess_list,
                    src_model=policy_model
                ).to(cand_tokens.device)
                cand_rewards[:, self.unseen_mask] = -self.inf
                cand_rewards = cand_rewards.gather(dim=-1, index=cand_tokens)


                if cand_policies.numel() > 0:

                    prev_R_list = [nodes[idx].risk_R for idx in cand_indices]
                    prev_R = torch.as_tensor(
                        prev_R_list,
                        device=cand_policies.device,
                        dtype=cand_policies.dtype,
                    )


                    step_risk = compute_step_risk(
                        probs=cand_policies,
                        reward_logits=cand_rewards,
                        mask=cand_mask,
                        kappa=self.risk_kappa,
                        lambda_c=self.risk_lambda_c,
                        C0=self.risk_C0,
                    )


                    new_R = update_sequence_risk(
                        prev_R=prev_R,
                        step_risk=step_risk,
                        rho=self.risk_rho,
                    )

     
                    safe_scores = compute_safe_scores(
                        R=new_R,
                        gamma_abs=self.risk_gamma_abs,
                        gamma_rel=self.risk_gamma_rel,
                        w_abs=self.risk_w_abs,
                        w_rel=self.risk_w_rel,
                    )


                    if torch.all(safe_scores < self.safety_threshold):
                        safe_node = SearchNode(
                            sess=base_sess_for_safe.add(*safe_response_tokens),
                            risk_R=0.0,
                        )
                        print(
                            f'Trigger the safety pruning condition '
                            f'(safety_threshold={self.safety_threshold:.4f}), '
                            f'and return a safety prompt'
                        )
                        return safe_node.sess, verbose, True


                    new_R_per_index = {
                        idx: R_val.item()
                        for idx, R_val in zip(cand_indices, new_R)
                    }
                else:
                    new_R_per_index = {}


                filtered_indices = cand_indices_expanded[cand_mask]
                filtered_tokens = cand_tokens[cand_mask]
                filtered_rewards = cand_rewards[cand_mask]

                if filtered_rewards.numel() > 0:
                    top_k = min(self.budget, filtered_rewards.shape[-1])
                    top_rewards, top_indices = filtered_rewards.topk(
                        dim=-1, k=top_k, largest=True, sorted=False
                    )
                    top_cand_indices = filtered_indices[top_indices]
                    top_cand_tokens = filtered_tokens[top_indices]

                    for idx, token, reward in zip(
                        top_cand_indices.tolist(),
                        top_cand_tokens.tolist(),
                        top_rewards
                    ):
                        node = nodes[idx]

                        risk_R = new_R_per_index.get(idx, node.risk_R)
                        next_nodes.append(
                            node.add(
                                token,
                                reward=reward if policy_model.is_eos(token) else node.reward,
                                risk_R=risk_R,
                            )
                        )

                    next_nodes.sort(key=lambda node: node.reward, reverse=True)
                    del next_nodes[self.budget:]


            nodes = next_nodes
            verbose.append(nodes)

        output_rewards = [(node.sess, node.reward) for node in nodes]
        best_output_reward = max(output_rewards, key=lambda x: x[1])
        best_session = best_output_reward[0]
        return best_session, verbose, False

    @torch.inference_mode()
    def __call__(self, policy_model: Model, reward_model: Model, sess: Sess, max_new_tokens: int) -> Tuple[Sess, Any]:
        """
        Given an initial session, a policy model, a reward model, and a maximum number of new tokens,
        generate a sequence of sessions with the highest reward.
        """
        base_sess = sess  

        check_text = (
            "Waiting to check output safety, safety guidelines: reject any harmful requests. "
            "If the previous output contains harmful content, directly refuse to answer the question "
            "without repeating any previous content on the question."
        )
        check_tokens = policy_model.tokenizer.encode(check_text, add_special_tokens=False)

        second_check_text = (
            "Final security check: Any harmful output will be blocked. Only safe responses will be "
            "generated according to the security policy (i.e. directly refuse to generate harmful "
            "content without reproducing its content)"
        )
        second_check_tokens = policy_model.tokenizer.encode(second_check_text, add_special_tokens=False)

        safe_response_text = 'Due to security policy, I cannot answer this question'
        safe_response_tokens = policy_model.tokenizer.encode(safe_response_text, add_special_tokens=False)


        best_session, verbose, pruned = self._run_search_once(
            policy_model=policy_model,
            start_sess=sess,
            max_new_tokens=max_new_tokens,
            base_sess_for_safe=base_sess,
            safe_response_tokens=safe_response_tokens,
        )

        if pruned:

            return best_session, verbose

        eval_reward = reward_model(sess_list=[best_session], src_model=policy_model).item()


        if eval_reward >= -10000:
            print("Satisfy security requirements and return directly. The current eval_reward is:", eval_reward)
            return best_session, verbose


        print(
            "The security requirements are not met. Start safety retries. "
            "Initial eval_reward is:", eval_reward
        )

        verbose_all = verbose
        last_best_session = best_session


        check_token_sequences = [check_tokens, second_check_tokens]

        for retry_idx in range(self.max_retries):

            if retry_idx < len(check_token_sequences):
                extra_tokens = check_token_sequences[retry_idx]
            else:

                extra_tokens = check_token_sequences[-1]

            print(f"Performing safety retry {retry_idx + 1} ...")


            current_node = SearchNode(sess=last_best_session)
            for token in extra_tokens:
                current_node = current_node.add(token, reward=current_node.reward)
            new_sess = current_node.sess


            best_session_retry, verbose_retry, pruned = self._run_search_once(
                policy_model=policy_model,
                start_sess=new_sess,
                max_new_tokens=max_new_tokens,
                base_sess_for_safe=base_sess,
                safe_response_tokens=safe_response_tokens,
            )

            verbose_all = verbose_all + verbose_retry

            if pruned:

                return best_session_retry, verbose_all

            eval_reward_retry = reward_model(
                sess_list=[best_session_retry],
                src_model=policy_model
            ).item()

            print(f'The eval_reward after retry {retry_idx + 1} is', eval_reward_retry)

            if eval_reward_retry >= 0:
                print(f"The safety requirements are satisfied after retry {retry_idx + 1}, returning result.")
                return best_session_retry, verbose_all


            last_best_session = best_session_retry


        print("All safety retries finished. Return the last retry result.")
        return last_best_session, verbose_all


if __name__ == '__main__':
    test(Gen)

