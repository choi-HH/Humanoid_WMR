import torch
import torch.nn as nn # 신경망 모듈, 레이어, 활성화 함수 등 가져옴
from .utils import create_MLP

# Estimator network
class WMR_Estimator(nn.Module):
    def __init__(self,
                 num_input_dim,  # obs history 전체 차원
                 num_obs_dim,    # 현재 obs 차원
                 num_output_dim, # latent vector 차원
                 num_privileged_dim, # privileged obs 차원
                 hidden_dims=[256, 256],
                 activation="elu",
                 **kwargs): # 추가 인자는 받지 않음
        super(WMR_Estimator, self).__init__()

        self.num_input_dim = num_input_dim
        self.num_obs_dim = num_obs_dim
        self.num_output_dim = num_output_dim

        # History 길이 계산
        self.history_len = num_input_dim // num_obs_dim
        self.encoder_hidden_dim = 256 # Encoder hidden

        # ===== Encoder network =====
        self.encoder = nn.LSTM(input_size=num_obs_dim,
                               hidden_size=self.encoder_hidden_dim,
                               batch_first=True)

        # ===== Continuous Decoder network =====
        """
            입력: Latent Vector
            출력: [Obs 복원 () + Lin Vel 추정(3)]
        """
        self.decoder_output_dim = num_obs_dim + num_privileged_dim # obs 복원 + lin vel 추정
        self.decoder = create_MLP(num_inputs=num_output_dim,   
                                  num_outputs=self.decoder_output_dim,
                                  hidden_dims=hidden_dims,
                                  activation=activation)

    # estimator 네트워크의 순전파
    def forward(self, obs_history_flat):
        """
        Args:
            obs_history_flat: [Batch, num_input_dim] (Flattened History)
        Returns:
            latent: [Batch, num_output_dim]
            decoder_out: [Batch, num_obs_dim + 3]
        """
        batch_size = obs_history_flat.shape[0]

        # (1) Reshape: [Batch, Flattened] -> [Batch, Seq_Len, Obs_Dim]
        # 예: [128, 960] -> [128, 10, 96]
        obs_sequence = obs_history_flat.view(batch_size, self.history_len, self.num_obs_dim)

        # (2) Encoder (LSTM)
        """
            self.lstm returns: output(매 순간의 출력), h_n(단기 기억), c_n(장기 기억)
            우리는 마지막 타임스텝의 Hidden State(h_n)만 필요함
        """
        _, (h_n, _) = self.encoder(obs_sequence)

        # h_n shape: [num_layers, Batch, Hidden] -> squeeze -> [Batch, Hidden]
        latent = h_n.squeeze(0) 

        # (4) Decoder (MLP)
        decoder_out = self.decoder(latent)

        return latent, decoder_out

    # estimator network 호출
    def encode(self, input):
        self.latent, self.decoder_out = self.forward(input)
        return self.latent
    
    def get_encoder_out(self):
        return self.latent

    def get_decoder_out(self):
        return self.decoder_out