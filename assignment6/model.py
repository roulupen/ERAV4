from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class MNISTNet4BlockWithBatchNormMaxPoolConvFinal(nn.Module):
    def __init__(self, dropout_prob=0.05):
        super(MNISTNet4BlockWithBatchNormMaxPoolConvFinal, self).__init__()

        # ------- Block 1 (Sequential) -------
        # Input: (B, 1, 28, 28) | Output: (B, 8, 14, 14) | Receptive Field: 5x5
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),      # (B, 1, 28, 28) -> (B, 8, 28, 28) | RF: 3x3
            nn.BatchNorm2d(8),                  # (B, 8, 28, 28) -> (B, 8, 28, 28) | RF: 3x3
            nn.ReLU(inplace=True),               # (B, 8, 28, 28) -> (B, 8, 28, 28) | RF: 3x3

            nn.Conv2d(8, 16, 3, padding=1),     # (B, 8, 28, 28) -> (B, 16, 28, 28) | RF: 5x5
            nn.BatchNorm2d(16),                  # (B, 16, 28, 28) -> (B, 16, 28, 28) | RF: 5x5
            nn.ReLU(inplace=True),               # (B, 16, 28, 28) -> (B, 16, 28, 28) | RF: 5x5
            
            # Transition layer
            nn.Conv2d(16, 8, 1),                # (B, 16, 28, 28) -> (B, 8, 28, 28) | RF: 5x5
            nn.BatchNorm2d(8),                   # (B, 8, 28, 28) -> (B, 8, 28, 28) | RF: 5x5
            nn.ReLU(inplace=True),               # (B, 8, 28, 28) -> (B, 8, 28, 28) | RF: 5x5
            nn.MaxPool2d(2)                      # (B, 8, 28, 28) -> (B, 8, 14, 14) | RF: 5x5
        )

        # ------- Block 2 (Sequential) -------
        # Input: (B, 8, 14, 14) | Output: (B, 6, 7, 7) | Receptive Field: 13x13
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 12, 3, padding=1),      # (B, 8, 14, 14) -> (B, 12, 14, 14) | RF: 9x9
            nn.BatchNorm2d(12),                  # (B, 12, 14, 14) -> (B, 12, 14, 14) | RF: 9x9
            nn.ReLU(inplace=True),               # (B, 12, 14, 14) -> (B, 12, 14, 14) | RF: 9x9
            
            nn.Conv2d(12, 16, 3, padding=1),     # (B, 12, 14, 14) -> (B, 16, 14, 14) | RF: 13x13
            nn.BatchNorm2d(16),                  # (B, 16, 14, 14) -> (B, 16, 14, 14) | RF: 13x13
            nn.ReLU(inplace=True),               # (B, 16, 14, 14) -> (B, 16, 14, 14) | RF: 13x13
            
            # Transition layer
            nn.Conv2d(16, 6, 1),                # (B, 16, 14, 14) -> (B, 6, 14, 14) | RF: 13x13
            nn.BatchNorm2d(6),                   # (B, 6, 14, 14) -> (B, 6, 14, 14) | RF: 13x13
            nn.ReLU(inplace=True),               # (B, 6, 14, 14) -> (B, 6, 14, 14) | RF: 13x13
            nn.MaxPool2d(2)                      # (B, 6, 14, 14) -> (B, 6, 7, 7) | RF: 13x13
        )

        # ------- Block 3 (Sequential) -------
        # Input: (B, 6, 7, 7) | Output: (B, 6, 3, 3) | Receptive Field: 29x29
        self.block3 = nn.Sequential(
            nn.Conv2d(6, 12, 3, padding=1),      # (B, 6, 7, 7) -> (B, 12, 7, 7) | RF: 17x17
            nn.BatchNorm2d(12),                  # (B, 12, 7, 7) -> (B, 12, 7, 7) | RF: 17x17
            nn.ReLU(inplace=True),               # (B, 12, 7, 7) -> (B, 12, 7, 7) | RF: 17x17

            nn.Conv2d(12, 12, 3, padding=1),     # (B, 12, 7, 7) -> (B, 12, 7, 7) | RF: 25x25
            nn.BatchNorm2d(12),                  # (B, 12, 7, 7) -> (B, 12, 7, 7) | RF: 25x25
            nn.ReLU(inplace=True),               # (B, 12, 7, 7) -> (B, 12, 7, 7) | RF: 25x25

            # Transition layer
            nn.Conv2d(12, 6, 1),                # (B, 12, 7, 7) -> (B, 6, 7, 7) | RF: 25x25
            nn.BatchNorm2d(6),                   # (B, 6, 7, 7) -> (B, 6, 7, 7) | RF: 25x25
            nn.ReLU(inplace=True),               # (B, 6, 7, 7) -> (B, 6, 7, 7) | RF: 25x25
            nn.MaxPool2d(2)                      # (B, 6, 7, 7) -> (B, 6, 3, 3) | RF: 29x29
        )

        # ------- Block 4 (Sequential) -------
        # Input: (B, 6, 3, 3) | Output: (B, 10, 3, 3) | Receptive Field: 45x45
        self.block4 = nn.Sequential(
            nn.Conv2d(6, 10, 3, padding=1),      # (B, 6, 3, 3) -> (B, 10, 3, 3) | RF: 33x33
            nn.BatchNorm2d(10),                  # (B, 10, 3, 3) -> (B, 10, 3, 3) | RF: 33x33
            nn.ReLU(inplace=True),               # (B, 10, 3, 3) -> (B, 10, 3, 3) | RF: 33x33

            nn.Conv2d(10, 10, 3, padding=1),     # (B, 10, 3, 3) -> (B, 10, 3, 3) | RF: 45x45
            nn.BatchNorm2d(10),                  # (B, 10, 3, 3) -> (B, 10, 3, 3) | RF: 45x45
            nn.ReLU(inplace=True)                # (B, 10, 3, 3) -> (B, 10, 3, 3) | RF: 45x45
        )

        # ------- Final Conv Block (instead of GAP) -------
        # Input: (B, 10, 3, 3) | Output: (B, 10, 1, 1) | Receptive Field: 45x45
        self.final_conv = nn.Sequential(
            nn.Conv2d(10, 10, 3, padding=0),     # (B, 10, 3, 3) -> (B, 10, 1, 1) | RF: 45x45
            nn.BatchNorm2d(10),                  # (B, 10, 1, 1) -> (B, 10, 1, 1) | RF: 45x45
            nn.ReLU(inplace=True)                # (B, 10, 1, 1) -> (B, 10, 1, 1) | RF: 45x45
        )

        # ------- Final Classifier (Sequential) -------
        self.classifier = nn.Sequential(
            nn.Flatten(),                        # (B, 10, 1, 1) -> (B, 10)
            nn.Linear(10, 10)                    # (B, 10) -> (B, 10)
        )

    def forward(self, x):
        # Process through all sequential blocks with detailed shape tracking
        x = self.block1(x)  # (B, 1, 28, 28) -> (B, 8, 14, 14) | RF: 5x5
        x = self.block2(x)  # (B, 8, 14, 14) -> (B, 6, 7, 7) | RF: 13x13
        x = self.block3(x)  # (B, 6, 7, 7) -> (B, 6, 3, 3) | RF: 29x29
        x = self.block4(x)  # (B, 6, 3, 3) -> (B, 10, 3, 3) | RF: 45x45
        
        # Final Conv Block (instead of GAP)
        x = self.final_conv(x)  # (B, 10, 3, 3) -> (B, 10, 1, 1) | RF: 45x45
        
        # Final classification
        x = self.classifier(x)  # (B, 10, 1, 1) -> (B, 10)
        
        return F.log_softmax(x, dim=1)  # (B, 10) -> (B, 10)


class MNISTNet4BlockWithBatchNormMaxPoolDropoutConvFinal(nn.Module):
    def __init__(self, dropout_prob=0.05):
        super(MNISTNet4BlockWithBatchNormMaxPoolDropoutConvFinal, self).__init__()

        # ------- Block 1 (Sequential) -------
        # Input: (B, 1, 28, 28) | Output: (B, 8, 14, 14) | Receptive Field: 5x5
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),      # (B, 1, 28, 28) -> (B, 8, 28, 28) | RF: 3x3
            nn.BatchNorm2d(8),                  # (B, 8, 28, 28) -> (B, 8, 28, 28) | RF: 3x3
            nn.ReLU(inplace=True),               # (B, 8, 28, 28) -> (B, 8, 28, 28) | RF: 3x3
            nn.Dropout(dropout_prob),

            nn.Conv2d(8, 16, 3, padding=1),     # (B, 8, 28, 28) -> (B, 16, 28, 28) | RF: 5x5
            nn.BatchNorm2d(16),                  # (B, 16, 28, 28) -> (B, 16, 28, 28) | RF: 5x5
            nn.ReLU(inplace=True),               # (B, 16, 28, 28) -> (B, 16, 28, 28) | RF: 5x5
            nn.Dropout(dropout_prob),
            
            # Transition layer
            nn.Conv2d(16, 8, 1),                # (B, 16, 28, 28) -> (B, 8, 28, 28) | RF: 5x5
            nn.BatchNorm2d(8),                   # (B, 8, 28, 28) -> (B, 8, 28, 28) | RF: 5x5
            nn.ReLU(inplace=True),               # (B, 8, 28, 28) -> (B, 8, 28, 28) | RF: 5x5
            nn.MaxPool2d(2)                      # (B, 8, 28, 28) -> (B, 8, 14, 14) | RF: 5x5
        )

        # ------- Block 2 (Sequential) -------
        # Input: (B, 8, 14, 14) | Output: (B, 6, 7, 7) | Receptive Field: 13x13
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 12, 3, padding=1),      # (B, 8, 14, 14) -> (B, 12, 14, 14) | RF: 9x9
            nn.BatchNorm2d(12),                  # (B, 12, 14, 14) -> (B, 12, 14, 14) | RF: 9x9
            nn.ReLU(inplace=True),               # (B, 12, 14, 14) -> (B, 12, 14, 14) | RF: 9x9
            nn.Dropout(dropout_prob),
            
            nn.Conv2d(12, 16, 3, padding=1),     # (B, 12, 14, 14) -> (B, 16, 14, 14) | RF: 13x13
            nn.BatchNorm2d(16),                  # (B, 16, 14, 14) -> (B, 16, 14, 14) | RF: 13x13
            nn.ReLU(inplace=True),               # (B, 16, 14, 14) -> (B, 16, 14, 14) | RF: 13x13
            nn.Dropout(dropout_prob),
            
            # Transition layer
            nn.Conv2d(16, 6, 1),                # (B, 16, 14, 14) -> (B, 6, 14, 14) | RF: 13x13
            nn.BatchNorm2d(6),                   # (B, 6, 14, 14) -> (B, 6, 14, 14) | RF: 13x13
            nn.ReLU(inplace=True),               # (B, 6, 14, 14) -> (B, 6, 14, 14) | RF: 13x13
            nn.MaxPool2d(2)                      # (B, 6, 14, 14) -> (B, 6, 7, 7) | RF: 13x13
        )

        # ------- Block 3 (Sequential) -------
        # Input: (B, 6, 7, 7) | Output: (B, 6, 3, 3) | Receptive Field: 29x29
        self.block3 = nn.Sequential(
            nn.Conv2d(6, 12, 3, padding=1),      # (B, 6, 7, 7) -> (B, 12, 7, 7) | RF: 17x17
            nn.BatchNorm2d(12),                  # (B, 12, 7, 7) -> (B, 12, 7, 7) | RF: 17x17
            nn.ReLU(inplace=True),               # (B, 12, 7, 7) -> (B, 12, 7, 7) | RF: 17x17
            #nn.Dropout(dropout_prob),

            nn.Conv2d(12, 12, 3, padding=1),     # (B, 12, 7, 7) -> (B, 12, 7, 7) | RF: 25x25
            nn.BatchNorm2d(12),                  # (B, 12, 7, 7) -> (B, 12, 7, 7) | RF: 25x25
            nn.ReLU(inplace=True),               # (B, 12, 7, 7) -> (B, 12, 7, 7) | RF: 25x25
            #nn.Dropout(dropout_prob),

            # Transition layer
            nn.Conv2d(12, 6, 1),                # (B, 12, 7, 7) -> (B, 6, 7, 7) | RF: 25x25
            nn.BatchNorm2d(6),                   # (B, 6, 7, 7) -> (B, 6, 7, 7) | RF: 25x25
            nn.ReLU(inplace=True),               # (B, 6, 7, 7) -> (B, 6, 7, 7) | RF: 25x25
            nn.MaxPool2d(2)                      # (B, 6, 7, 7) -> (B, 6, 3, 3) | RF: 29x29
        )

        # ------- Block 4 (Sequential) -------
        # Input: (B, 6, 3, 3) | Output: (B, 10, 3, 3) | Receptive Field: 45x45
        self.block4 = nn.Sequential(
            nn.Conv2d(6, 10, 3, padding=1),      # (B, 6, 3, 3) -> (B, 10, 3, 3) | RF: 33x33
            nn.BatchNorm2d(10),                  # (B, 10, 3, 3) -> (B, 10, 3, 3) | RF: 33x33
            nn.ReLU(inplace=True),               # (B, 10, 3, 3) -> (B, 10, 3, 3) | RF: 33x33
            #nn.Dropout(dropout_prob),

            nn.Conv2d(10, 10, 3, padding=1),     # (B, 10, 3, 3) -> (B, 10, 3, 3) | RF: 45x45
            nn.BatchNorm2d(10),                  # (B, 10, 3, 3) -> (B, 10, 3, 3) | RF: 45x45
            nn.ReLU(inplace=True)                # (B, 10, 3, 3) -> (B, 10, 3, 3) | RF: 45x45
        )

        # ------- Final Conv Block (instead of GAP) -------
        # Input: (B, 10, 3, 3) | Output: (B, 10, 1, 1) | Receptive Field: 45x45
        self.final_conv = nn.Sequential(
            nn.Conv2d(10, 10, 3, padding=0),     # (B, 10, 3, 3) -> (B, 10, 1, 1) | RF: 45x45
            nn.BatchNorm2d(10),                  # (B, 10, 1, 1) -> (B, 10, 1, 1) | RF: 45x45
            nn.ReLU(inplace=True)                # (B, 10, 1, 1) -> (B, 10, 1, 1) | RF: 45x45
        )

        # ------- Final Classifier (Sequential) -------
        self.classifier = nn.Sequential(
            nn.Flatten(),                        # (B, 10, 1, 1) -> (B, 10)
            nn.Linear(10, 10)                    # (B, 10) -> (B, 10)
        )

    def forward(self, x):
        # Process through all sequential blocks with detailed shape tracking
        x = self.block1(x)  # (B, 1, 28, 28) -> (B, 8, 14, 14) | RF: 5x5
        x = self.block2(x)  # (B, 8, 14, 14) -> (B, 6, 7, 7) | RF: 13x13
        x = self.block3(x)  # (B, 6, 7, 7) -> (B, 6, 3, 3) | RF: 29x29
        x = self.block4(x)  # (B, 6, 3, 3) -> (B, 10, 3, 3) | RF: 45x45
        
        # Final Conv Block (instead of GAP)
        x = self.final_conv(x)  # (B, 10, 3, 3) -> (B, 10, 1, 1) | RF: 45x45
        
        # Final classification
        x = self.classifier(x)  # (B, 10, 1, 1) -> (B, 10)
        
        return F.log_softmax(x, dim=1)  # (B, 10) -> (B, 10)


class MNISTNet4BlockWithBatchNormMaxPoolDropoutAveragePooling(nn.Module):
    def __init__(self, dropout_prob=0.05):
        super(MNISTNet4BlockWithBatchNormMaxPoolDropoutAveragePooling, self).__init__()

        # ------- Block 1 (Sequential) -------
        # Input: (B, 1, 28, 28) | Output: (B, 8, 14, 14) | Receptive Field: 5x5
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),      # (B, 1, 28, 28) -> (B, 8, 28, 28) | RF: 3x3
            nn.BatchNorm2d(8),                  # (B, 8, 28, 28) -> (B, 8, 28, 28) | RF: 3x3
            nn.ReLU(inplace=True),               # (B, 8, 28, 28) -> (B, 8, 28, 28) | RF: 3x3
            nn.Dropout(dropout_prob),

            nn.Conv2d(8, 16, 3, padding=1),     # (B, 8, 28, 28) -> (B, 16, 28, 28) | RF: 5x5
            nn.BatchNorm2d(16),                  # (B, 16, 28, 28) -> (B, 16, 28, 28) | RF: 5x5
            nn.ReLU(inplace=True),               # (B, 16, 28, 28) -> (B, 16, 28, 28) | RF: 5x5
            nn.Dropout(dropout_prob),
            
            # Transition layer
            nn.Conv2d(16, 8, 1),                # (B, 16, 28, 28) -> (B, 8, 28, 28) | RF: 5x5
            nn.BatchNorm2d(8),                   # (B, 8, 28, 28) -> (B, 8, 28, 28) | RF: 5x5
            nn.ReLU(inplace=True),               # (B, 8, 28, 28) -> (B, 8, 28, 28) | RF: 5x5
            nn.MaxPool2d(2)                      # (B, 8, 28, 28) -> (B, 8, 14, 14) | RF: 5x5
        )

        # ------- Block 2 (Sequential) -------
        # Input: (B, 8, 14, 14) | Output: (B, 6, 7, 7) | Receptive Field: 13x13
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 12, 3, padding=1),      # (B, 8, 14, 14) -> (B, 12, 14, 14) | RF: 9x9
            nn.BatchNorm2d(12),                  # (B, 12, 14, 14) -> (B, 12, 14, 14) | RF: 9x9
            nn.ReLU(inplace=True),               # (B, 12, 14, 14) -> (B, 12, 14, 14) | RF: 9x9
            nn.Dropout(dropout_prob),
            
            nn.Conv2d(12, 16, 3, padding=1),     # (B, 12, 14, 14) -> (B, 16, 14, 14) | RF: 13x13
            nn.BatchNorm2d(16),                  # (B, 16, 14, 14) -> (B, 16, 14, 14) | RF: 13x13
            nn.ReLU(inplace=True),               # (B, 16, 14, 14) -> (B, 16, 14, 14) | RF: 13x13
            nn.Dropout(dropout_prob),
            
            # Transition layer
            nn.Conv2d(16, 6, 1),                # (B, 16, 14, 14) -> (B, 6, 14, 14) | RF: 13x13
            nn.BatchNorm2d(6),                   # (B, 6, 14, 14) -> (B, 6, 14, 14) | RF: 13x13
            nn.ReLU(inplace=True),               # (B, 6, 14, 14) -> (B, 6, 14, 14) | RF: 13x13
            nn.MaxPool2d(2)                      # (B, 6, 14, 14) -> (B, 6, 7, 7) | RF: 13x13
        )

        # ------- Block 3 (Sequential) -------
        # Input: (B, 6, 7, 7) | Output: (B, 6, 3, 3) | Receptive Field: 29x29
        self.block3 = nn.Sequential(
            nn.Conv2d(6, 12, 3, padding=1),      # (B, 6, 7, 7) -> (B, 12, 7, 7) | RF: 17x17
            nn.BatchNorm2d(12),                  # (B, 12, 7, 7) -> (B, 12, 7, 7) | RF: 17x17
            nn.ReLU(inplace=True),               # (B, 12, 7, 7) -> (B, 12, 7, 7) | RF: 17x17
            #nn.Dropout(dropout_prob),

            nn.Conv2d(12, 12, 3, padding=1),     # (B, 12, 7, 7) -> (B, 12, 7, 7) | RF: 25x25
            nn.BatchNorm2d(12),                  # (B, 12, 7, 7) -> (B, 12, 7, 7) | RF: 25x25
            nn.ReLU(inplace=True),               # (B, 12, 7, 7) -> (B, 12, 7, 7) | RF: 25x25
            #nn.Dropout(dropout_prob),

            # Transition layer
            nn.Conv2d(12, 6, 1),                # (B, 12, 7, 7) -> (B, 6, 7, 7) | RF: 25x25
            nn.BatchNorm2d(6),                   # (B, 6, 7, 7) -> (B, 6, 7, 7) | RF: 25x25
            nn.ReLU(inplace=True),               # (B, 6, 7, 7) -> (B, 6, 7, 7) | RF: 25x25
            nn.MaxPool2d(2)                      # (B, 6, 7, 7) -> (B, 6, 3, 3) | RF: 29x29
        )

        # ------- Block 4 (Sequential) -------
        # Input: (B, 6, 3, 3) | Output: (B, 10, 3, 3) | Receptive Field: 45x45
        self.block4 = nn.Sequential(
            nn.Conv2d(6, 10, 3, padding=1),      # (B, 6, 3, 3) -> (B, 10, 3, 3) | RF: 33x33
            nn.BatchNorm2d(10),                  # (B, 10, 3, 3) -> (B, 10, 3, 3) | RF: 33x33
            nn.ReLU(inplace=True),               # (B, 10, 3, 3) -> (B, 10, 3, 3) | RF: 33x33
            #nn.Dropout(dropout_prob),

            nn.Conv2d(10, 10, 3, padding=1),     # (B, 10, 3, 3) -> (B, 10, 3, 3) | RF: 45x45
            nn.BatchNorm2d(10),                  # (B, 10, 3, 3) -> (B, 10, 3, 3) | RF: 45x45
            nn.ReLU(inplace=True)                # (B, 10, 3, 3) -> (B, 10, 3, 3) | RF: 45x45
        )

        # ------- Global Average Pooling -------
        self.gap = nn.AdaptiveAvgPool2d(1)       # (B, 10, 3, 3) -> (B, 10, 1, 1) | RF: 45x45

        # ------- Final Classifier (Sequential) -------
        self.classifier = nn.Sequential(
            nn.Flatten(),                        # (B, 10, 1, 1) -> (B, 10)
            nn.Linear(10, 10)                    # (B, 10) -> (B, 10)
        )

    def forward(self, x):
        # Process through all sequential blocks with detailed shape tracking
        x = self.block1(x)  # (B, 1, 28, 28) -> (B, 8, 14, 14) | RF: 5x5
        x = self.block2(x)  # (B, 8, 14, 14) -> (B, 6, 7, 7) | RF: 13x13
        x = self.block3(x)  # (B, 6, 7, 7) -> (B, 6, 3, 3) | RF: 29x29
        x = self.block4(x)  # (B, 6, 3, 3) -> (B, 10, 3, 3) | RF: 45x45
        
        # GAP
        x = self.gap(x)  # (B, 10, 3, 3) -> (B, 10, 1, 1) | RF: 45x45
        
        # Final classification
        x = self.classifier(x)  # (B, 10, 1, 1) -> (B, 10)
        
        return F.log_softmax(x, dim=1)  # (B, 10) -> (B, 10)

