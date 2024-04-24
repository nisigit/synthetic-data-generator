import dataclasses


@dataclasses.dataclass
class DegreeDistribution:
    activity_dict: dict

    def __post_init__(self):
        self.degrees = list(sorted(self.activity_dict.keys()))
        self.freq_dist = [self.activity_dict[i] if i in self.activity_dict else 0
                          for i in range(max(self.degrees) + 1)]

        self.probabilities = [self.activity_dict[deg] for deg in self.degrees]
        p_count = sum(self.probabilities)
        self.probabilities = [prob / p_count for prob in self.probabilities]

    def get_dist_with_min_degree(self, min_degree: int) -> 'DegreeDistribution':
        return DegreeDistribution({deg: prob for deg, prob in zip(self.degrees, self.probabilities) if deg >= min_degree})
