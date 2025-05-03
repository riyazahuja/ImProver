@[to_additive]
instance NormedGroup.to_isometricSMul_right : IsometricSMul Eᵐᵒᵖ E :=
                                              /-
                                                𝓕 : Type u_1
                                                E : Type u_2
                                                F : Type u_3
                                                inst✝¹ : SeminormedGroup E
                                                inst✝ : SeminormedGroup F
                                                s : Set E
                                                a✝ b✝ : E
                                                r : Real
                                                a : MulOpposite E
                                                b c : E
                                                ⊢ Eq (Dist.dist (HSMul.hSMul a b) (HSMul.hSMul a c)) (Dist.dist b c)
                                              -/
  ⟨fun a => Isometry.of_dist_eq fun b c => by simp [dist_eq_norm_div]⟩
                                              /-
                                                🎉 no goals
                                              -/


@[to_additive]
theorem Isometry.norm_map_of_map_one {f : E → F} (hi : Isometry f) (h₁ : f 1 = 1) (x : E) :
                      /-
                        E : Type u_2
                        F : Type u_3
                        inst✝¹ : SeminormedGroup E
                        inst✝ : SeminormedGroup F
                        f : E → F
                        hi : Isometry f
                        h₁ : Eq (f 1) 1
                        x : E
                        ⊢ Eq (Norm.norm (f x)) (Norm.norm x)
                      -/
    ‖f x‖ = ‖x‖ := by rw [← dist_one_right, ← h₁, hi.dist_eq, dist_one_right]
                      /-
                        🎉 no goals
                      -/


@[to_additive (attr := simp)]
theorem dist_mul_self_right (a b : E) : dist b (a * b) = ‖a‖ := by
  /-
    E : Type u_2
    inst✝ : SeminormedGroup E
    a b : E
    ⊢ Eq (Dist.dist b (HMul.hMul a b)) (Norm.norm a)
  -/
  rw [← dist_one_left, ← dist_mul_right 1 a b, one_mul]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem dist_mul_self_left (a b : E) : dist (a * b) b = ‖a‖ := by
  /-
    E : Type u_2
    inst✝ : SeminormedGroup E
    a b : E
    ⊢ Eq (Dist.dist (HMul.hMul a b) b) (Norm.norm a)
  -/
  rw [dist_comm, dist_mul_self_right]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem dist_div_eq_dist_mul_left (a b c : E) : dist (a / b) c = dist a (c * b) := by
  /-
    E : Type u_2
    inst✝ : SeminormedGroup E
    a b c : E
    ⊢ Eq (Dist.dist (HDiv.hDiv a b) c) (Dist.dist a (HMul.hMul c b))
  -/
  rw [← dist_mul_right _ _ b, div_mul_cancel]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem dist_div_eq_dist_mul_right (a b c : E) : dist a (b / c) = dist (a * c) b := by
  /-
    E : Type u_2
    inst✝ : SeminormedGroup E
    a b c : E
    ⊢ Eq (Dist.dist a (HDiv.hDiv b c)) (Dist.dist (HMul.hMul a c) b)
  -/
  rw [← dist_mul_right _ _ c, div_mul_cancel]
  /-
    🎉 no goals
  -/


/-- A homomorphism `f` of seminormed groups is Lipschitz, if there exists a constant `C` such that
for all `x`, one has `‖f x‖ ≤ C * ‖x‖`. The analogous condition for a linear map of
(semi)normed spaces is in `Mathlib/Analysis/NormedSpace/OperatorNorm.lean`. -/
@[to_additive "A homomorphism `f` of seminormed groups is Lipschitz, if there exists a constant
`C` such that for all `x`, one has `‖f x‖ ≤ C * ‖x‖`. The analogous condition for a linear map of
(semi)normed spaces is in `Mathlib/Analysis/NormedSpace/OperatorNorm.lean`."]
theorem MonoidHomClass.lipschitz_of_bound [MonoidHomClass 𝓕 E F] (f : 𝓕) (C : ℝ)
    (h : ∀ x, ‖f x‖ ≤ C * ‖x‖) : LipschitzWith (Real.toNNReal C) f :=
                                          /-
                                            𝓕 : Type u_1
                                            E : Type u_2
                                            F : Type u_3
                                            inst✝³ : SeminormedGroup E
                                            inst✝² : SeminormedGroup F
                                            inst✝¹ : FunLike 𝓕 E F
                                            inst✝ : MonoidHomClass 𝓕 E F
                                            f : 𝓕
                                            C : Real
                                            h : ∀ (x : E), LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.norm x))
                                            x y : E
                                            ⊢ LE.le (Dist.dist (f x) (f y)) (HMul.hMul C (Dist.dist x y))
                                          -/
  LipschitzWith.of_dist_le' fun x y => by simpa only [dist_eq_norm_div, map_div] using h (x / y)
                                          /-
                                            🎉 no goals
                                          -/


@[to_additive]
theorem lipschitzOnWith_iff_norm_div_le {f : E → F} {C : ℝ≥0} :
    LipschitzOnWith C f s ↔ ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → ‖f x / f y‖ ≤ C * ‖x / y‖ := by
  /-
    E : Type u_2
    F : Type u_3
    inst✝¹ : SeminormedGroup E
    inst✝ : SeminormedGroup F
    s : Set E
    f : E → F
    C : NNReal
    ⊢ Iff (LipschitzOnWith C f s) (∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Memb …
  -/
  simp only [lipschitzOnWith_iff_dist_le_mul, dist_eq_norm_div]
  /-
    🎉 no goals
  -/


alias ⟨LipschitzOnWith.norm_div_le, _⟩ := lipschitzOnWith_iff_norm_div_le


attribute [to_additive] LipschitzOnWith.norm_div_le


@[to_additive]
theorem LipschitzOnWith.norm_div_le_of_le {f : E → F} {C : ℝ≥0} (h : LipschitzOnWith C f s)
    (ha : a ∈ s) (hb : b ∈ s) (hr : ‖a / b‖ ≤ r) : ‖f a / f b‖ ≤ C * r :=
                                    /-
                                      E : Type u_2
                                      F : Type u_3
                                      inst✝¹ : SeminormedGroup E
                                      inst✝ : SeminormedGroup F
                                      s : Set E
                                      a b : E
                                      r : Real
                                      f : E → F
                                      C : NNReal
                                      h : LipschitzOnWith C f s
                                      ha : Membership.mem s a
                                      hb : Membership.mem s b
                                      hr : LE.le (Norm.norm (HDiv.hDiv a b)) r
                                      ⊢ LE.le (HMul.hMul (↑C) (Norm.norm (HDiv.hDiv a b))) (HMul.hMul (↑C) r)
                                    -/
  (h.norm_div_le ha hb).trans <| by gcongr
                                    /-
                                      🎉 no goals
                                    -/


@[to_additive]
theorem lipschitzWith_iff_norm_div_le {f : E → F} {C : ℝ≥0} :
    LipschitzWith C f ↔ ∀ x y, ‖f x / f y‖ ≤ C * ‖x / y‖ := by
  /-
    E : Type u_2
    F : Type u_3
    inst✝¹ : SeminormedGroup E
    inst✝ : SeminormedGroup F
    f : E → F
    C : NNReal
    ⊢ Iff (LipschitzWith C f) (∀ (x y : E), LE.le (Norm.norm (HDiv.hDiv (f x) (f y …
  -/
  simp only [lipschitzWith_iff_dist_le_mul, dist_eq_norm_div]
  /-
    🎉 no goals
  -/


alias ⟨LipschitzWith.norm_div_le, _⟩ := lipschitzWith_iff_norm_div_le


attribute [to_additive] LipschitzWith.norm_div_le


@[to_additive]
theorem LipschitzWith.norm_div_le_of_le {f : E → F} {C : ℝ≥0} (h : LipschitzWith C f)
    (hr : ‖a / b‖ ≤ r) : ‖f a / f b‖ ≤ C * r :=
                                  /-
                                    E : Type u_2
                                    F : Type u_3
                                    inst✝¹ : SeminormedGroup E
                                    inst✝ : SeminormedGroup F
                                    a b : E
                                    r : Real
                                    f : E → F
                                    C : NNReal
                                    h : LipschitzWith C f
                                    hr : LE.le (Norm.norm (HDiv.hDiv a b)) r
                                    ⊢ LE.le (HMul.hMul (↑C) (Norm.norm (HDiv.hDiv a b))) (HMul.hMul (↑C) r)
                                  -/
  (h.norm_div_le _ _).trans <| by gcongr
                                  /-
                                    🎉 no goals
                                  -/


/-- A homomorphism `f` of seminormed groups is continuous, if there exists a constant `C` such that
for all `x`, one has `‖f x‖ ≤ C * ‖x‖`. -/
@[to_additive "A homomorphism `f` of seminormed groups is continuous, if there exists a constant `C`
such that for all `x`, one has `‖f x‖ ≤ C * ‖x‖`"]
theorem MonoidHomClass.continuous_of_bound [MonoidHomClass 𝓕 E F] (f : 𝓕) (C : ℝ)
    (h : ∀ x, ‖f x‖ ≤ C * ‖x‖) : Continuous f :=
  (MonoidHomClass.lipschitz_of_bound f C h).continuous


@[to_additive]
theorem MonoidHomClass.uniformContinuous_of_bound [MonoidHomClass 𝓕 E F] (f : 𝓕) (C : ℝ)
    (h : ∀ x, ‖f x‖ ≤ C * ‖x‖) : UniformContinuous f :=
  (MonoidHomClass.lipschitz_of_bound f C h).uniformContinuous


@[to_additive]
theorem MonoidHomClass.isometry_iff_norm [MonoidHomClass 𝓕 E F] (f : 𝓕) :
    Isometry f ↔ ∀ x, ‖f x‖ = ‖x‖ := by
  /-
    𝓕 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : SeminormedGroup E
    inst✝² : SeminormedGroup F
    inst✝¹ : FunLike 𝓕 E F
    inst✝ : MonoidHomClass 𝓕 E F
    f : 𝓕
    ⊢ Iff (Isometry ⇑f) (∀ (x : E), Eq (Norm.norm (f x)) (Norm.norm x))
  -/
  simp only [isometry_iff_dist_eq, dist_eq_norm_div, ← map_div]
  /-
    𝓕 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : SeminormedGroup E
    inst✝² : SeminormedGroup F
    inst✝¹ : FunLike 𝓕 E F
    inst✝ : MonoidHomClass 𝓕 E F
    f : 𝓕
    ⊢ Iff (∀ (x y : E), Eq (Norm.norm (f (HDiv.hDiv x y))) (Norm.norm (HDiv.hDiv x …
  -/
  refine ⟨fun h x => ?_, fun h x y => h _⟩
  /-
    𝓕 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : SeminormedGroup E
    inst✝² : SeminormedGroup F
    inst✝¹ : FunLike 𝓕 E F
    inst✝ : MonoidHomClass 𝓕 E F
    f : 𝓕
    h : ∀ (x y : E), Eq (Norm.norm (f (HDiv.hDiv x y))) (Norm.norm (HDiv.hDiv x y))
    x : E
    ⊢ Eq (Norm.norm (f x)) (Norm.norm x)
  -/
  simpa using h x 1
  /-
    🎉 no goals
  -/


alias ⟨_, MonoidHomClass.isometry_of_norm⟩ := MonoidHomClass.isometry_iff_norm


attribute [to_additive] MonoidHomClass.isometry_of_norm


@[to_additive]
theorem MonoidHomClass.lipschitz_of_bound_nnnorm [MonoidHomClass 𝓕 E F] (f : 𝓕) (C : ℝ≥0)
    (h : ∀ x, ‖f x‖₊ ≤ C * ‖x‖₊) : LipschitzWith C f :=
  @Real.toNNReal_coe C ▸ MonoidHomClass.lipschitz_of_bound f C h


@[to_additive]
theorem MonoidHomClass.antilipschitz_of_bound [MonoidHomClass 𝓕 E F] (f : 𝓕) {K : ℝ≥0}
    (h : ∀ x, ‖x‖ ≤ K * ‖f x‖) : AntilipschitzWith K f :=
  AntilipschitzWith.of_le_mul_dist fun x y => by
    /-
      𝓕 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝³ : SeminormedGroup E
      inst✝² : SeminormedGroup F
      inst✝¹ : FunLike 𝓕 E F
      inst✝ : MonoidHomClass 𝓕 E F
      f : 𝓕
      K : NNReal
      h : ∀ (x : E), LE.le (Norm.norm x) (HMul.hMul (↑K) (Norm.norm (f x)))
      x y : E
      ⊢ LE.le (Dist.dist x y) (HMul.hMul (↑K) (Dist.dist (f x) (f y)))
    -/
    simpa only [dist_eq_norm_div, map_div] using h (x / y)
    /-
      🎉 no goals
    -/


@[to_additive LipschitzWith.norm_le_mul]
theorem LipschitzWith.norm_le_mul' {f : E → F} {K : ℝ≥0} (h : LipschitzWith K f) (hf : f 1 = 1)
                                /-
                                  E : Type u_2
                                  F : Type u_3
                                  inst✝¹ : SeminormedGroup E
                                  inst✝ : SeminormedGroup F
                                  f : E → F
                                  K : NNReal
                                  h : LipschitzWith K f
                                  hf : Eq (f 1) 1
                                  x : E
                                  ⊢ LE.le (Norm.norm (f x)) (HMul.hMul (↑K) (Norm.norm x))
                                -/
    (x) : ‖f x‖ ≤ K * ‖x‖ := by simpa only [dist_one_right, hf] using h.dist_le_mul x 1
                                /-
                                  🎉 no goals
                                -/


@[to_additive LipschitzWith.nnorm_le_mul]
theorem LipschitzWith.nnorm_le_mul' {f : E → F} {K : ℝ≥0} (h : LipschitzWith K f) (hf : f 1 = 1)
    (x) : ‖f x‖₊ ≤ K * ‖x‖₊ :=
  h.norm_le_mul' hf x


@[to_additive AntilipschitzWith.le_mul_norm]
theorem AntilipschitzWith.le_mul_norm' {f : E → F} {K : ℝ≥0} (h : AntilipschitzWith K f)
    (hf : f 1 = 1) (x) : ‖x‖ ≤ K * ‖f x‖ := by
  /-
    E : Type u_2
    F : Type u_3
    inst✝¹ : SeminormedGroup E
    inst✝ : SeminormedGroup F
    f : E → F
    K : NNReal
    h : AntilipschitzWith K f
    hf : Eq (f 1) 1
    x : E
    ⊢ LE.le (Norm.norm x) (HMul.hMul (↑K) (Norm.norm (f x)))
  -/
  simpa only [dist_one_right, hf] using h.le_mul_dist x 1
  /-
    🎉 no goals
  -/


@[to_additive AntilipschitzWith.le_mul_nnnorm]
theorem AntilipschitzWith.le_mul_nnnorm' {f : E → F} {K : ℝ≥0} (h : AntilipschitzWith K f)
    (hf : f 1 = 1) (x) : ‖x‖₊ ≤ K * ‖f x‖₊ :=
  h.le_mul_norm' hf x


@[to_additive]
theorem OneHomClass.bound_of_antilipschitz [OneHomClass 𝓕 E F] (f : 𝓕) {K : ℝ≥0}
    (h : AntilipschitzWith K f) (x) : ‖x‖ ≤ K * ‖f x‖ :=
  h.le_mul_nnnorm' (map_one f) x


@[to_additive]
theorem Isometry.nnnorm_map_of_map_one {f : E → F} (hi : Isometry f) (h₁ : f 1 = 1) (x : E) :
    ‖f x‖₊ = ‖x‖₊ :=
  Subtype.ext <| hi.norm_map_of_map_one h₁ x


@[to_additive lipschitzWith_one_norm]
theorem lipschitzWith_one_norm' : LipschitzWith 1 (norm : E → ℝ) := by
  /-
    E : Type u_2
    inst✝ : SeminormedGroup E
    ⊢ LipschitzWith 1 Norm.norm
  -/
  simpa only [dist_one_left] using LipschitzWith.dist_right (1 : E)
  /-
    🎉 no goals
  -/


@[to_additive lipschitzWith_one_nnnorm]
theorem lipschitzWith_one_nnnorm' : LipschitzWith 1 (NNNorm.nnnorm : E → ℝ≥0) :=
  lipschitzWith_one_norm'


@[to_additive uniformContinuous_norm]
theorem uniformContinuous_norm' : UniformContinuous (norm : E → ℝ) :=
  lipschitzWith_one_norm'.uniformContinuous


@[to_additive uniformContinuous_nnnorm]
theorem uniformContinuous_nnnorm' : UniformContinuous fun a : E => ‖a‖₊ :=
  uniformContinuous_norm'.subtype_mk _


@[to_additive]
instance NormedGroup.to_isometricSMul_left : IsometricSMul E E :=
                                              /-
                                                𝓕 : Type u_1
                                                E : Type u_2
                                                F : Type u_3
                                                inst✝¹ : SeminormedCommGroup E
                                                inst✝ : SeminormedCommGroup F
                                                a₁ a₂ b₁ b₂ : E
                                                r₁ r₂ : Real
                                                a b c : E
                                                ⊢ Eq (Dist.dist (HSMul.hSMul a b) (HSMul.hSMul a c)) (Dist.dist b c)
                                              -/
  ⟨fun a => Isometry.of_dist_eq fun b c => by simp [dist_eq_norm_div]⟩
                                              /-
                                                🎉 no goals
                                              -/


@[to_additive (attr := simp)]
theorem dist_self_mul_right (a b : E) : dist a (a * b) = ‖b‖ := by
  /-
    E : Type u_2
    inst✝ : SeminormedCommGroup E
    a b : E
    ⊢ Eq (Dist.dist a (HMul.hMul a b)) (Norm.norm b)
  -/
  rw [← dist_one_left, ← dist_mul_left a 1 b, mul_one]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem dist_self_mul_left (a b : E) : dist (a * b) a = ‖b‖ := by
  /-
    E : Type u_2
    inst✝ : SeminormedCommGroup E
    a b : E
    ⊢ Eq (Dist.dist (HMul.hMul a b) a) (Norm.norm b)
  -/
  rw [dist_comm, dist_self_mul_right]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp 1001)] -- Increase priority because `simp` can prove this
theorem dist_self_div_right (a b : E) : dist a (a / b) = ‖b‖ := by
  /-
    E : Type u_2
    inst✝ : SeminormedCommGroup E
    a b : E
    ⊢ Eq (Dist.dist a (HDiv.hDiv a b)) (Norm.norm b)
  -/
  rw [div_eq_mul_inv, dist_self_mul_right, norm_inv']
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp 1001)] -- Increase priority because `simp` can prove this
theorem dist_self_div_left (a b : E) : dist (a / b) a = ‖b‖ := by
  /-
    E : Type u_2
    inst✝ : SeminormedCommGroup E
    a b : E
    ⊢ Eq (Dist.dist (HDiv.hDiv a b) a) (Norm.norm b)
  -/
  rw [dist_comm, dist_self_div_right]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem dist_mul_mul_le (a₁ a₂ b₁ b₂ : E) : dist (a₁ * a₂) (b₁ * b₂) ≤ dist a₁ b₁ + dist a₂ b₂ := by
  /-
    E : Type u_2
    inst✝ : SeminormedCommGroup E
    a₁ a₂ b₁ b₂ : E
    ⊢ LE.le (Dist.dist (HMul.hMul a₁ a₂) (HMul.hMul b₁ b₂)) (HAdd.hAdd (Dist.dist  …
  -/
  simpa only [dist_mul_left, dist_mul_right] using dist_triangle (a₁ * a₂) (b₁ * a₂) (b₁ * b₂)
  /-
    🎉 no goals
  -/


@[to_additive]
theorem dist_mul_mul_le_of_le (h₁ : dist a₁ b₁ ≤ r₁) (h₂ : dist a₂ b₂ ≤ r₂) :
    dist (a₁ * a₂) (b₁ * b₂) ≤ r₁ + r₂ :=
  (dist_mul_mul_le a₁ a₂ b₁ b₂).trans <| add_le_add h₁ h₂


@[to_additive]
theorem dist_div_div_le (a₁ a₂ b₁ b₂ : E) : dist (a₁ / a₂) (b₁ / b₂) ≤ dist a₁ b₁ + dist a₂ b₂ := by
  /-
    E : Type u_2
    inst✝ : SeminormedCommGroup E
    a₁ a₂ b₁ b₂ : E
    ⊢ LE.le (Dist.dist (HDiv.hDiv a₁ a₂) (HDiv.hDiv b₁ b₂)) (HAdd.hAdd (Dist.dist  …
  -/
  simpa only [div_eq_mul_inv, dist_inv_inv] using dist_mul_mul_le a₁ a₂⁻¹ b₁ b₂⁻¹
  /-
    🎉 no goals
  -/


@[to_additive]
theorem dist_div_div_le_of_le (h₁ : dist a₁ b₁ ≤ r₁) (h₂ : dist a₂ b₂ ≤ r₂) :
    dist (a₁ / a₂) (b₁ / b₂) ≤ r₁ + r₂ :=
  (dist_div_div_le a₁ a₂ b₁ b₂).trans <| add_le_add h₁ h₂


@[to_additive]
theorem abs_dist_sub_le_dist_mul_mul (a₁ a₂ b₁ b₂ : E) :
    |dist a₁ b₁ - dist a₂ b₂| ≤ dist (a₁ * a₂) (b₁ * b₂) := by
  simpa only [dist_mul_left, dist_mul_right, dist_comm b₂] using
    abs_dist_sub_le (a₁ * a₂) (b₁ * b₂) (b₁ * a₂)


@[to_additive]
theorem nndist_mul_mul_le (a₁ a₂ b₁ b₂ : E) :
    nndist (a₁ * a₂) (b₁ * b₂) ≤ nndist a₁ b₁ + nndist a₂ b₂ :=
  NNReal.coe_le_coe.1 <| dist_mul_mul_le a₁ a₂ b₁ b₂


@[to_additive]
theorem edist_mul_mul_le (a₁ a₂ b₁ b₂ : E) :
    edist (a₁ * a₂) (b₁ * b₂) ≤ edist a₁ b₁ + edist a₂ b₂ := by
  /-
    E : Type u_2
    inst✝ : SeminormedCommGroup E
    a₁ a₂ b₁ b₂ : E
    ⊢ LE.le (EDist.edist (HMul.hMul a₁ a₂) (HMul.hMul b₁ b₂)) (HAdd.hAdd (EDist.ed …
  -/
  simp only [edist_nndist]
  /-
    E : Type u_2
    inst✝ : SeminormedCommGroup E
    a₁ a₂ b₁ b₂ : E
    ⊢ LE.le (↑(NNDist.nndist (HMul.hMul a₁ a₂) (HMul.hMul b₁ b₂))) (HAdd.hAdd ↑(NN …
  -/
  norm_cast
  /-
    E : Type u_2
    inst✝ : SeminormedCommGroup E
    a₁ a₂ b₁ b₂ : E
    ⊢ LE.le (NNDist.nndist (HMul.hMul a₁ a₂) (HMul.hMul b₁ b₂)) (HAdd.hAdd (NNDist …
  -/
  apply nndist_mul_mul_le
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
                                                                            /-
                                                                              α : Type u_4
                                                                              E : Type u_5
                                                                              inst✝¹ : SeminormedCommGroup E
                                                                              inst✝ : PseudoEMetricSpace α
                                                                              K : NNReal
                                                                              f : α → E
                                                                              ⊢ Iff (LipschitzWith K (Inv.inv f)) (LipschitzWith K f)
                                                                            -/
lemma lipschitzWith_inv_iff : LipschitzWith K f⁻¹ ↔ LipschitzWith K f := by simp [LipschitzWith]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


@[to_additive (attr := simp)]
lemma antilipschitzWith_inv_iff : AntilipschitzWith K f⁻¹ ↔ AntilipschitzWith K f := by
  /-
    α : Type u_4
    E : Type u_5
    inst✝¹ : SeminormedCommGroup E
    inst✝ : PseudoEMetricSpace α
    K : NNReal
    f : α → E
    ⊢ Iff (AntilipschitzWith K (Inv.inv f)) (AntilipschitzWith K f)
  -/
  simp [AntilipschitzWith]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma lipschitzOnWith_inv_iff : LipschitzOnWith K f⁻¹ s ↔ LipschitzOnWith K f s := by
  /-
    α : Type u_4
    E : Type u_5
    inst✝¹ : SeminormedCommGroup E
    inst✝ : PseudoEMetricSpace α
    K : NNReal
    f : α → E
    s : Set α
    ⊢ Iff (LipschitzOnWith K (Inv.inv f) s) (LipschitzOnWith K f s)
  -/
  simp [LipschitzOnWith]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma locallyLipschitz_inv_iff : LocallyLipschitz f⁻¹ ↔ LocallyLipschitz f := by
  /-
    α : Type u_4
    E : Type u_5
    inst✝¹ : SeminormedCommGroup E
    inst✝ : PseudoEMetricSpace α
    f : α → E
    ⊢ Iff (LocallyLipschitz (Inv.inv f)) (LocallyLipschitz f)
  -/
  simp [LocallyLipschitz]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma locallyLipschitzOn_inv_iff : LocallyLipschitzOn s f⁻¹ ↔ LocallyLipschitzOn s f := by
  /-
    α : Type u_4
    E : Type u_5
    inst✝¹ : SeminormedCommGroup E
    inst✝ : PseudoEMetricSpace α
    f : α → E
    s : Set α
    ⊢ Iff (LocallyLipschitzOn s (Inv.inv f)) (LocallyLipschitzOn s f)
  -/
  simp [LocallyLipschitzOn]
  /-
    🎉 no goals
  -/


@[to_additive] alias ⟨LipschitzWith.of_inv, LipschitzWith.inv⟩ := lipschitzWith_inv_iff

@[to_additive] alias ⟨AntilipschitzWith.of_inv, AntilipschitzWith.inv⟩ := antilipschitzWith_inv_iff

@[to_additive] alias ⟨LipschitzOnWith.of_inv, LipschitzOnWith.inv⟩ := lipschitzOnWith_inv_iff

@[to_additive] alias ⟨LocallyLipschitz.of_inv, LocallyLipschitz.inv⟩ := locallyLipschitz_inv_iff

@[to_additive]
alias ⟨LocallyLipschitzOn.of_inv, LocallyLipschitzOn.inv⟩ := locallyLipschitzOn_inv_iff


@[to_additive]
lemma LipschitzOnWith.mul (hf : LipschitzOnWith Kf f s) (hg : LipschitzOnWith Kg g s) :
    LipschitzOnWith (Kf + Kg) (fun x ↦ f x * g x) s := fun x hx y hy ↦
  calc
    edist (f x * g x) (f y * g y) ≤ edist (f x) (f y) + edist (g x) (g y) :=
      edist_mul_mul_le _ _ _ _
    _ ≤ Kf * edist x y + Kg * edist x y := add_le_add (hf hx hy) (hg hx hy)
    _ = (Kf + Kg) * edist x y := (add_mul _ _ _).symm


@[to_additive]
lemma LipschitzWith.mul (hf : LipschitzWith Kf f) (hg : LipschitzWith Kg g) :
    LipschitzWith (Kf + Kg) fun x ↦ f x * g x := by
  /-
    α : Type u_4
    E : Type u_5
    inst✝¹ : SeminormedCommGroup E
    inst✝ : PseudoEMetricSpace α
    Kf Kg : NNReal
    f g : α → E
    hf : LipschitzWith Kf f
    hg : LipschitzWith Kg g
    ⊢ LipschitzWith (HAdd.hAdd Kf Kg) fun x => HMul.hMul (f x) (g x)
  -/
  simpa [← lipschitzOnWith_univ] using hf.lipschitzOnWith.mul hg.lipschitzOnWith
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-25")] alias LipschitzWith.mul' := LipschitzWith.mul


@[to_additive]
lemma LocallyLipschitzOn.mul (hf : LocallyLipschitzOn s f) (hg : LocallyLipschitzOn s g) :
    LocallyLipschitzOn s fun x ↦ f x * g x := fun x hx ↦ by
  /-
    α : Type u_4
    E : Type u_5
    inst✝¹ : SeminormedCommGroup E
    inst✝ : PseudoEMetricSpace α
    f g : α → E
    s : Set α
    hf : LocallyLipschitzOn s f
    hg : LocallyLipschitzOn s g
    x : α
    hx : Membership.mem s x
    ⊢ Exists fun K => Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Lip …
  -/
  obtain ⟨Kf, t, ht, hKf⟩ := hf hx
  /-
    case intro.intro.intro
    α : Type u_4
    E : Type u_5
    inst✝¹ : SeminormedCommGroup E
    inst✝ : PseudoEMetricSpace α
    f g : α → E
    s : Set α
    hf : LocallyLipschitzOn s f
    hg : LocallyLipschitzOn s g
    x : α
    hx : Membership.mem s x
    Kf : NNReal
    t : Set α
    ht : Membership.mem (nhdsWithin x s) t
    hKf : LipschitzOnWith Kf f t
    ⊢ Exists fun K => Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Lip …
  -/
  obtain ⟨Kg, u, hu, hKg⟩ := hg hx
  exact ⟨Kf + Kg, t ∩ u, inter_mem ht hu,
    (hKf.mono Set.inter_subset_left).mul (hKg.mono Set.inter_subset_right)⟩


@[to_additive]
lemma LocallyLipschitz.mul (hf : LocallyLipschitz f) (hg : LocallyLipschitz g) :
    LocallyLipschitz fun x ↦ f x * g x := by
  /-
    α : Type u_4
    E : Type u_5
    inst✝¹ : SeminormedCommGroup E
    inst✝ : PseudoEMetricSpace α
    f g : α → E
    hf : LocallyLipschitz f
    hg : LocallyLipschitz g
    ⊢ LocallyLipschitz fun x => HMul.hMul (f x) (g x)
  -/
  simpa [← locallyLipschitzOn_univ] using hf.locallyLipschitzOn.mul hg.locallyLipschitzOn
  /-
    🎉 no goals
  -/


@[to_additive]
lemma LipschitzOnWith.div (hf : LipschitzOnWith Kf f s) (hg : LipschitzOnWith Kg g s) :
    LipschitzOnWith (Kf + Kg) (fun x ↦ f x / g x) s := by
  /-
    α : Type u_4
    E : Type u_5
    inst✝¹ : SeminormedCommGroup E
    inst✝ : PseudoEMetricSpace α
    Kf Kg : NNReal
    f g : α → E
    s : Set α
    hf : LipschitzOnWith Kf f s
    hg : LipschitzOnWith Kg g s
    ⊢ LipschitzOnWith (HAdd.hAdd Kf Kg) (fun x => HDiv.hDiv (f x) (g x)) s
  -/
  simpa only [div_eq_mul_inv] using hf.mul hg.inv
  /-
    🎉 no goals
  -/


@[to_additive]
theorem LipschitzWith.div (hf : LipschitzWith Kf f) (hg : LipschitzWith Kg g) :
    LipschitzWith (Kf + Kg) fun x => f x / g x := by
  /-
    α : Type u_4
    E : Type u_5
    inst✝¹ : SeminormedCommGroup E
    inst✝ : PseudoEMetricSpace α
    Kf Kg : NNReal
    f g : α → E
    hf : LipschitzWith Kf f
    hg : LipschitzWith Kg g
    ⊢ LipschitzWith (HAdd.hAdd Kf Kg) fun x => HDiv.hDiv (f x) (g x)
  -/
  simpa only [div_eq_mul_inv] using hf.mul hg.inv
  /-
    🎉 no goals
  -/


@[to_additive]
lemma LocallyLipschitzOn.div (hf : LocallyLipschitzOn s f) (hg : LocallyLipschitzOn s g) :
    LocallyLipschitzOn s fun x ↦ f x / g x := by
  /-
    α : Type u_4
    E : Type u_5
    inst✝¹ : SeminormedCommGroup E
    inst✝ : PseudoEMetricSpace α
    f g : α → E
    s : Set α
    hf : LocallyLipschitzOn s f
    hg : LocallyLipschitzOn s g
    ⊢ LocallyLipschitzOn s fun x => HDiv.hDiv (f x) (g x)
  -/
  simpa only [div_eq_mul_inv] using hf.mul hg.inv
  /-
    🎉 no goals
  -/


@[to_additive]
lemma LocallyLipschitz.div (hf : LocallyLipschitz f) (hg : LocallyLipschitz g) :
    LocallyLipschitz fun x ↦ f x / g x := by
  /-
    α : Type u_4
    E : Type u_5
    inst✝¹ : SeminormedCommGroup E
    inst✝ : PseudoEMetricSpace α
    f g : α → E
    hf : LocallyLipschitz f
    hg : LocallyLipschitz g
    ⊢ LocallyLipschitz fun x => HDiv.hDiv (f x) (g x)
  -/
  simpa only [div_eq_mul_inv] using hf.mul hg.inv
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mul_lipschitzWith (hf : AntilipschitzWith Kf f) (hg : LipschitzWith Kg g) (hK : Kg < Kf⁻¹) :
    AntilipschitzWith (Kf⁻¹ - Kg)⁻¹ fun x => f x * g x := by
  /-
    α : Type u_4
    E : Type u_5
    inst✝¹ : SeminormedCommGroup E
    inst✝ : PseudoEMetricSpace α
    Kf Kg : NNReal
    f g : α → E
    hf : AntilipschitzWith Kf f
    hg : LipschitzWith Kg g
    hK : LT.lt Kg (Inv.inv Kf)
    ⊢ AntilipschitzWith (Inv.inv (HSub.hSub (Inv.inv Kf) Kg)) fun x => HMul.hMul ( …
  -/
  letI : PseudoMetricSpace α := PseudoEMetricSpace.toPseudoMetricSpace hf.edist_ne_top
  /-
    α : Type u_4
    E : Type u_5
    inst✝¹ : SeminormedCommGroup E
    inst✝ : PseudoEMetricSpace α
    Kf Kg : NNReal
    f g : α → E
    hf : AntilipschitzWith Kf f
    hg : LipschitzWith Kg g
    hK : LT.lt Kg (Inv.inv Kf)
    this : PseudoMetricSpace α := PseudoEMetricSpace.toPseudoMetricSpace ⋯
    ⊢ AntilipschitzWith (Inv.inv (HSub.hSub (Inv.inv Kf) Kg)) fun x => HMul.hMul ( …
  -/
  refine AntilipschitzWith.of_le_mul_dist fun x y => ?_
  /-
    α : Type u_4
    E : Type u_5
    inst✝¹ : SeminormedCommGroup E
    inst✝ : PseudoEMetricSpace α
    Kf Kg : NNReal
    f g : α → E
    hf : AntilipschitzWith Kf f
    hg : LipschitzWith Kg g
    hK : LT.lt Kg (Inv.inv Kf)
    this : PseudoMetricSpace α := PseudoEMetricSpace.toPseudoMetricSpace ⋯
    x y : α
    ⊢ LE.le (Dist.dist x y) (HMul.hMul (↑(Inv.inv (HSub.hSub (Inv.inv Kf) Kg))) (D …
  -/
  rw [NNReal.coe_inv, ← _root_.div_eq_inv_mul]
  /-
    α : Type u_4
    E : Type u_5
    inst✝¹ : SeminormedCommGroup E
    inst✝ : PseudoEMetricSpace α
    Kf Kg : NNReal
    f g : α → E
    hf : AntilipschitzWith Kf f
    hg : LipschitzWith Kg g
    hK : LT.lt Kg (Inv.inv Kf)
    this : PseudoMetricSpace α := PseudoEMetricSpace.toPseudoMetricSpace ⋯
    x y : α
    ⊢ LE.le (Dist.dist x y) (HDiv.hDiv (Dist.dist (HMul.hMul (f x) (g x)) (HMul.hM …
  -/
  rw [le_div_iff₀ (NNReal.coe_pos.2 <| tsub_pos_iff_lt.2 hK)]
  /-
    α : Type u_4
    E : Type u_5
    inst✝¹ : SeminormedCommGroup E
    inst✝ : PseudoEMetricSpace α
    Kf Kg : NNReal
    f g : α → E
    hf : AntilipschitzWith Kf f
    hg : LipschitzWith Kg g
    hK : LT.lt Kg (Inv.inv Kf)
    this : PseudoMetricSpace α := PseudoEMetricSpace.toPseudoMetricSpace ⋯
    x y : α
    ⊢ LE.le (HMul.hMul (Dist.dist x y) ↑(HSub.hSub (Inv.inv Kf) Kg)) (Dist.dist (H …
  -/
  rw [mul_comm, NNReal.coe_sub hK.le, sub_mul]
  calc
    ↑Kf⁻¹ * dist x y - Kg * dist x y ≤ dist (f x) (f y) - dist (g x) (g y) :=
      sub_le_sub (hf.mul_le_dist x y) (hg.dist_le_mul x y)
    _ ≤ _ := le_trans (le_abs_self _) (abs_dist_sub_le_dist_mul_mul _ _ _ _)


@[to_additive]
theorem mul_div_lipschitzWith (hf : AntilipschitzWith Kf f) (hg : LipschitzWith Kg (g / f))
    (hK : Kg < Kf⁻¹) : AntilipschitzWith (Kf⁻¹ - Kg)⁻¹ g := by
  /-
    α : Type u_4
    E : Type u_5
    inst✝¹ : SeminormedCommGroup E
    inst✝ : PseudoEMetricSpace α
    Kf Kg : NNReal
    f g : α → E
    hf : AntilipschitzWith Kf f
    hg : LipschitzWith Kg (HDiv.hDiv g f)
    hK : LT.lt Kg (Inv.inv Kf)
    ⊢ AntilipschitzWith (Inv.inv (HSub.hSub (Inv.inv Kf) Kg)) g
  -/
  simpa only [Pi.div_apply, mul_div_cancel] using hf.mul_lipschitzWith hg hK
  /-
    🎉 no goals
  -/


@[to_additive le_mul_norm_sub]
theorem le_mul_norm_div {f : E → F} (hf : AntilipschitzWith K f) (x y : E) :
                                    /-
                                      F : Type u_3
                                      inst✝¹ : SeminormedCommGroup F
                                      E : Type u_5
                                      inst✝ : SeminormedCommGroup E
                                      K : NNReal
                                      f : E → F
                                      hf : AntilipschitzWith K f
                                      x y : E
                                      ⊢ LE.le (Norm.norm (HDiv.hDiv x y)) (HMul.hMul (↑K) (Norm.norm (HDiv.hDiv (f x …
                                    -/
    ‖x / y‖ ≤ K * ‖f x / f y‖ := by simp [← dist_eq_norm_div, hf.le_mul_dist x y]
                                    /-
                                      🎉 no goals
                                    -/


@[to_additive]
instance (priority := 100) SeminormedCommGroup.to_lipschitzMul : LipschitzMul E :=
  ⟨⟨1 + 1, LipschitzWith.prod_fst.mul LipschitzWith.prod_snd⟩⟩

-- See note [lower instance priority]

/-- A seminormed group is a uniform group, i.e., multiplication and division are uniformly
continuous. -/
@[to_additive "A seminormed group is a uniform additive group, i.e., addition and subtraction are
uniformly continuous."]
instance (priority := 100) SeminormedCommGroup.to_uniformGroup : UniformGroup E :=
  ⟨(LipschitzWith.prod_fst.div LipschitzWith.prod_snd).uniformContinuous⟩

-- short-circuit type class inference
-- See note [lower instance priority]

@[to_additive]
instance (priority := 100) SeminormedCommGroup.toTopologicalGroup : TopologicalGroup E :=
  inferInstance


@[to_additive instNorm]
instance instMulNorm : Norm (SeparationQuotient E) where
  norm := lift Norm.norm fun _ _ h => h.norm_eq_norm'


set_option linter.docPrime false in
@[to_additive (attr := simp) norm_mk]
theorem norm_mk' (p : E) : ‖mk p‖ = ‖p‖ := rfl


@[to_additive]
instance : NormedCommGroup (SeparationQuotient E) where
  __ : CommGroup (SeparationQuotient E) := instCommGroup
  dist_eq := Quotient.ind₂ dist_eq_norm_div


@[to_additive]
theorem mk_eq_one_iff {p : E} : mk p = 1 ↔ ‖p‖ = 0 := by
  /-
    E : Type u_2
    inst✝ : SeminormedCommGroup E
    p : E
    ⊢ Iff (Eq (SeparationQuotient.mk p) 1) (Eq (Norm.norm p) 0)
  -/
  rw [← norm_mk', norm_eq_zero']
  /-
    🎉 no goals
  -/


set_option linter.docPrime false in
@[to_additive (attr := simp) nnnorm_mk]
theorem nnnorm_mk' (p : E) : ‖mk p‖₊ = ‖p‖₊ := rfl


@[to_additive]
theorem cauchySeq_prod_of_eventually_eq {u v : ℕ → E} {N : ℕ} (huv : ∀ n ≥ N, u n = v n)
    (hv : CauchySeq fun n => ∏ k ∈ range (n + 1), v k) :
    CauchySeq fun n => ∏ k ∈ range (n + 1), u k := by
  /-
    E : Type u_2
    inst✝ : SeminormedCommGroup E
    u v : Nat → E
    N : Nat
    huv : ∀ (n : Nat), GE.ge n N → Eq (u n) (v n)
    hv : CauchySeq fun n => (Finset.range (HAdd.hAdd n 1)).prod fun k => v k
    ⊢ CauchySeq fun n => (Finset.range (HAdd.hAdd n 1)).prod fun k => u k
  -/
  let d : ℕ → E := fun n => ∏ k ∈ range (n + 1), u k / v k
  rw [show (fun n => ∏ k ∈ range (n + 1), u k) = d * fun n => ∏ k ∈ range (n + 1), v k
      by ext n; simp [d]]
  /-
    E : Type u_2
    inst✝ : SeminormedCommGroup E
    u v : Nat → E
    N : Nat
    huv : ∀ (n : Nat), GE.ge n N → Eq (u n) (v n)
    hv : CauchySeq fun n => (Finset.range (HAdd.hAdd n 1)).prod fun k => v k
    d : Nat → E := fun n => (Finset.range (HAdd.hAdd n 1)).prod fun k => HDiv.hDiv …
    ⊢ CauchySeq (HMul.hMul d fun n => (Finset.range (HAdd.hAdd n 1)).prod fun k => …
  -/
  suffices ∀ n ≥ N, d n = d N from (tendsto_atTop_of_eventually_const this).cauchySeq.mul hv
  /-
    E : Type u_2
    inst✝ : SeminormedCommGroup E
    u v : Nat → E
    N : Nat
    huv : ∀ (n : Nat), GE.ge n N → Eq (u n) (v n)
    hv : CauchySeq fun n => (Finset.range (HAdd.hAdd n 1)).prod fun k => v k
    d : Nat → E := fun n => (Finset.range (HAdd.hAdd n 1)).prod fun k => HDiv.hDiv …
    ⊢ ∀ (n : Nat), GE.ge n N → Eq (d n) (d N)
  -/
  intro n hn
  /-
    E : Type u_2
    inst✝ : SeminormedCommGroup E
    u v : Nat → E
    N : Nat
    huv : ∀ (n : Nat), GE.ge n N → Eq (u n) (v n)
    hv : CauchySeq fun n => (Finset.range (HAdd.hAdd n 1)).prod fun k => v k
    d : Nat → E := fun n => (Finset.range (HAdd.hAdd n 1)).prod fun k => HDiv.hDiv …
    n : Nat
    hn : GE.ge n N
    ⊢ Eq (d n) (d N)
  -/
  dsimp [d]
  /-
    E : Type u_2
    inst✝ : SeminormedCommGroup E
    u v : Nat → E
    N : Nat
    huv : ∀ (n : Nat), GE.ge n N → Eq (u n) (v n)
    hv : CauchySeq fun n => (Finset.range (HAdd.hAdd n 1)).prod fun k => v k
    d : Nat → E := fun n => (Finset.range (HAdd.hAdd n 1)).prod fun k => HDiv.hDiv …
    n : Nat
    hn : GE.ge n N
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).prod fun k => HDiv.hDiv (u k) (v k)) ((Fi …
  -/
  rw [eventually_constant_prod _ (add_le_add_right hn 1)]
  /-
    E : Type u_2
    inst✝ : SeminormedCommGroup E
    u v : Nat → E
    N : Nat
    huv : ∀ (n : Nat), GE.ge n N → Eq (u n) (v n)
    hv : CauchySeq fun n => (Finset.range (HAdd.hAdd n 1)).prod fun k => v k
    d : Nat → E := fun n => (Finset.range (HAdd.hAdd n 1)).prod fun k => HDiv.hDiv …
    n : Nat
    hn : GE.ge n N
    ⊢ ∀ (n : Nat), GE.ge n (HAdd.hAdd N 1) → Eq (HDiv.hDiv (u n) (v n)) 1
  -/
  intro m hm
  /-
    E : Type u_2
    inst✝ : SeminormedCommGroup E
    u v : Nat → E
    N : Nat
    huv : ∀ (n : Nat), GE.ge n N → Eq (u n) (v n)
    hv : CauchySeq fun n => (Finset.range (HAdd.hAdd n 1)).prod fun k => v k
    d : Nat → E := fun n => (Finset.range (HAdd.hAdd n 1)).prod fun k => HDiv.hDiv …
    n : Nat
    hn : GE.ge n N
    m : Nat
    hm : GE.ge m (HAdd.hAdd N 1)
    ⊢ Eq (HDiv.hDiv (u m) (v m)) 1
  -/
  simp [huv m (le_of_lt hm)]
  /-
    🎉 no goals
  -/


@[to_additive CauchySeq.norm_bddAbove]
lemma CauchySeq.mul_norm_bddAbove {G : Type*} [SeminormedGroup G] {u : ℕ → G}
    (hu : CauchySeq u) : BddAbove (Set.range (fun n ↦ ‖u n‖)) := by
  /-
    G : Type u_4
    inst✝ : SeminormedGroup G
    u : Nat → G
    hu : CauchySeq u
    ⊢ BddAbove (Set.range fun n => Norm.norm (u n))
  -/
  obtain ⟨C, -, hC⟩ := cauchySeq_bdd hu
  /-
    case intro.intro
    G : Type u_4
    inst✝ : SeminormedGroup G
    u : Nat → G
    hu : CauchySeq u
    C : Real
    hC : ∀ (m n : Nat), LT.lt (Dist.dist (u m) (u n)) C
    ⊢ BddAbove (Set.range fun n => Norm.norm (u n))
  -/
  simp_rw [SeminormedGroup.dist_eq] at hC
  have : ∀ n, ‖u n‖ ≤ C + ‖u 0‖ := by
    intro n
    rw [add_comm]
    refine (norm_le_norm_add_norm_div' (u n) (u 0)).trans ?_
    simp [(hC _ _).le]
  /-
    case intro.intro
    G : Type u_4
    inst✝ : SeminormedGroup G
    u : Nat → G
    hu : CauchySeq u
    C : Real
    hC : ∀ (m n : Nat), LT.lt (Norm.norm (HDiv.hDiv (u m) (u n))) C
    this : ∀ (n : Nat), LE.le (Norm.norm (u n)) (HAdd.hAdd C (Norm.norm (u 0)))
    ⊢ BddAbove (Set.range fun n => Norm.norm (u n))
  -/
  rw [bddAbove_def]
  /-
    case intro.intro
    G : Type u_4
    inst✝ : SeminormedGroup G
    u : Nat → G
    hu : CauchySeq u
    C : Real
    hC : ∀ (m n : Nat), LT.lt (Norm.norm (HDiv.hDiv (u m) (u n))) C
    this : ∀ (n : Nat), LE.le (Norm.norm (u n)) (HAdd.hAdd C (Norm.norm (u 0)))
    ⊢ Exists fun x => ∀ (y : Real), Membership.mem (Set.range fun n => Norm.norm ( …
  -/
  exact ⟨C + ‖u 0‖, by simpa using this⟩
  /-
    🎉 no goals
  -/


