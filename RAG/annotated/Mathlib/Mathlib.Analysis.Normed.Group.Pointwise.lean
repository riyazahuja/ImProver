@[to_additive]
theorem Bornology.IsBounded.mul (hs : IsBounded s) (ht : IsBounded t) : IsBounded (s * t) := by
  /-
    E : Type u_1
    inst✝ : SeminormedGroup E
    s t : Set E
    hs : Bornology.IsBounded s
    ht : Bornology.IsBounded t
    ⊢ Bornology.IsBounded (HMul.hMul s t)
  -/
  obtain ⟨Rs, hRs⟩ : ∃ R, ∀ x ∈ s, ‖x‖ ≤ R := hs.exists_norm_le'
  /-
    case intro
    E : Type u_1
    inst✝ : SeminormedGroup E
    s t : Set E
    hs : Bornology.IsBounded s
    ht : Bornology.IsBounded t
    Rs : Real
    hRs : ∀ (x : E), Membership.mem s x → LE.le (Norm.norm x) Rs
    ⊢ Bornology.IsBounded (HMul.hMul s t)
  -/
  obtain ⟨Rt, hRt⟩ : ∃ R, ∀ x ∈ t, ‖x‖ ≤ R := ht.exists_norm_le'
  /-
    case intro.intro
    E : Type u_1
    inst✝ : SeminormedGroup E
    s t : Set E
    hs : Bornology.IsBounded s
    ht : Bornology.IsBounded t
    Rs : Real
    hRs : ∀ (x : E), Membership.mem s x → LE.le (Norm.norm x) Rs
    Rt : Real
    hRt : ∀ (x : E), Membership.mem t x → LE.le (Norm.norm x) Rt
    ⊢ Bornology.IsBounded (HMul.hMul s t)
  -/
  refine isBounded_iff_forall_norm_le'.2 ⟨Rs + Rt, ?_⟩
  /-
    case intro.intro
    E : Type u_1
    inst✝ : SeminormedGroup E
    s t : Set E
    hs : Bornology.IsBounded s
    ht : Bornology.IsBounded t
    Rs : Real
    hRs : ∀ (x : E), Membership.mem s x → LE.le (Norm.norm x) Rs
    Rt : Real
    hRt : ∀ (x : E), Membership.mem t x → LE.le (Norm.norm x) Rt
    ⊢ ∀ (x : E), Membership.mem (HMul.hMul s t) x → LE.le (Norm.norm x) (HAdd.hAdd …
  -/
  rintro z ⟨x, hx, y, hy, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝ : SeminormedGroup E
    s t : Set E
    hs : Bornology.IsBounded s
    ht : Bornology.IsBounded t
    Rs : Real
    hRs : ∀ (x : E), Membership.mem s x → LE.le (Norm.norm x) Rs
    Rt : Real
    hRt : ∀ (x : E), Membership.mem t x → LE.le (Norm.norm x) Rt
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem t y
    ⊢ LE.le (Norm.norm ((fun x1 x2 => HMul.hMul x1 x2) x y)) (HAdd.hAdd Rs Rt)
  -/
  exact norm_mul_le_of_le' (hRs x hx) (hRt y hy)
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Bornology.IsBounded.of_mul (hst : IsBounded (s * t)) : IsBounded s ∨ IsBounded t :=
  AntilipschitzWith.isBounded_of_image2_left _ (fun x => (isometry_mul_right x).antilipschitz) hst


@[to_additive]
theorem Bornology.IsBounded.inv : IsBounded s → IsBounded s⁻¹ := by
  /-
    E : Type u_1
    inst✝ : SeminormedGroup E
    s : Set E
    ⊢ Bornology.IsBounded s → Bornology.IsBounded (Inv.inv s)
  -/
  simp_rw [isBounded_iff_forall_norm_le', ← image_inv_eq_inv, forall_mem_image, norm_inv']
  /-
    E : Type u_1
    inst✝ : SeminormedGroup E
    s : Set E
    ⊢ (Exists fun C => ∀ ⦃x : E⦄, Membership.mem s x → LE.le (Norm.norm x) C) → Ex …
  -/
  exact id
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Bornology.IsBounded.div (hs : IsBounded s) (ht : IsBounded t) : IsBounded (s / t) :=
  div_eq_mul_inv s t ▸ hs.mul ht.inv


@[to_additive (attr := simp)]
theorem infEdist_inv_inv (x : E) (s : Set E) : infEdist x⁻¹ s⁻¹ = infEdist x s := by
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    x : E
    s : Set E
    ⊢ Eq (EMetric.infEdist (Inv.inv x) (Inv.inv s)) (EMetric.infEdist x s)
  -/
  rw [← image_inv_eq_inv, infEdist_image isometry_inv]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem infEdist_inv (x : E) (s : Set E) : infEdist x⁻¹ s = infEdist x s⁻¹ := by
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    x : E
    s : Set E
    ⊢ Eq (EMetric.infEdist (Inv.inv x) s) (EMetric.infEdist x (Inv.inv s))
  -/
  rw [← infEdist_inv_inv, inv_inv]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem ediam_mul_le (x y : Set E) : EMetric.diam (x * y) ≤ EMetric.diam x + EMetric.diam y :=
  (LipschitzOnWith.ediam_image2_le (· * ·) _ _
        (fun _ _ => (isometry_mul_right _).lipschitz.lipschitzOnWith) fun _ _ =>
        (isometry_mul_left _).lipschitz.lipschitzOnWith).trans_eq <|
       /-
         E : Type u_1
         inst✝ : SeminormedCommGroup E
         x y : Set E
         ⊢ Eq (HAdd.hAdd (HMul.hMul (↑1) (EMetric.diam x)) (HMul.hMul (↑1) (EMetric.dia …
       -/
    by simp only [ENNReal.coe_one, one_mul]
       /-
         🎉 no goals
       -/


@[to_additive (attr := simp)]
theorem inv_thickening : (thickening δ s)⁻¹ = thickening δ s⁻¹ := by
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    s : Set E
    ⊢ Eq (Inv.inv (Metric.thickening δ s)) (Metric.thickening δ (Inv.inv s))
  -/
  simp_rw [thickening, ← infEdist_inv]
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    s : Set E
    ⊢ Eq (Inv.inv (setOf fun x => LT.lt (EMetric.infEdist x s) (ENNReal.ofReal δ)) …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem inv_cthickening : (cthickening δ s)⁻¹ = cthickening δ s⁻¹ := by
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    s : Set E
    ⊢ Eq (Inv.inv (Metric.cthickening δ s)) (Metric.cthickening δ (Inv.inv s))
  -/
  simp_rw [cthickening, ← infEdist_inv]
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    s : Set E
    ⊢ Eq (Inv.inv (setOf fun x => LE.le (EMetric.infEdist x s) (ENNReal.ofReal δ)) …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem inv_ball : (ball x δ)⁻¹ = ball x⁻¹ δ := (IsometryEquiv.inv E).preimage_ball x δ


@[to_additive (attr := simp)]
theorem inv_closedBall : (closedBall x δ)⁻¹ = closedBall x⁻¹ δ :=
  (IsometryEquiv.inv E).preimage_closedBall x δ


@[to_additive]
theorem singleton_mul_ball : {x} * ball y δ = ball (x * y) δ := by
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    x y : E
    ⊢ Eq (HMul.hMul (Singleton.singleton x) (Metric.ball y δ)) (Metric.ball (HMul. …
  -/
  simp only [preimage_mul_ball, image_mul_left, singleton_mul, div_inv_eq_mul, mul_comm y x]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem singleton_div_ball : {x} / ball y δ = ball (x / y) δ := by
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    x y : E
    ⊢ Eq (HDiv.hDiv (Singleton.singleton x) (Metric.ball y δ)) (Metric.ball (HDiv. …
  -/
  simp_rw [div_eq_mul_inv, inv_ball, singleton_mul_ball]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem ball_mul_singleton : ball x δ * {y} = ball (x * y) δ := by
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    x y : E
    ⊢ Eq (HMul.hMul (Metric.ball x δ) (Singleton.singleton y)) (Metric.ball (HMul. …
  -/
  rw [mul_comm, singleton_mul_ball, mul_comm y]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem ball_div_singleton : ball x δ / {y} = ball (x / y) δ := by
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    x y : E
    ⊢ Eq (HDiv.hDiv (Metric.ball x δ) (Singleton.singleton y)) (Metric.ball (HDiv. …
  -/
  simp_rw [div_eq_mul_inv, inv_singleton, ball_mul_singleton]
  /-
    🎉 no goals
  -/


@[to_additive]
                                                                 /-
                                                                   E : Type u_1
                                                                   inst✝ : SeminormedCommGroup E
                                                                   δ : Real
                                                                   x : E
                                                                   ⊢ Eq (HMul.hMul (Singleton.singleton x) (Metric.ball 1 δ)) (Metric.ball x δ)
                                                                 -/
theorem singleton_mul_ball_one : {x} * ball 1 δ = ball x δ := by simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[to_additive]
theorem singleton_div_ball_one : {x} / ball 1 δ = ball x δ := by
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    x : E
    ⊢ Eq (HDiv.hDiv (Singleton.singleton x) (Metric.ball 1 δ)) (Metric.ball x δ)
  -/
  rw [singleton_div_ball, div_one]
  /-
    🎉 no goals
  -/


@[to_additive]
                                                                 /-
                                                                   E : Type u_1
                                                                   inst✝ : SeminormedCommGroup E
                                                                   δ : Real
                                                                   x : E
                                                                   ⊢ Eq (HMul.hMul (Metric.ball 1 δ) (Singleton.singleton x)) (Metric.ball x δ)
                                                                 -/
theorem ball_one_mul_singleton : ball 1 δ * {x} = ball x δ := by simp [ball_mul_singleton]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[to_additive]
theorem ball_one_div_singleton : ball 1 δ / {x} = ball x⁻¹ δ := by
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    x : E
    ⊢ Eq (HDiv.hDiv (Metric.ball 1 δ) (Singleton.singleton x)) (Metric.ball (Inv.i …
  -/
  rw [ball_div_singleton, one_div]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem smul_ball_one : x • ball (1 : E) δ = ball x δ := by
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    x : E
    ⊢ Eq (HSMul.hSMul x (Metric.ball 1 δ)) (Metric.ball x δ)
  -/
  rw [smul_ball, smul_eq_mul, mul_one]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp 1100)]
theorem singleton_mul_closedBall : {x} * closedBall y δ = closedBall (x * y) δ := by
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    x y : E
    ⊢ Eq (HMul.hMul (Singleton.singleton x) (Metric.closedBall y δ)) (Metric.close …
  -/
  simp_rw [singleton_mul, ← smul_eq_mul, image_smul, smul_closedBall]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp 1100)]
theorem singleton_div_closedBall : {x} / closedBall y δ = closedBall (x / y) δ := by
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    x y : E
    ⊢ Eq (HDiv.hDiv (Singleton.singleton x) (Metric.closedBall y δ)) (Metric.close …
  -/
  simp_rw [div_eq_mul_inv, inv_closedBall, singleton_mul_closedBall]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp 1100)]
theorem closedBall_mul_singleton : closedBall x δ * {y} = closedBall (x * y) δ := by
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    x y : E
    ⊢ Eq (HMul.hMul (Metric.closedBall x δ) (Singleton.singleton y)) (Metric.close …
  -/
  simp [mul_comm _ {y}, mul_comm y]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp 1100)]
theorem closedBall_div_singleton : closedBall x δ / {y} = closedBall (x / y) δ := by
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    x y : E
    ⊢ Eq (HDiv.hDiv (Metric.closedBall x δ) (Singleton.singleton y)) (Metric.close …
  -/
  simp [div_eq_mul_inv]
  /-
    🎉 no goals
  -/


@[to_additive]
                                                                                   /-
                                                                                     E : Type u_1
                                                                                     inst✝ : SeminormedCommGroup E
                                                                                     δ : Real
                                                                                     x : E
                                                                                     ⊢ Eq (HMul.hMul (Singleton.singleton x) (Metric.closedBall 1 δ)) (Metric.close …
                                                                                   -/
theorem singleton_mul_closedBall_one : {x} * closedBall 1 δ = closedBall x δ := by simp
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


@[to_additive]
theorem singleton_div_closedBall_one : {x} / closedBall 1 δ = closedBall x δ := by
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    x : E
    ⊢ Eq (HDiv.hDiv (Singleton.singleton x) (Metric.closedBall 1 δ)) (Metric.close …
  -/
  rw [singleton_div_closedBall, div_one]
  /-
    🎉 no goals
  -/


@[to_additive]
                                                                                   /-
                                                                                     E : Type u_1
                                                                                     inst✝ : SeminormedCommGroup E
                                                                                     δ : Real
                                                                                     x : E
                                                                                     ⊢ Eq (HMul.hMul (Metric.closedBall 1 δ) (Singleton.singleton x)) (Metric.close …
                                                                                   -/
theorem closedBall_one_mul_singleton : closedBall 1 δ * {x} = closedBall x δ := by simp
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


@[to_additive]
                                                                                     /-
                                                                                       E : Type u_1
                                                                                       inst✝ : SeminormedCommGroup E
                                                                                       δ : Real
                                                                                       x : E
                                                                                       ⊢ Eq (HDiv.hDiv (Metric.closedBall 1 δ) (Singleton.singleton x)) (Metric.close …
                                                                                     -/
theorem closedBall_one_div_singleton : closedBall 1 δ / {x} = closedBall x⁻¹ δ := by simp
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


@[to_additive (attr := simp 1100)]
                                                                              /-
                                                                                E : Type u_1
                                                                                inst✝ : SeminormedCommGroup E
                                                                                δ : Real
                                                                                x : E
                                                                                ⊢ Eq (HSMul.hSMul x (Metric.closedBall 1 δ)) (Metric.closedBall x δ)
                                                                              -/
theorem smul_closedBall_one : x • closedBall (1 : E) δ = closedBall x δ := by simp
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


@[to_additive]
theorem mul_ball_one : s * ball 1 δ = thickening δ s := by
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    s : Set E
    ⊢ Eq (HMul.hMul s (Metric.ball 1 δ)) (Metric.thickening δ s)
  -/
  rw [thickening_eq_biUnion_ball]
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    s : Set E
    ⊢ Eq (HMul.hMul s (Metric.ball 1 δ)) (Set.iUnion fun x => Set.iUnion fun h =>  …
  -/
  convert iUnion₂_mul (fun x (_ : x ∈ s) => {x}) (ball (1 : E) δ)
    /-
      case h.e'_2.h.e'_5
      E : Type u_1
      inst✝ : SeminormedCommGroup E
      δ : Real
      s : Set E
      ⊢ Eq s (Set.iUnion fun i => Set.iUnion fun j => Singleton.singleton i)
    -/
  · exact s.biUnion_of_singleton.symm
    /-
      🎉 no goals
    -/
  /-
    case h.e'_3.h.e'_3.h.f
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    s : Set E
    x✝¹ : E
    x✝ : Membership.mem s x✝¹
    ⊢ Eq (Metric.ball x✝¹ δ) (HMul.hMul (Singleton.singleton x✝¹) (Metric.ball 1 δ))
  -/
  ext x
  /-
    case h.e'_3.h.e'_3.h.f.h
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    s : Set E
    x✝¹ : E
    x✝ : Membership.mem s x✝¹
    x : E
    ⊢ Iff (Membership.mem (Metric.ball x✝¹ δ) x) (Membership.mem (HMul.hMul (Singl …
  -/
  simp_rw [singleton_mul_ball, mul_one]
  /-
    🎉 no goals
  -/


@[to_additive]
                                                           /-
                                                             E : Type u_1
                                                             inst✝ : SeminormedCommGroup E
                                                             δ : Real
                                                             s : Set E
                                                             ⊢ Eq (HDiv.hDiv s (Metric.ball 1 δ)) (Metric.thickening δ s)
                                                           -/
theorem div_ball_one : s / ball 1 δ = thickening δ s := by simp [div_eq_mul_inv, mul_ball_one]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[to_additive]
                                                           /-
                                                             E : Type u_1
                                                             inst✝ : SeminormedCommGroup E
                                                             δ : Real
                                                             s : Set E
                                                             ⊢ Eq (HMul.hMul (Metric.ball 1 δ) s) (Metric.thickening δ s)
                                                           -/
theorem ball_mul_one : ball 1 δ * s = thickening δ s := by rw [mul_comm, mul_ball_one]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[to_additive]
                                                             /-
                                                               E : Type u_1
                                                               inst✝ : SeminormedCommGroup E
                                                               δ : Real
                                                               s : Set E
                                                               ⊢ Eq (HDiv.hDiv (Metric.ball 1 δ) s) (Metric.thickening δ (Inv.inv s))
                                                             -/
theorem ball_div_one : ball 1 δ / s = thickening δ s⁻¹ := by simp [div_eq_mul_inv, ball_mul_one]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[to_additive (attr := simp)]
theorem mul_ball : s * ball x δ = x • thickening δ s := by
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    s : Set E
    x : E
    ⊢ Eq (HMul.hMul s (Metric.ball x δ)) (HSMul.hSMul x (Metric.thickening δ s))
  -/
  rw [← smul_ball_one, mul_smul_comm, mul_ball_one]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
                                                             /-
                                                               E : Type u_1
                                                               inst✝ : SeminormedCommGroup E
                                                               δ : Real
                                                               s : Set E
                                                               x : E
                                                               ⊢ Eq (HDiv.hDiv s (Metric.ball x δ)) (HSMul.hSMul (Inv.inv x) (Metric.thickeni …
                                                             -/
theorem div_ball : s / ball x δ = x⁻¹ • thickening δ s := by simp [div_eq_mul_inv]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[to_additive (attr := simp)]
                                                           /-
                                                             E : Type u_1
                                                             inst✝ : SeminormedCommGroup E
                                                             δ : Real
                                                             s : Set E
                                                             x : E
                                                             ⊢ Eq (HMul.hMul (Metric.ball x δ) s) (HSMul.hSMul x (Metric.thickening δ s))
                                                           -/
theorem ball_mul : ball x δ * s = x • thickening δ s := by rw [mul_comm, mul_ball]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[to_additive (attr := simp)]
                                                             /-
                                                               E : Type u_1
                                                               inst✝ : SeminormedCommGroup E
                                                               δ : Real
                                                               s : Set E
                                                               x : E
                                                               ⊢ Eq (HDiv.hDiv (Metric.ball x δ) s) (HSMul.hSMul x (Metric.thickening δ (Inv. …
                                                             -/
theorem ball_div : ball x δ / s = x • thickening δ s⁻¹ := by simp [div_eq_mul_inv]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[to_additive]
theorem IsCompact.mul_closedBall_one (hs : IsCompact s) (hδ : 0 ≤ δ) :
    s * closedBall (1 : E) δ = cthickening δ s := by
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    s : Set E
    hs : IsCompact s
    hδ : LE.le 0 δ
    ⊢ Eq (HMul.hMul s (Metric.closedBall 1 δ)) (Metric.cthickening δ s)
  -/
  rw [hs.cthickening_eq_biUnion_closedBall hδ]
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    s : Set E
    hs : IsCompact s
    hδ : LE.le 0 δ
    ⊢ Eq (HMul.hMul s (Metric.closedBall 1 δ)) (Set.iUnion fun x => Set.iUnion fun …
  -/
  ext x
  simp only [mem_mul, dist_eq_norm_div, exists_prop, mem_iUnion, mem_closedBall, exists_and_left,
    mem_closedBall_one_iff, ← eq_div_iff_mul_eq'', div_one, exists_eq_right]


@[to_additive]
theorem IsCompact.div_closedBall_one (hs : IsCompact s) (hδ : 0 ≤ δ) :
                                               /-
                                                 E : Type u_1
                                                 inst✝ : SeminormedCommGroup E
                                                 δ : Real
                                                 s : Set E
                                                 hs : IsCompact s
                                                 hδ : LE.le 0 δ
                                                 ⊢ Eq (HDiv.hDiv s (Metric.closedBall 1 δ)) (Metric.cthickening δ s)
                                               -/
    s / closedBall 1 δ = cthickening δ s := by simp [div_eq_mul_inv, hs.mul_closedBall_one hδ]
                                               /-
                                                 🎉 no goals
                                               -/


@[to_additive]
theorem IsCompact.closedBall_one_mul (hs : IsCompact s) (hδ : 0 ≤ δ) :
                                               /-
                                                 E : Type u_1
                                                 inst✝ : SeminormedCommGroup E
                                                 δ : Real
                                                 s : Set E
                                                 hs : IsCompact s
                                                 hδ : LE.le 0 δ
                                                 ⊢ Eq (HMul.hMul (Metric.closedBall 1 δ) s) (Metric.cthickening δ s)
                                               -/
    closedBall 1 δ * s = cthickening δ s := by rw [mul_comm, hs.mul_closedBall_one hδ]
                                               /-
                                                 🎉 no goals
                                               -/


@[to_additive]
theorem IsCompact.closedBall_one_div (hs : IsCompact s) (hδ : 0 ≤ δ) :
    closedBall 1 δ / s = cthickening δ s⁻¹ := by
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    s : Set E
    hs : IsCompact s
    hδ : LE.le 0 δ
    ⊢ Eq (HDiv.hDiv (Metric.closedBall 1 δ) s) (Metric.cthickening δ (Inv.inv s))
  -/
  simp [div_eq_mul_inv, mul_comm, hs.inv.mul_closedBall_one hδ]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem IsCompact.mul_closedBall (hs : IsCompact s) (hδ : 0 ≤ δ) (x : E) :
    s * closedBall x δ = x • cthickening δ s := by
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    s : Set E
    hs : IsCompact s
    hδ : LE.le 0 δ
    x : E
    ⊢ Eq (HMul.hMul s (Metric.closedBall x δ)) (HSMul.hSMul x (Metric.cthickening  …
  -/
  rw [← smul_closedBall_one, mul_smul_comm, hs.mul_closedBall_one hδ]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem IsCompact.div_closedBall (hs : IsCompact s) (hδ : 0 ≤ δ) (x : E) :
    s / closedBall x δ = x⁻¹ • cthickening δ s := by
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    s : Set E
    hs : IsCompact s
    hδ : LE.le 0 δ
    x : E
    ⊢ Eq (HDiv.hDiv s (Metric.closedBall x δ)) (HSMul.hSMul (Inv.inv x) (Metric.ct …
  -/
  simp [div_eq_mul_inv, mul_comm, hs.mul_closedBall hδ]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem IsCompact.closedBall_mul (hs : IsCompact s) (hδ : 0 ≤ δ) (x : E) :
                                                   /-
                                                     E : Type u_1
                                                     inst✝ : SeminormedCommGroup E
                                                     δ : Real
                                                     s : Set E
                                                     hs : IsCompact s
                                                     hδ : LE.le 0 δ
                                                     x : E
                                                     ⊢ Eq (HMul.hMul (Metric.closedBall x δ) s) (HSMul.hSMul x (Metric.cthickening  …
                                                   -/
    closedBall x δ * s = x • cthickening δ s := by rw [mul_comm, hs.mul_closedBall hδ]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[to_additive]
theorem IsCompact.closedBall_div (hs : IsCompact s) (hδ : 0 ≤ δ) (x : E) :
    closedBall x δ * s = x • cthickening δ s := by
  /-
    E : Type u_1
    inst✝ : SeminormedCommGroup E
    δ : Real
    s : Set E
    hs : IsCompact s
    hδ : LE.le 0 δ
    x : E
    ⊢ Eq (HMul.hMul (Metric.closedBall x δ) s) (HSMul.hSMul x (Metric.cthickening  …
  -/
  simp [div_eq_mul_inv, hs.closedBall_mul hδ]
  /-
    🎉 no goals
  -/


