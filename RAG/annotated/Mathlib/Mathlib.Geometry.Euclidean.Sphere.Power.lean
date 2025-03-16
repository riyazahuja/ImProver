theorem mul_norm_eq_abs_sub_sq_norm {x y z : V} (h₁ : ∃ k : ℝ, k ≠ 1 ∧ x + y = k • (x - y))
    (h₂ : ‖z - y‖ = ‖z + y‖) : ‖x - y‖ * ‖x + y‖ = |‖z + y‖ ^ 2 - ‖z - x‖ ^ 2| := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y z : V
    h₁ : Exists fun k => And (Ne k 1) (Eq (HAdd.hAdd x y) (HSMul.hSMul k (HSub.hSu …
    h₂ : Eq (Norm.norm (HSub.hSub z y)) (Norm.norm (HAdd.hAdd z y))
    ⊢ Eq (HMul.hMul (Norm.norm (HSub.hSub x y)) (Norm.norm (HAdd.hAdd x y))) (abs  …
  -/
  obtain ⟨k, hk_ne_one, hk⟩ := h₁
  /-
    case intro.intro
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y z : V
    h₂ : Eq (Norm.norm (HSub.hSub z y)) (Norm.norm (HAdd.hAdd z y))
    k : Real
    hk_ne_one : Ne k 1
    hk : Eq (HAdd.hAdd x y) (HSMul.hSMul k (HSub.hSub x y))
    ⊢ Eq (HMul.hMul (Norm.norm (HSub.hSub x y)) (Norm.norm (HAdd.hAdd x y))) (abs  …
  -/
  let r := (k - 1)⁻¹ * (k + 1)
  have hxy : x = r • y := by
    rw [← smul_smul, eq_inv_smul_iff₀ (sub_ne_zero.mpr hk_ne_one), ← sub_eq_zero]
    calc
      (k - 1) • x - (k + 1) • y = k • x - x - (k • y + y) := by
        simp_rw [sub_smul, add_smul, one_smul]
      _ = k • x - k • y - (x + y) := by simp_rw [← sub_sub, sub_right_comm]
      _ = k • (x - y) - (x + y) := by rw [← smul_sub k x y]
      _ = 0 := sub_eq_zero.mpr hk.symm
  have hzy : ⟪z, y⟫ = 0 := by
    rwa [inner_eq_zero_iff_angle_eq_pi_div_two, ← norm_add_eq_norm_sub_iff_angle_eq_pi_div_two,
      eq_comm]
  /-
    case intro.intro
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y z : V
    h₂ : Eq (Norm.norm (HSub.hSub z y)) (Norm.norm (HAdd.hAdd z y))
    k : Real
    hk_ne_one : Ne k 1
    hk : Eq (HAdd.hAdd x y) (HSMul.hSMul k (HSub.hSub x y))
    r : Real := HMul.hMul (Inv.inv (HSub.hSub k 1)) (HAdd.hAdd k 1)
    hxy : Eq x (HSMul.hSMul r y)
    hzy : Eq (Inner.inner z y) 0
    ⊢ Eq (HMul.hMul (Norm.norm (HSub.hSub x y)) (Norm.norm (HAdd.hAdd x y))) (abs  …
  -/
  have hzx : ⟪z, x⟫ = 0 := by rw [hxy, inner_smul_right, hzy, mul_zero]
  calc
    ‖x - y‖ * ‖x + y‖ = ‖(r - 1) • y‖ * ‖(r + 1) • y‖ := by simp [sub_smul, add_smul, hxy]
    _ = ‖r - 1‖ * ‖y‖ * (‖r + 1‖ * ‖y‖) := by simp_rw [norm_smul]
    _ = ‖r - 1‖ * ‖r + 1‖ * ‖y‖ ^ 2 := by ring
    _ = |(r - 1) * (r + 1) * ‖y‖ ^ 2| := by simp [abs_mul]
    _ = |r ^ 2 * ‖y‖ ^ 2 - ‖y‖ ^ 2| := by ring_nf
    _ = |‖x‖ ^ 2 - ‖y‖ ^ 2| := by simp [hxy, norm_smul, mul_pow, sq_abs]
    _ = |‖z + y‖ ^ 2 - ‖z - x‖ ^ 2| := by
      simp [norm_add_sq_real, norm_sub_sq_real, hzy, hzx, abs_sub_comm]


/-- If `P` is a point on the line `AB` and `Q` is equidistant from `A` and `B`, then
`AP * BP = abs (BQ ^ 2 - PQ ^ 2)`. -/
theorem mul_dist_eq_abs_sub_sq_dist {a b p q : P} (hp : ∃ k : ℝ, k ≠ 1 ∧ b -ᵥ p = k • (a -ᵥ p))
    (hq : dist a q = dist b q) : dist a p * dist b p = |dist b q ^ 2 - dist p q ^ 2| := by
  /-
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b p q : P
    hp : Exists fun k => And (Ne k 1) (Eq (VSub.vsub b p) (HSMul.hSMul k (VSub.vsu …
    hq : Eq (Dist.dist a q) (Dist.dist b q)
    ⊢ Eq (HMul.hMul (Dist.dist a p) (Dist.dist b p)) (abs (HSub.hSub (HPow.hPow (D …
  -/
  let m : P := midpoint ℝ a b
  /-
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b p q : P
    hp : Exists fun k => And (Ne k 1) (Eq (VSub.vsub b p) (HSMul.hSMul k (VSub.vsu …
    hq : Eq (Dist.dist a q) (Dist.dist b q)
    m : P := midpoint Real a b
    ⊢ Eq (HMul.hMul (Dist.dist a p) (Dist.dist b p)) (abs (HSub.hSub (HPow.hPow (D …
  -/
  have h1 := vsub_sub_vsub_cancel_left a p m
  /-
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b p q : P
    hp : Exists fun k => And (Ne k 1) (Eq (VSub.vsub b p) (HSMul.hSMul k (VSub.vsu …
    hq : Eq (Dist.dist a q) (Dist.dist b q)
    m : P := midpoint Real a b
    h1 : Eq (HSub.hSub (VSub.vsub m p) (VSub.vsub m a)) (VSub.vsub a p)
    ⊢ Eq (HMul.hMul (Dist.dist a p) (Dist.dist b p)) (abs (HSub.hSub (HPow.hPow (D …
  -/
  have h2 := vsub_sub_vsub_cancel_left p q m
  /-
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b p q : P
    hp : Exists fun k => And (Ne k 1) (Eq (VSub.vsub b p) (HSMul.hSMul k (VSub.vsu …
    hq : Eq (Dist.dist a q) (Dist.dist b q)
    m : P := midpoint Real a b
    h1 : Eq (HSub.hSub (VSub.vsub m p) (VSub.vsub m a)) (VSub.vsub a p)
    h2 : Eq (HSub.hSub (VSub.vsub m q) (VSub.vsub m p)) (VSub.vsub p q)
    ⊢ Eq (HMul.hMul (Dist.dist a p) (Dist.dist b p)) (abs (HSub.hSub (HPow.hPow (D …
  -/
  have h3 := vsub_sub_vsub_cancel_left a q m
  have h : ∀ r, b -ᵥ r = m -ᵥ r + (m -ᵥ a) := fun r => by
    rw [midpoint_vsub_left, ← right_vsub_midpoint, add_comm, vsub_add_vsub_cancel]
  /-
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b p q : P
    hp : Exists fun k => And (Ne k 1) (Eq (VSub.vsub b p) (HSMul.hSMul k (VSub.vsu …
    hq : Eq (Dist.dist a q) (Dist.dist b q)
    m : P := midpoint Real a b
    h1 : Eq (HSub.hSub (VSub.vsub m p) (VSub.vsub m a)) (VSub.vsub a p)
    h2 : Eq (HSub.hSub (VSub.vsub m q) (VSub.vsub m p)) (VSub.vsub p q)
    h3 : Eq (HSub.hSub (VSub.vsub m q) (VSub.vsub m a)) (VSub.vsub a q)
    h : ∀ (r : P), Eq (VSub.vsub b r) (HAdd.hAdd (VSub.vsub m r) (VSub.vsub m a))
    ⊢ Eq (HMul.hMul (Dist.dist a p) (Dist.dist b p)) (abs (HSub.hSub (HPow.hPow (D …
  -/
  iterate 4 rw [dist_eq_norm_vsub V]
  /-
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b p q : P
    hp : Exists fun k => And (Ne k 1) (Eq (VSub.vsub b p) (HSMul.hSMul k (VSub.vsu …
    hq : Eq (Dist.dist a q) (Dist.dist b q)
    m : P := midpoint Real a b
    h1 : Eq (HSub.hSub (VSub.vsub m p) (VSub.vsub m a)) (VSub.vsub a p)
    h2 : Eq (HSub.hSub (VSub.vsub m q) (VSub.vsub m p)) (VSub.vsub p q)
    h3 : Eq (HSub.hSub (VSub.vsub m q) (VSub.vsub m a)) (VSub.vsub a q)
    h : ∀ (r : P), Eq (VSub.vsub b r) (HAdd.hAdd (VSub.vsub m r) (VSub.vsub m a))
    ⊢ Eq (HMul.hMul (Norm.norm (VSub.vsub a p)) (Norm.norm (VSub.vsub b p))) (abs  …
  -/
  rw [← h1, ← h2, h, h]
  /-
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b p q : P
    hp : Exists fun k => And (Ne k 1) (Eq (VSub.vsub b p) (HSMul.hSMul k (VSub.vsu …
    hq : Eq (Dist.dist a q) (Dist.dist b q)
    m : P := midpoint Real a b
    h1 : Eq (HSub.hSub (VSub.vsub m p) (VSub.vsub m a)) (VSub.vsub a p)
    h2 : Eq (HSub.hSub (VSub.vsub m q) (VSub.vsub m p)) (VSub.vsub p q)
    h3 : Eq (HSub.hSub (VSub.vsub m q) (VSub.vsub m a)) (VSub.vsub a q)
    h : ∀ (r : P), Eq (VSub.vsub b r) (HAdd.hAdd (VSub.vsub m r) (VSub.vsub m a))
    ⊢ Eq (HMul.hMul (Norm.norm (HSub.hSub (VSub.vsub m p) (VSub.vsub m a))) (Norm. …
  -/
  rw [← h1, h] at hp
  /-
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b p q : P
    hq : Eq (Dist.dist a q) (Dist.dist b q)
    m : P := midpoint Real a b
    hp : Exists fun k => And (Ne k 1) (Eq (HAdd.hAdd (VSub.vsub m p) (VSub.vsub m  …
    h1 : Eq (HSub.hSub (VSub.vsub m p) (VSub.vsub m a)) (VSub.vsub a p)
    h2 : Eq (HSub.hSub (VSub.vsub m q) (VSub.vsub m p)) (VSub.vsub p q)
    h3 : Eq (HSub.hSub (VSub.vsub m q) (VSub.vsub m a)) (VSub.vsub a q)
    h : ∀ (r : P), Eq (VSub.vsub b r) (HAdd.hAdd (VSub.vsub m r) (VSub.vsub m a))
    ⊢ Eq (HMul.hMul (Norm.norm (HSub.hSub (VSub.vsub m p) (VSub.vsub m a))) (Norm. …
  -/
  rw [dist_eq_norm_vsub V a q, dist_eq_norm_vsub V b q, ← h3, h] at hq
  /-
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b p q : P
    m : P := midpoint Real a b
    hq : Eq (Norm.norm (HSub.hSub (VSub.vsub m q) (VSub.vsub m a))) (Norm.norm (HA …
    hp : Exists fun k => And (Ne k 1) (Eq (HAdd.hAdd (VSub.vsub m p) (VSub.vsub m  …
    h1 : Eq (HSub.hSub (VSub.vsub m p) (VSub.vsub m a)) (VSub.vsub a p)
    h2 : Eq (HSub.hSub (VSub.vsub m q) (VSub.vsub m p)) (VSub.vsub p q)
    h3 : Eq (HSub.hSub (VSub.vsub m q) (VSub.vsub m a)) (VSub.vsub a q)
    h : ∀ (r : P), Eq (VSub.vsub b r) (HAdd.hAdd (VSub.vsub m r) (VSub.vsub m a))
    ⊢ Eq (HMul.hMul (Norm.norm (HSub.hSub (VSub.vsub m p) (VSub.vsub m a))) (Norm. …
  -/
  exact mul_norm_eq_abs_sub_sq_norm hp hq
  /-
    🎉 no goals
  -/


/-- If `A`, `B`, `C`, `D` are cospherical and `P` is on both lines `AB` and `CD`, then
`AP * BP = CP * DP`. -/
theorem mul_dist_eq_mul_dist_of_cospherical {a b c d p : P} (h : Cospherical ({a, b, c, d} : Set P))
    (hapb : ∃ k₁ : ℝ, k₁ ≠ 1 ∧ b -ᵥ p = k₁ • (a -ᵥ p))
    (hcpd : ∃ k₂ : ℝ, k₂ ≠ 1 ∧ d -ᵥ p = k₂ • (c -ᵥ p)) :
    dist a p * dist b p = dist c p * dist d p := by
  /-
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c d p : P
    h : EuclideanGeometry.Cospherical (Insert.insert a (Insert.insert b (Insert.in …
    hapb : Exists fun k₁ => And (Ne k₁ 1) (Eq (VSub.vsub b p) (HSMul.hSMul k₁ (VSu …
    hcpd : Exists fun k₂ => And (Ne k₂ 1) (Eq (VSub.vsub d p) (HSMul.hSMul k₂ (VSu …
    ⊢ Eq (HMul.hMul (Dist.dist a p) (Dist.dist b p)) (HMul.hMul (Dist.dist c p) (D …
  -/
  obtain ⟨q, r, h'⟩ := (cospherical_def {a, b, c, d}).mp h
  /-
    case intro.intro
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c d p : P
    h : EuclideanGeometry.Cospherical (Insert.insert a (Insert.insert b (Insert.in …
    hapb : Exists fun k₁ => And (Ne k₁ 1) (Eq (VSub.vsub b p) (HSMul.hSMul k₁ (VSu …
    hcpd : Exists fun k₂ => And (Ne k₂ 1) (Eq (VSub.vsub d p) (HSMul.hSMul k₂ (VSu …
    q : P
    r : Real
    h' : ∀ (p : P), Membership.mem (Insert.insert a (Insert.insert b (Insert.inser …
    ⊢ Eq (HMul.hMul (Dist.dist a p) (Dist.dist b p)) (HMul.hMul (Dist.dist c p) (D …
  -/
  obtain ⟨ha, hb, hc, hd⟩ := h' a (by simp), h' b (by simp), h' c (by simp), h' d (by simp)
  /-
    case intro.intro
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c d p : P
    h : EuclideanGeometry.Cospherical (Insert.insert a (Insert.insert b (Insert.in …
    hapb : Exists fun k₁ => And (Ne k₁ 1) (Eq (VSub.vsub b p) (HSMul.hSMul k₁ (VSu …
    hcpd : Exists fun k₂ => And (Ne k₂ 1) (Eq (VSub.vsub d p) (HSMul.hSMul k₂ (VSu …
    q : P
    r : Real
    h' : ∀ (p : P), Membership.mem (Insert.insert a (Insert.insert b (Insert.inser …
    ha : Eq (Dist.dist a q) r
    hb : Eq (Dist.dist b q) r
    hc : Eq (Dist.dist c q) r
    hd : Eq (Dist.dist d q) r
    ⊢ Eq (HMul.hMul (Dist.dist a p) (Dist.dist b p)) (HMul.hMul (Dist.dist c p) (D …
  -/
  rw [← hd] at hc
  /-
    case intro.intro
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c d p : P
    h : EuclideanGeometry.Cospherical (Insert.insert a (Insert.insert b (Insert.in …
    hapb : Exists fun k₁ => And (Ne k₁ 1) (Eq (VSub.vsub b p) (HSMul.hSMul k₁ (VSu …
    hcpd : Exists fun k₂ => And (Ne k₂ 1) (Eq (VSub.vsub d p) (HSMul.hSMul k₂ (VSu …
    q : P
    r : Real
    h' : ∀ (p : P), Membership.mem (Insert.insert a (Insert.insert b (Insert.inser …
    ha : Eq (Dist.dist a q) r
    hb : Eq (Dist.dist b q) r
    hc : Eq (Dist.dist c q) (Dist.dist d q)
    hd : Eq (Dist.dist d q) r
    ⊢ Eq (HMul.hMul (Dist.dist a p) (Dist.dist b p)) (HMul.hMul (Dist.dist c p) (D …
  -/
  rw [← hb] at ha
  /-
    case intro.intro
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c d p : P
    h : EuclideanGeometry.Cospherical (Insert.insert a (Insert.insert b (Insert.in …
    hapb : Exists fun k₁ => And (Ne k₁ 1) (Eq (VSub.vsub b p) (HSMul.hSMul k₁ (VSu …
    hcpd : Exists fun k₂ => And (Ne k₂ 1) (Eq (VSub.vsub d p) (HSMul.hSMul k₂ (VSu …
    q : P
    r : Real
    h' : ∀ (p : P), Membership.mem (Insert.insert a (Insert.insert b (Insert.inser …
    ha : Eq (Dist.dist a q) (Dist.dist b q)
    hb : Eq (Dist.dist b q) r
    hc : Eq (Dist.dist c q) (Dist.dist d q)
    hd : Eq (Dist.dist d q) r
    ⊢ Eq (HMul.hMul (Dist.dist a p) (Dist.dist b p)) (HMul.hMul (Dist.dist c p) (D …
  -/
  rw [mul_dist_eq_abs_sub_sq_dist hapb ha, hb, mul_dist_eq_abs_sub_sq_dist hcpd hc, hd]
  /-
    🎉 no goals
  -/


/-- **Intersecting Chords Theorem**. -/
theorem mul_dist_eq_mul_dist_of_cospherical_of_angle_eq_pi {a b c d p : P}
    (h : Cospherical ({a, b, c, d} : Set P)) (hapb : ∠ a p b = π) (hcpd : ∠ c p d = π) :
    dist a p * dist b p = dist c p * dist d p := by
  /-
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c d p : P
    h : EuclideanGeometry.Cospherical (Insert.insert a (Insert.insert b (Insert.in …
    hapb : Eq (EuclideanGeometry.angle a p b) Real.pi
    hcpd : Eq (EuclideanGeometry.angle c p d) Real.pi
    ⊢ Eq (HMul.hMul (Dist.dist a p) (Dist.dist b p)) (HMul.hMul (Dist.dist c p) (D …
  -/
  obtain ⟨-, k₁, _, hab⟩ := angle_eq_pi_iff.mp hapb
  /-
    case intro.intro.intro
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c d p : P
    h : EuclideanGeometry.Cospherical (Insert.insert a (Insert.insert b (Insert.in …
    hapb : Eq (EuclideanGeometry.angle a p b) Real.pi
    hcpd : Eq (EuclideanGeometry.angle c p d) Real.pi
    k₁ : Real
    left✝ : LT.lt k₁ 0
    hab : Eq (VSub.vsub b p) (HSMul.hSMul k₁ (VSub.vsub a p))
    ⊢ Eq (HMul.hMul (Dist.dist a p) (Dist.dist b p)) (HMul.hMul (Dist.dist c p) (D …
  -/
  obtain ⟨-, k₂, _, hcd⟩ := angle_eq_pi_iff.mp hcpd
  /-
    case intro.intro.intro.intro.intro.intro
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c d p : P
    h : EuclideanGeometry.Cospherical (Insert.insert a (Insert.insert b (Insert.in …
    hapb : Eq (EuclideanGeometry.angle a p b) Real.pi
    hcpd : Eq (EuclideanGeometry.angle c p d) Real.pi
    k₁ : Real
    left✝¹ : LT.lt k₁ 0
    hab : Eq (VSub.vsub b p) (HSMul.hSMul k₁ (VSub.vsub a p))
    k₂ : Real
    left✝ : LT.lt k₂ 0
    hcd : Eq (VSub.vsub d p) (HSMul.hSMul k₂ (VSub.vsub c p))
    ⊢ Eq (HMul.hMul (Dist.dist a p) (Dist.dist b p)) (HMul.hMul (Dist.dist c p) (D …
  -/
  exact mul_dist_eq_mul_dist_of_cospherical h ⟨k₁, by linarith, hab⟩ ⟨k₂, by linarith, hcd⟩
  /-
    🎉 no goals
  -/


/-- **Intersecting Secants Theorem**. -/
theorem mul_dist_eq_mul_dist_of_cospherical_of_angle_eq_zero {a b c d p : P}
    (h : Cospherical ({a, b, c, d} : Set P)) (hab : a ≠ b) (hcd : c ≠ d) (hapb : ∠ a p b = 0)
    (hcpd : ∠ c p d = 0) : dist a p * dist b p = dist c p * dist d p := by
  /-
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c d p : P
    h : EuclideanGeometry.Cospherical (Insert.insert a (Insert.insert b (Insert.in …
    hab : Ne a b
    hcd : Ne c d
    hapb : Eq (EuclideanGeometry.angle a p b) 0
    hcpd : Eq (EuclideanGeometry.angle c p d) 0
    ⊢ Eq (HMul.hMul (Dist.dist a p) (Dist.dist b p)) (HMul.hMul (Dist.dist c p) (D …
  -/
  obtain ⟨-, k₁, -, hab₁⟩ := angle_eq_zero_iff.mp hapb
  /-
    case intro.intro.intro
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c d p : P
    h : EuclideanGeometry.Cospherical (Insert.insert a (Insert.insert b (Insert.in …
    hab : Ne a b
    hcd : Ne c d
    hapb : Eq (EuclideanGeometry.angle a p b) 0
    hcpd : Eq (EuclideanGeometry.angle c p d) 0
    k₁ : Real
    hab₁ : Eq (VSub.vsub b p) (HSMul.hSMul k₁ (VSub.vsub a p))
    ⊢ Eq (HMul.hMul (Dist.dist a p) (Dist.dist b p)) (HMul.hMul (Dist.dist c p) (D …
  -/
  obtain ⟨-, k₂, -, hcd₁⟩ := angle_eq_zero_iff.mp hcpd
  /-
    case intro.intro.intro.intro.intro.intro
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c d p : P
    h : EuclideanGeometry.Cospherical (Insert.insert a (Insert.insert b (Insert.in …
    hab : Ne a b
    hcd : Ne c d
    hapb : Eq (EuclideanGeometry.angle a p b) 0
    hcpd : Eq (EuclideanGeometry.angle c p d) 0
    k₁ : Real
    hab₁ : Eq (VSub.vsub b p) (HSMul.hSMul k₁ (VSub.vsub a p))
    k₂ : Real
    hcd₁ : Eq (VSub.vsub d p) (HSMul.hSMul k₂ (VSub.vsub c p))
    ⊢ Eq (HMul.hMul (Dist.dist a p) (Dist.dist b p)) (HMul.hMul (Dist.dist c p) (D …
  -/
  refine mul_dist_eq_mul_dist_of_cospherical h ⟨k₁, ?_, hab₁⟩ ⟨k₂, ?_, hcd₁⟩ <;> by_contra hnot <;>
    /-
      case intro.intro.intro.intro.intro.intro.refine_1
      V : Type u_1
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      P : Type u_2
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      a b c d p : P
      h : EuclideanGeometry.Cospherical (Insert.insert a (Insert.insert b (Insert.in …
      hab : Ne a b
      hcd : Ne c d
      hapb : Eq (EuclideanGeometry.angle a p b) 0
      hcpd : Eq (EuclideanGeometry.angle c p d) 0
      k₁ : Real
      hab₁ : Eq (VSub.vsub b p) (HSMul.hSMul k₁ (VSub.vsub a p))
      k₂ : Real
      hcd₁ : Eq (VSub.vsub d p) (HSMul.hSMul k₂ (VSub.vsub c p))
      hnot : Eq k₁ 1
      ⊢ False
    -/
    simp_all only [Classical.not_not, one_smul]
  /-
    case intro.intro.intro.intro.intro.intro.refine_1
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c d p : P
    h : EuclideanGeometry.Cospherical (Insert.insert a (Insert.insert b (Insert.in …
    hab : Ne a b
    hcd : Ne c d
    hapb : Eq (EuclideanGeometry.angle a p b) 0
    hcpd : Eq (EuclideanGeometry.angle c p d) 0
    k₁ k₂ : Real
    hab₁ : Eq (VSub.vsub b p) (VSub.vsub a p)
    hcd₁ : Eq (VSub.vsub d p) (HSMul.hSMul k₂ (VSub.vsub c p))
    hnot : Eq k₁ 1
    ⊢ False
  -/
  exacts [hab (vsub_left_cancel hab₁).symm, hcd (vsub_left_cancel hcd₁).symm]
  /-
    🎉 no goals
  -/


