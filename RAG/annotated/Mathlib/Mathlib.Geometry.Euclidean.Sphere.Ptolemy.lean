/-- **Ptolemy’s Theorem**. -/
theorem mul_dist_add_mul_dist_eq_mul_dist_of_cospherical {a b c d p : P}
    (h : Cospherical ({a, b, c, d} : Set P)) (hapc : ∠ a p c = π) (hbpd : ∠ b p d = π) :
    dist a b * dist c d + dist b c * dist d a = dist a c * dist b d := by
  /-
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c d p : P
    h : EuclideanGeometry.Cospherical (Insert.insert a (Insert.insert b (Insert.in …
    hapc : Eq (EuclideanGeometry.angle a p c) Real.pi
    hbpd : Eq (EuclideanGeometry.angle b p d) Real.pi
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Dist.dist a b) (Dist.dist c d)) (HMul.hMul (Dist.d …
  -/
  have h' : Cospherical ({a, c, b, d} : Set P) := by rwa [Set.insert_comm c b {d}]
  /-
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c d p : P
    h : EuclideanGeometry.Cospherical (Insert.insert a (Insert.insert b (Insert.in …
    hapc : Eq (EuclideanGeometry.angle a p c) Real.pi
    hbpd : Eq (EuclideanGeometry.angle b p d) Real.pi
    h' : EuclideanGeometry.Cospherical (Insert.insert a (Insert.insert c (Insert.i …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Dist.dist a b) (Dist.dist c d)) (HMul.hMul (Dist.d …
  -/
  have hmul := mul_dist_eq_mul_dist_of_cospherical_of_angle_eq_pi h' hapc hbpd
  /-
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c d p : P
    h : EuclideanGeometry.Cospherical (Insert.insert a (Insert.insert b (Insert.in …
    hapc : Eq (EuclideanGeometry.angle a p c) Real.pi
    hbpd : Eq (EuclideanGeometry.angle b p d) Real.pi
    h' : EuclideanGeometry.Cospherical (Insert.insert a (Insert.insert c (Insert.i …
    hmul : Eq (HMul.hMul (Dist.dist a p) (Dist.dist c p)) (HMul.hMul (Dist.dist b  …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Dist.dist a b) (Dist.dist c d)) (HMul.hMul (Dist.d …
  -/
  have hbp := left_dist_ne_zero_of_angle_eq_pi hbpd
  have h₁ : dist c d = dist c p / dist b p * dist a b := by
    rw [dist_mul_of_eq_angle_of_dist_mul b p a c p d, dist_comm a b]
    · rw [angle_eq_angle_of_angle_eq_pi_of_angle_eq_pi hbpd hapc, angle_comm]
    all_goals field_simp [mul_comm, hmul]
  have h₂ : dist d a = dist a p / dist b p * dist b c := by
    rw [dist_mul_of_eq_angle_of_dist_mul c p b d p a, dist_comm c b]
    · rwa [angle_comm, angle_eq_angle_of_angle_eq_pi_of_angle_eq_pi]; rwa [angle_comm]
    all_goals field_simp [mul_comm, hmul]
  /-
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c d p : P
    h : EuclideanGeometry.Cospherical (Insert.insert a (Insert.insert b (Insert.in …
    hapc : Eq (EuclideanGeometry.angle a p c) Real.pi
    hbpd : Eq (EuclideanGeometry.angle b p d) Real.pi
    h' : EuclideanGeometry.Cospherical (Insert.insert a (Insert.insert c (Insert.i …
    hmul : Eq (HMul.hMul (Dist.dist a p) (Dist.dist c p)) (HMul.hMul (Dist.dist b  …
    hbp : Ne (Dist.dist b p) 0
    h₁ : Eq (Dist.dist c d) (HMul.hMul (HDiv.hDiv (Dist.dist c p) (Dist.dist b p)) …
    h₂ : Eq (Dist.dist d a) (HMul.hMul (HDiv.hDiv (Dist.dist a p) (Dist.dist b p)) …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Dist.dist a b) (Dist.dist c d)) (HMul.hMul (Dist.d …
  -/
  have h₃ : dist d p = dist a p * dist c p / dist b p := by field_simp [mul_comm, hmul]
  /-
    V : Type u_1
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    P : Type u_2
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c d p : P
    h : EuclideanGeometry.Cospherical (Insert.insert a (Insert.insert b (Insert.in …
    hapc : Eq (EuclideanGeometry.angle a p c) Real.pi
    hbpd : Eq (EuclideanGeometry.angle b p d) Real.pi
    h' : EuclideanGeometry.Cospherical (Insert.insert a (Insert.insert c (Insert.i …
    hmul : Eq (HMul.hMul (Dist.dist a p) (Dist.dist c p)) (HMul.hMul (Dist.dist b  …
    hbp : Ne (Dist.dist b p) 0
    h₁ : Eq (Dist.dist c d) (HMul.hMul (HDiv.hDiv (Dist.dist c p) (Dist.dist b p)) …
    h₂ : Eq (Dist.dist d a) (HMul.hMul (HDiv.hDiv (Dist.dist a p) (Dist.dist b p)) …
    h₃ : Eq (Dist.dist d p) (HDiv.hDiv (HMul.hMul (Dist.dist a p) (Dist.dist c p)) …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Dist.dist a b) (Dist.dist c d)) (HMul.hMul (Dist.d …
  -/
  have h₄ : ∀ x y : ℝ, x * (y * x) = x * x * y := fun x y => by rw [mul_left_comm, mul_comm]
  -- takes 450ms, but the "equivalent" simp call leaves some remaining goals
  field_simp [h₁, h₂, dist_eq_add_dist_of_angle_eq_pi hbpd, h₃, hbp, dist_comm a b, h₄, ← sq,
    dist_sq_mul_dist_add_dist_sq_mul_dist b, hapc]


