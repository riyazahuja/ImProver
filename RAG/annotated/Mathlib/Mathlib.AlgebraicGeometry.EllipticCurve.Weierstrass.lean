local macro "map_simp" : tactic =>
  `(tactic| simp only [map_ofNat, map_neg, map_add, map_sub, map_mul, map_pow])


/-- A Weierstrass curve $Y^2 + a_1XY + a_3Y = X^3 + a_2X^2 + a_4X + a_6$ with parameters $a_i$. -/
@[ext]
structure WeierstrassCurve (R : Type u) where
  /-- The `a₁` coefficient of a Weierstrass curve. -/
  a₁ : R
  /-- The `a₂` coefficient of a Weierstrass curve. -/
  a₂ : R
  /-- The `a₃` coefficient of a Weierstrass curve. -/
  a₃ : R
  /-- The `a₄` coefficient of a Weierstrass curve. -/
  a₄ : R
  /-- The `a₆` coefficient of a Weierstrass curve. -/
  a₆ : R


instance instInhabited {R : Type u} [Inhabited R] :
    Inhabited <| WeierstrassCurve R :=
  ⟨⟨default, default, default, default, default⟩⟩


/-- The `b₂` coefficient of a Weierstrass curve. -/
def b₂ : R :=
  W.a₁ ^ 2 + 4 * W.a₂


/-- The `b₄` coefficient of a Weierstrass curve. -/
def b₄ : R :=
  2 * W.a₄ + W.a₁ * W.a₃


/-- The `b₆` coefficient of a Weierstrass curve. -/
def b₆ : R :=
  W.a₃ ^ 2 + 4 * W.a₆


/-- The `b₈` coefficient of a Weierstrass curve. -/
def b₈ : R :=
  W.a₁ ^ 2 * W.a₆ + 4 * W.a₂ * W.a₆ - W.a₁ * W.a₃ * W.a₄ + W.a₂ * W.a₃ ^ 2 - W.a₄ ^ 2


lemma b_relation : 4 * W.b₈ = W.b₂ * W.b₆ - W.b₄ ^ 2 := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    ⊢ Eq (HMul.hMul 4 W.b₈) (HSub.hSub (HMul.hMul W.b₂ W.b₆) (HPow.hPow W.b₄ 2))
  -/
  simp only [b₂, b₄, b₆, b₈]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    ⊢ Eq (HMul.hMul 4 (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HMul.hMul (HPow …
  -/
  ring1
  /-
    🎉 no goals
  -/


/-- The `c₄` coefficient of a Weierstrass curve. -/
def c₄ : R :=
  W.b₂ ^ 2 - 24 * W.b₄


/-- The `c₆` coefficient of a Weierstrass curve. -/
def c₆ : R :=
  -W.b₂ ^ 3 + 36 * W.b₂ * W.b₄ - 216 * W.b₆


/-- The discriminant `Δ` of a Weierstrass curve. If `R` is a field, then this polynomial vanishes
if and only if the cubic curve cut out by this equation is singular. Sometimes only defined up to
sign in the literature; we choose the sign used by the LMFDB. For more discussion, see
[the LMFDB page on discriminants](https://www.lmfdb.org/knowledge/show/ec.discriminant). -/
def Δ : R :=
  -W.b₂ ^ 2 * W.b₈ - 8 * W.b₄ ^ 3 - 27 * W.b₆ ^ 2 + 9 * W.b₂ * W.b₄ * W.b₆


lemma c_relation : 1728 * W.Δ = W.c₄ ^ 3 - W.c₆ ^ 2 := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    ⊢ Eq (HMul.hMul 1728 W.Δ) (HSub.hSub (HPow.hPow W.c₄ 3) (HPow.hPow W.c₆ 2))
  -/
  simp only [b₂, b₄, b₆, b₈, c₄, c₆, Δ]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    ⊢ Eq (HMul.hMul 1728 (HAdd.hAdd (HSub.hSub (HSub.hSub (HMul.hMul (Neg.neg (HPo …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma b₂_of_char_two : W.b₂ = W.a₁ ^ 2 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 2
    ⊢ Eq W.b₂ (HPow.hPow W.a₁ 2)
  -/
  rw [b₂]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 2
    ⊢ Eq (HAdd.hAdd (HPow.hPow W.a₁ 2) (HMul.hMul 4 W.a₂)) (HPow.hPow W.a₁ 2)
  -/
  linear_combination 2 * W.a₂ * CharP.cast_eq_zero R 2
  /-
    🎉 no goals
  -/


lemma b₄_of_char_two : W.b₄ = W.a₁ * W.a₃ := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 2
    ⊢ Eq W.b₄ (HMul.hMul W.a₁ W.a₃)
  -/
  rw [b₄]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 2
    ⊢ Eq (HAdd.hAdd (HMul.hMul 2 W.a₄) (HMul.hMul W.a₁ W.a₃)) (HMul.hMul W.a₁ W.a₃)
  -/
  linear_combination W.a₄ * CharP.cast_eq_zero R 2
  /-
    🎉 no goals
  -/


lemma b₆_of_char_two : W.b₆ = W.a₃ ^ 2 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 2
    ⊢ Eq W.b₆ (HPow.hPow W.a₃ 2)
  -/
  rw [b₆]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 2
    ⊢ Eq (HAdd.hAdd (HPow.hPow W.a₃ 2) (HMul.hMul 4 W.a₆)) (HPow.hPow W.a₃ 2)
  -/
  linear_combination 2 * W.a₆ * CharP.cast_eq_zero R 2
  /-
    🎉 no goals
  -/


lemma b₈_of_char_two :
    W.b₈ = W.a₁ ^ 2 * W.a₆ + W.a₁ * W.a₃ * W.a₄ + W.a₂ * W.a₃ ^ 2 + W.a₄ ^ 2 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 2
    ⊢ Eq W.b₈ (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPow.hPow W.a₁ 2) W.a₆) …
  -/
  rw [b₈]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 2
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HMul.hMul (HPow.hPow W.a₁ 2) …
  -/
  linear_combination (2 * W.a₂ * W.a₆ - W.a₁ * W.a₃ * W.a₄ - W.a₄ ^ 2) * CharP.cast_eq_zero R 2
  /-
    🎉 no goals
  -/


lemma c₄_of_char_two : W.c₄ = W.a₁ ^ 4 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 2
    ⊢ Eq W.c₄ (HPow.hPow W.a₁ 4)
  -/
  rw [c₄, b₂_of_char_two]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 2
    ⊢ Eq (HSub.hSub (HPow.hPow (HPow.hPow W.a₁ 2) 2) (HMul.hMul 24 W.b₄)) (HPow.hP …
  -/
  linear_combination -12 * W.b₄ * CharP.cast_eq_zero R 2
  /-
    🎉 no goals
  -/


lemma c₆_of_char_two : W.c₆ = W.a₁ ^ 6 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 2
    ⊢ Eq W.c₆ (HPow.hPow W.a₁ 6)
  -/
  rw [c₆, b₂_of_char_two]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 2
    ⊢ Eq (HSub.hSub (HAdd.hAdd (Neg.neg (HPow.hPow (HPow.hPow W.a₁ 2) 3)) (HMul.hM …
  -/
  linear_combination (18 * W.a₁ ^ 2 * W.b₄ - 108 * W.b₆ - W.a₁ ^ 6) * CharP.cast_eq_zero R 2
  /-
    🎉 no goals
  -/


lemma Δ_of_char_two : W.Δ = W.a₁ ^ 4 * W.b₈ + W.a₃ ^ 4 + W.a₁ ^ 3 * W.a₃ ^ 3 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 2
    ⊢ Eq W.Δ (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPow.hPow W.a₁ 4) W.b₈) (HPow.hPow  …
  -/
  rw [Δ, b₂_of_char_two, b₄_of_char_two, b₆_of_char_two]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 2
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HMul.hMul (Neg.neg (HPow.hPow (HPow.hPo …
  -/
  linear_combination (-W.a₁ ^ 4 * W.b₈ - 14 * W.a₃ ^ 4) * CharP.cast_eq_zero R 2
  /-
    🎉 no goals
  -/


lemma b_relation_of_char_two : W.b₂ * W.b₆ = W.b₄ ^ 2 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 2
    ⊢ Eq (HMul.hMul W.b₂ W.b₆) (HPow.hPow W.b₄ 2)
  -/
  linear_combination -W.b_relation + 2 * W.b₈ * CharP.cast_eq_zero R 2
  /-
    🎉 no goals
  -/


lemma c_relation_of_char_two : W.c₄ ^ 3 = W.c₆ ^ 2 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 2
    ⊢ Eq (HPow.hPow W.c₄ 3) (HPow.hPow W.c₆ 2)
  -/
  linear_combination -W.c_relation + 864 * W.Δ * CharP.cast_eq_zero R 2
  /-
    🎉 no goals
  -/


lemma b₂_of_char_three : W.b₂ = W.a₁ ^ 2 + W.a₂ := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 3
    ⊢ Eq W.b₂ (HAdd.hAdd (HPow.hPow W.a₁ 2) W.a₂)
  -/
  rw [b₂]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 3
    ⊢ Eq (HAdd.hAdd (HPow.hPow W.a₁ 2) (HMul.hMul 4 W.a₂)) (HAdd.hAdd (HPow.hPow W …
  -/
  linear_combination W.a₂ * CharP.cast_eq_zero R 3
  /-
    🎉 no goals
  -/


lemma b₄_of_char_three : W.b₄ = -W.a₄ + W.a₁ * W.a₃ := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 3
    ⊢ Eq W.b₄ (HAdd.hAdd (Neg.neg W.a₄) (HMul.hMul W.a₁ W.a₃))
  -/
  rw [b₄]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 3
    ⊢ Eq (HAdd.hAdd (HMul.hMul 2 W.a₄) (HMul.hMul W.a₁ W.a₃)) (HAdd.hAdd (Neg.neg  …
  -/
  linear_combination W.a₄ * CharP.cast_eq_zero R 3
  /-
    🎉 no goals
  -/


lemma b₆_of_char_three : W.b₆ = W.a₃ ^ 2 + W.a₆ := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 3
    ⊢ Eq W.b₆ (HAdd.hAdd (HPow.hPow W.a₃ 2) W.a₆)
  -/
  rw [b₆]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 3
    ⊢ Eq (HAdd.hAdd (HPow.hPow W.a₃ 2) (HMul.hMul 4 W.a₆)) (HAdd.hAdd (HPow.hPow W …
  -/
  linear_combination W.a₆ * CharP.cast_eq_zero R 3
  /-
    🎉 no goals
  -/


lemma b₈_of_char_three :
    W.b₈ = W.a₁ ^ 2 * W.a₆ + W.a₂ * W.a₆ - W.a₁ * W.a₃ * W.a₄ + W.a₂ * W.a₃ ^ 2 - W.a₄ ^ 2 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 3
    ⊢ Eq W.b₈ (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HMul.hMul (HPow.hPow W. …
  -/
  rw [b₈]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 3
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HMul.hMul (HPow.hPow W.a₁ 2) …
  -/
  linear_combination W.a₂ * W.a₆ * CharP.cast_eq_zero R 3
  /-
    🎉 no goals
  -/


lemma c₄_of_char_three : W.c₄ = W.b₂ ^ 2 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 3
    ⊢ Eq W.c₄ (HPow.hPow W.b₂ 2)
  -/
  rw [c₄]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 3
    ⊢ Eq (HSub.hSub (HPow.hPow W.b₂ 2) (HMul.hMul 24 W.b₄)) (HPow.hPow W.b₂ 2)
  -/
  linear_combination -8 * W.b₄ * CharP.cast_eq_zero R 3
  /-
    🎉 no goals
  -/


lemma c₆_of_char_three : W.c₆ = -W.b₂ ^ 3 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 3
    ⊢ Eq W.c₆ (Neg.neg (HPow.hPow W.b₂ 3))
  -/
  rw [c₆]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 3
    ⊢ Eq (HSub.hSub (HAdd.hAdd (Neg.neg (HPow.hPow W.b₂ 3)) (HMul.hMul (HMul.hMul  …
  -/
  linear_combination (12 * W.b₂ * W.b₄ - 72 * W.b₆) * CharP.cast_eq_zero R 3
  /-
    🎉 no goals
  -/


lemma Δ_of_char_three : W.Δ = -W.b₂ ^ 2 * W.b₈ - 8 * W.b₄ ^ 3 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 3
    ⊢ Eq W.Δ (HSub.hSub (HMul.hMul (Neg.neg (HPow.hPow W.b₂ 2)) W.b₈) (HMul.hMul 8 …
  -/
  rw [Δ]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 3
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HMul.hMul (Neg.neg (HPow.hPow W.b₂ 2))  …
  -/
  linear_combination (-9 * W.b₆ ^ 2 + 3 * W.b₂ * W.b₄ * W.b₆) * CharP.cast_eq_zero R 3
  /-
    🎉 no goals
  -/


lemma b_relation_of_char_three : W.b₈ = W.b₂ * W.b₆ - W.b₄ ^ 2 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 3
    ⊢ Eq W.b₈ (HSub.hSub (HMul.hMul W.b₂ W.b₆) (HPow.hPow W.b₄ 2))
  -/
  linear_combination W.b_relation - W.b₈ * CharP.cast_eq_zero R 3
  /-
    🎉 no goals
  -/


lemma c_relation_of_char_three : W.c₄ ^ 3 = W.c₆ ^ 2 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 3
    ⊢ Eq (HPow.hPow W.c₄ 3) (HPow.hPow W.c₆ 2)
  -/
  linear_combination -W.c_relation + 576 * W.Δ * CharP.cast_eq_zero R 3
  /-
    🎉 no goals
  -/


/-- The Weierstrass curve mapped over a ring homomorphism `φ : R →+* A`. -/
@[simps]
def map : WeierstrassCurve A :=
  ⟨φ W.a₁, φ W.a₂, φ W.a₃, φ W.a₄, φ W.a₆⟩


/-- The Weierstrass curve base changed to an algebra `A` over `R`. -/
abbrev baseChange [Algebra R A] : WeierstrassCurve A :=
  W.map <| algebraMap R A


@[simp]
lemma map_b₂ : (W.map φ).b₂ = φ W.b₂ := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    ⊢ Eq (W.map φ).b₂ (φ W.b₂)
  -/
  simp only [b₂, map_a₁, map_a₂]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    ⊢ Eq (HAdd.hAdd (HPow.hPow (φ W.a₁) 2) (HMul.hMul 4 (φ W.a₂))) (φ (HAdd.hAdd ( …
  -/
  map_simp
  /-
    🎉 no goals
  -/


@[simp]
lemma map_b₄ : (W.map φ).b₄ = φ W.b₄ := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    ⊢ Eq (W.map φ).b₄ (φ W.b₄)
  -/
  simp only [b₄, map_a₁, map_a₃, map_a₄]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    ⊢ Eq (HAdd.hAdd (HMul.hMul 2 (φ W.a₄)) (HMul.hMul (φ W.a₁) (φ W.a₃))) (φ (HAdd …
  -/
  map_simp
  /-
    🎉 no goals
  -/


@[simp]
lemma map_b₆ : (W.map φ).b₆ = φ W.b₆ := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    ⊢ Eq (W.map φ).b₆ (φ W.b₆)
  -/
  simp only [b₆, map_a₃, map_a₆]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    ⊢ Eq (HAdd.hAdd (HPow.hPow (φ W.a₃) 2) (HMul.hMul 4 (φ W.a₆))) (φ (HAdd.hAdd ( …
  -/
  map_simp
  /-
    🎉 no goals
  -/


@[simp]
lemma map_b₈ : (W.map φ).b₈ = φ W.b₈ := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    ⊢ Eq (W.map φ).b₈ (φ W.b₈)
  -/
  simp only [b₈, map_a₁, map_a₂, map_a₃, map_a₄, map_a₆]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HMul.hMul (HPow.hPow (φ W.a₁ …
  -/
  map_simp
  /-
    🎉 no goals
  -/


@[simp]
lemma map_c₄ : (W.map φ).c₄ = φ W.c₄ := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    ⊢ Eq (W.map φ).c₄ (φ W.c₄)
  -/
  simp only [c₄, map_b₂, map_b₄]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    ⊢ Eq (HSub.hSub (HPow.hPow (φ W.b₂) 2) (HMul.hMul 24 (φ W.b₄))) (φ (HSub.hSub  …
  -/
  map_simp
  /-
    🎉 no goals
  -/


@[simp]
lemma map_c₆ : (W.map φ).c₆ = φ W.c₆ := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    ⊢ Eq (W.map φ).c₆ (φ W.c₆)
  -/
  simp only [c₆, map_b₂, map_b₄, map_b₆]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    ⊢ Eq (HSub.hSub (HAdd.hAdd (Neg.neg (HPow.hPow (φ W.b₂) 3)) (HMul.hMul (HMul.h …
  -/
  map_simp
  /-
    🎉 no goals
  -/


@[simp]
lemma map_Δ : (W.map φ).Δ = φ W.Δ := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    ⊢ Eq (W.map φ).Δ (φ W.Δ)
  -/
  simp only [Δ, map_b₂, map_b₄, map_b₆, map_b₈]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HMul.hMul (Neg.neg (HPow.hPow (φ W.b₂)  …
  -/
  map_simp
  /-
    🎉 no goals
  -/


@[simp]
lemma map_id : W.map (RingHom.id R) = W :=
  rfl


lemma map_map {B : Type w} [CommRing B] (ψ : A →+* B) : (W.map φ).map ψ = W.map (ψ.comp φ) :=
  rfl


@[simp]
lemma map_baseChange {S : Type s} [CommRing S] [Algebra R S] {A : Type v} [CommRing A] [Algebra R A]
    [Algebra S A] [IsScalarTower R S A] {B : Type w} [CommRing B] [Algebra R B] [Algebra S B]
    [IsScalarTower R S B] (ψ : A →ₐ[S] B) : (W.baseChange A).map ψ = W.baseChange B :=
  congr_arg W.map <| ψ.comp_algebraMap_of_tower R


lemma map_injective {φ : R →+* A} (hφ : Function.Injective φ) :
    Function.Injective <| map (φ := φ) := fun _ _ h => by
  /-
    R : Type u
    inst✝¹ : CommRing R
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    hφ : Function.Injective ⇑φ
    x✝¹ x✝ : WeierstrassCurve R
    h : Eq ((fun W => W.map φ) x✝¹) ((fun W => W.map φ) x✝)
    ⊢ Eq x✝¹ x✝
  -/
  rcases mk.inj h with ⟨_, _, _, _, _⟩
  /-
    case intro.intro.intro.intro
    R : Type u
    inst✝¹ : CommRing R
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    hφ : Function.Injective ⇑φ
    x✝¹ x✝ : WeierstrassCurve R
    h : Eq ((fun W => W.map φ) x✝¹) ((fun W => W.map φ) x✝)
    left✝³ : Eq (φ x✝¹.a₁) (φ x✝.a₁)
    left✝² : Eq (φ x✝¹.a₂) (φ x✝.a₂)
    left✝¹ : Eq (φ x✝¹.a₃) (φ x✝.a₃)
    left✝ : Eq (φ x✝¹.a₄) (φ x✝.a₄)
    right✝ : Eq (φ x✝¹.a₆) (φ x✝.a₆)
    ⊢ Eq x✝¹ x✝
  -/
                                   /-
                                     🎉 no goals
                                   -/
                                   /-
                                     🎉 no goals
                                   -/
                                   /-
                                     🎉 no goals
                                   -/
                                   /-
                                     🎉 no goals
                                   -/
  ext <;> apply_fun _ using hφ <;> assumption
                                   /-
                                     🎉 no goals
                                   -/


/-- A cubic polynomial whose discriminant is a multiple of the Weierstrass curve discriminant. If
`W` is an elliptic curve over a field `R` of characteristic different from 2, then its roots over a
splitting field of `R` are precisely the $X$-coordinates of the non-zero 2-torsion points of `W`. -/
def twoTorsionPolynomial : Cubic R :=
  ⟨4, W.b₂, 2 * W.b₄, W.b₆⟩


lemma twoTorsionPolynomial_disc : W.twoTorsionPolynomial.disc = 16 * W.Δ := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    ⊢ Eq W.twoTorsionPolynomial.disc (HMul.hMul 16 W.Δ)
  -/
  simp only [b₂, b₄, b₆, b₈, Δ, twoTorsionPolynomial, Cubic.disc]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HSub.hSub (HMul.hMul (HPow.hPow (HAdd.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma twoTorsionPolynomial_of_char_two : W.twoTorsionPolynomial = ⟨0, W.b₂, 0, W.b₆⟩ := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 2
    ⊢ Eq W.twoTorsionPolynomial { a := 0, b := W.b₂, c := 0, d := W.b₆ }
  -/
  rw [twoTorsionPolynomial]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 2
    ⊢ Eq { a := 4, b := W.b₂, c := HMul.hMul 2 W.b₄, d := W.b₆ } { a := 0, b := W. …
  -/
          /-
            🎉 no goals
          -/
  ext <;> dsimp
          /-
            🎉 no goals
          -/
    /-
      case a
      R : Type u
      inst✝¹ : CommRing R
      W : WeierstrassCurve R
      inst✝ : CharP R 2
      ⊢ Eq 4 0
    -/
  · linear_combination 2 * CharP.cast_eq_zero R 2
    /-
      🎉 no goals
    -/
    /-
      case c
      R : Type u
      inst✝¹ : CommRing R
      W : WeierstrassCurve R
      inst✝ : CharP R 2
      ⊢ Eq (HMul.hMul 2 W.b₄) 0
    -/
  · linear_combination W.b₄ * CharP.cast_eq_zero R 2
    /-
      🎉 no goals
    -/


lemma twoTorsionPolynomial_disc_of_char_two : W.twoTorsionPolynomial.disc = 0 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 2
    ⊢ Eq W.twoTorsionPolynomial.disc 0
  -/
  linear_combination W.twoTorsionPolynomial_disc + 8 * W.Δ * CharP.cast_eq_zero R 2
  /-
    🎉 no goals
  -/


lemma twoTorsionPolynomial_of_char_three : W.twoTorsionPolynomial = ⟨1, W.b₂, -W.b₄, W.b₆⟩ := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 3
    ⊢ Eq W.twoTorsionPolynomial { a := 1, b := W.b₂, c := Neg.neg W.b₄, d := W.b₆ }
  -/
  rw [twoTorsionPolynomial]
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 3
    ⊢ Eq { a := 4, b := W.b₂, c := HMul.hMul 2 W.b₄, d := W.b₆ } { a := 1, b := W. …
  -/
          /-
            🎉 no goals
          -/
  ext <;> dsimp
          /-
            🎉 no goals
          -/
    /-
      case a
      R : Type u
      inst✝¹ : CommRing R
      W : WeierstrassCurve R
      inst✝ : CharP R 3
      ⊢ Eq 4 1
    -/
  · linear_combination CharP.cast_eq_zero R 3
    /-
      🎉 no goals
    -/
    /-
      case c
      R : Type u
      inst✝¹ : CommRing R
      W : WeierstrassCurve R
      inst✝ : CharP R 3
      ⊢ Eq (HMul.hMul 2 W.b₄) (Neg.neg W.b₄)
    -/
  · linear_combination W.b₄ * CharP.cast_eq_zero R 3
    /-
      🎉 no goals
    -/


lemma twoTorsionPolynomial_disc_of_char_three : W.twoTorsionPolynomial.disc = W.Δ := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 3
    ⊢ Eq W.twoTorsionPolynomial.disc W.Δ
  -/
  linear_combination W.twoTorsionPolynomial_disc + 5 * W.Δ * CharP.cast_eq_zero R 3
  /-
    🎉 no goals
  -/


lemma twoTorsionPolynomial_disc_isUnit (hu : IsUnit (2 : R)) :
    IsUnit W.twoTorsionPolynomial.disc ↔ IsUnit W.Δ := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    hu : IsUnit 2
    ⊢ Iff (IsUnit W.twoTorsionPolynomial.disc) (IsUnit W.Δ)
  -/
  rw [twoTorsionPolynomial_disc, IsUnit.mul_iff, show (16 : R) = 2 ^ 4 by norm_num1]
  /-
    R : Type u
    inst✝ : CommRing R
    W : WeierstrassCurve R
    hu : IsUnit 2
    ⊢ Iff (And (IsUnit (HPow.hPow 2 4)) (IsUnit W.Δ)) (IsUnit W.Δ)
  -/
  exact and_iff_right <| hu.pow 4
  /-
    🎉 no goals
  -/

-- TODO: change to `[IsUnit ...]` once #17458 is merged
-- TODO: In this case `IsUnit W.Δ` is just `W.IsElliptic`, consider removing/rephrasing this result

lemma twoTorsionPolynomial_disc_ne_zero [Nontrivial R] (hu : IsUnit (2 : R)) (hΔ : IsUnit W.Δ) :
    W.twoTorsionPolynomial.disc ≠ 0 :=
  ((W.twoTorsionPolynomial_disc_isUnit hu).mpr hΔ).ne_zero


/-- `WeierstrassCurve.IsElliptic` is a typeclass which asserts that a Weierstrass curve is an
elliptic curve: that its discriminant is a unit. Note that this definition is only mathematically
accurate for certain rings whose Picard group has trivial 12-torsion, such as a field or a PID. -/
@[mk_iff]
protected class IsElliptic : Prop where
  isUnit : IsUnit W.Δ


lemma isUnit_Δ : IsUnit W.Δ := IsElliptic.isUnit


/-- The discriminant `Δ'` of an elliptic curve over `R`, which is given as a unit in `R`.
Note that to prove two equal elliptic curves have the same `Δ'`, you need to use `simp_rw`,
as `rw` cannot transfer instance `WeierstrassCurve.IsElliptic` automatically. -/
noncomputable def Δ' : Rˣ := W.isUnit_Δ.unit


/-- The discriminant `Δ'` of an elliptic curve is equal to the
discriminant `Δ` of it as a Weierstrass curve. -/
@[simp]
lemma coe_Δ' : W.Δ' = W.Δ := rfl


/-- The j-invariant `j` of an elliptic curve, which is invariant under isomorphisms over `R`.
Note that to prove two equal elliptic curves have the same `j`, you need to use `simp_rw`,
as `rw` cannot transfer instance `WeierstrassCurve.IsElliptic` automatically. -/
noncomputable def j : R :=
  W.Δ'⁻¹ * W.c₄ ^ 3


/-- A variant of `WeierstrassCurve.j_eq_zero_iff` without assuming a reduced ring. -/
lemma j_eq_zero_iff' : W.j = 0 ↔ W.c₄ ^ 3 = 0 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsElliptic
    ⊢ Iff (Eq W.j 0) (Eq (HPow.hPow W.c₄ 3) 0)
  -/
  rw [j, Units.mul_right_eq_zero]
  /-
    🎉 no goals
  -/


lemma j_eq_zero (h : W.c₄ = 0) : W.j = 0 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsElliptic
    h : Eq W.c₄ 0
    ⊢ Eq W.j 0
  -/
  rw [j_eq_zero_iff', h, zero_pow three_ne_zero]
  /-
    🎉 no goals
  -/


lemma j_eq_zero_iff [IsReduced R] : W.j = 0 ↔ W.c₄ = 0 := by
  /-
    R : Type u
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsElliptic
    inst✝ : IsReduced R
    ⊢ Iff (Eq W.j 0) (Eq W.c₄ 0)
  -/
  rw [j_eq_zero_iff', IsReduced.pow_eq_zero_iff three_ne_zero]
  /-
    🎉 no goals
  -/


lemma j_of_char_two : W.j = W.Δ'⁻¹ * W.a₁ ^ 12 := by
  /-
    R : Type u
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsElliptic
    inst✝ : CharP R 2
    ⊢ Eq W.j (HMul.hMul (↑(Inv.inv W.Δ')) (HPow.hPow W.a₁ 12))
  -/
  rw [j, W.c₄_of_char_two, ← pow_mul]
  /-
    🎉 no goals
  -/


/-- A variant of `WeierstrassCurve.j_eq_zero_iff_of_char_two` without assuming a reduced ring. -/
lemma j_eq_zero_iff_of_char_two' : W.j = 0 ↔ W.a₁ ^ 12 = 0 := by
  /-
    R : Type u
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsElliptic
    inst✝ : CharP R 2
    ⊢ Iff (Eq W.j 0) (Eq (HPow.hPow W.a₁ 12) 0)
  -/
  rw [j_of_char_two, Units.mul_right_eq_zero]
  /-
    🎉 no goals
  -/


lemma j_eq_zero_of_char_two (h : W.a₁ = 0) : W.j = 0 := by
  /-
    R : Type u
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsElliptic
    inst✝ : CharP R 2
    h : Eq W.a₁ 0
    ⊢ Eq W.j 0
  -/
  rw [j_eq_zero_iff_of_char_two', h, zero_pow (Nat.succ_ne_zero _)]
  /-
    🎉 no goals
  -/


lemma j_eq_zero_iff_of_char_two [IsReduced R] : W.j = 0 ↔ W.a₁ = 0 := by
  /-
    R : Type u
    inst✝³ : CommRing R
    W : WeierstrassCurve R
    inst✝² : W.IsElliptic
    inst✝¹ : CharP R 2
    inst✝ : IsReduced R
    ⊢ Iff (Eq W.j 0) (Eq W.a₁ 0)
  -/
  rw [j_eq_zero_iff_of_char_two', IsReduced.pow_eq_zero_iff (Nat.succ_ne_zero _)]
  /-
    🎉 no goals
  -/


lemma j_of_char_three : W.j = W.Δ'⁻¹ * W.b₂ ^ 6 := by
  /-
    R : Type u
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsElliptic
    inst✝ : CharP R 3
    ⊢ Eq W.j (HMul.hMul (↑(Inv.inv W.Δ')) (HPow.hPow W.b₂ 6))
  -/
  rw [j, W.c₄_of_char_three, ← pow_mul]
  /-
    🎉 no goals
  -/


/-- A variant of `WeierstrassCurve.j_eq_zero_iff_of_char_three` without assuming a reduced ring. -/
lemma j_eq_zero_iff_of_char_three' : W.j = 0 ↔ W.b₂ ^ 6 = 0 := by
  /-
    R : Type u
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsElliptic
    inst✝ : CharP R 3
    ⊢ Iff (Eq W.j 0) (Eq (HPow.hPow W.b₂ 6) 0)
  -/
  rw [j_of_char_three, Units.mul_right_eq_zero]
  /-
    🎉 no goals
  -/


lemma j_eq_zero_of_char_three (h : W.b₂ = 0) : W.j = 0 := by
  /-
    R : Type u
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsElliptic
    inst✝ : CharP R 3
    h : Eq W.b₂ 0
    ⊢ Eq W.j 0
  -/
  rw [j_eq_zero_iff_of_char_three', h, zero_pow (Nat.succ_ne_zero _)]
  /-
    🎉 no goals
  -/


lemma j_eq_zero_iff_of_char_three [IsReduced R] : W.j = 0 ↔ W.b₂ = 0 := by
  /-
    R : Type u
    inst✝³ : CommRing R
    W : WeierstrassCurve R
    inst✝² : W.IsElliptic
    inst✝¹ : CharP R 3
    inst✝ : IsReduced R
    ⊢ Iff (Eq W.j 0) (Eq W.b₂ 0)
  -/
  rw [j_eq_zero_iff_of_char_three', IsReduced.pow_eq_zero_iff (Nat.succ_ne_zero _)]
  /-
    🎉 no goals
  -/


lemma twoTorsionPolynomial_disc_ne_zero_of_isElliptic [Nontrivial R] (hu : IsUnit (2 : R)) :
    W.twoTorsionPolynomial.disc ≠ 0 :=
  W.twoTorsionPolynomial_disc_ne_zero hu W.isUnit_Δ


instance : (W.map φ).IsElliptic := by
  /-
    R : Type u
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsElliptic
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    ⊢ (W.map φ).IsElliptic
  -/
  simp only [isElliptic_iff, map_Δ, W.isUnit_Δ.map]
  /-
    🎉 no goals
  -/


set_option linter.docPrime false in
lemma coe_map_Δ' : (W.map φ).Δ' = φ W.Δ' := by
  /-
    R : Type u
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsElliptic
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    ⊢ Eq (↑(W.map φ).Δ') (φ ↑W.Δ')
  -/
  rw [coe_Δ', map_Δ, coe_Δ']
  /-
    🎉 no goals
  -/


set_option linter.docPrime false in
@[simp]
lemma map_Δ' : (W.map φ).Δ' = Units.map φ W.Δ' := by
  /-
    R : Type u
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsElliptic
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    ⊢ Eq (W.map φ).Δ' ((Units.map ↑φ) W.Δ')
  -/
  ext
  /-
    case a
    R : Type u
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsElliptic
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    ⊢ Eq ↑(W.map φ).Δ' ↑((Units.map ↑φ) W.Δ')
  -/
  exact W.coe_map_Δ' φ
  /-
    🎉 no goals
  -/


set_option linter.docPrime false in
lemma coe_inv_map_Δ' : (W.map φ).Δ'⁻¹ = φ ↑W.Δ'⁻¹ := by
  /-
    R : Type u
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsElliptic
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    ⊢ Eq (↑(Inv.inv (W.map φ).Δ')) (φ ↑(Inv.inv W.Δ'))
  -/
  simp
  /-
    🎉 no goals
  -/


set_option linter.docPrime false in
lemma inv_map_Δ' : (W.map φ).Δ'⁻¹ = Units.map φ W.Δ'⁻¹ := by
  /-
    R : Type u
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsElliptic
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    ⊢ Eq (Inv.inv (W.map φ).Δ') ((Units.map ↑φ) (Inv.inv W.Δ'))
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma map_j : (W.map φ).j = φ W.j := by
  /-
    R : Type u
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsElliptic
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    ⊢ Eq (W.map φ).j (φ W.j)
  -/
  rw [j, coe_inv_map_Δ', map_c₄, j, map_mul, map_pow]
  /-
    🎉 no goals
  -/


